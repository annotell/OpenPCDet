from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import pickle
import time
from typing import List

import numpy as np
from annotell.apiclients.scene_input_api_client import SceneInputApiClient
from kognic.filestorage.filestorage import FileId
from kognic.filestorage.resource_parser import parse_file_id
from kognic.io.model.calibration.lidar.lidar_calibration import LidarCalibration
from kognic.judgement_shapes.cube_3d import Cube3D
from kognic.potree.potree import PointAttributes
from kognic.potree_filestorage.potree_filestorage import (
    MissingAttributeError,
    PotreeFileStorage,
)
from kognic.projection.projection import LidarProjector, get_camera_projector
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from urllib3.exceptions import ProtocolError
import yaml


potree_fs = PotreeFileStorage()
LIDAR_INFO = {}


class DatasetLoader:
    def __init__(self, config=None, save_dir=None, filter_pc=False):
        if config is not None:
            self.config_path = config
            self.scene = SceneInputApiClient(env="production")
            with open(config) as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)
            self.save_dir = os.path.join(
                self.config["dataset_root"], self.config["dataset_name"]
            )
            self.filter_pc = self.config.get("filter_pc", False)
            self.transform2LCS = self.config.get("transform2LCS", False)
        else:
            if save_dir is None:
                raise ValueError("save_dir must be provided if config is not provided")
            self.save_dir = save_dir
            self.filter_pc = filter_pc
        self.save_dir_pcs = os.path.join(self.save_dir, "pcs")
        self.save_dir_annos = os.path.join(self.save_dir, "annos")
        self.count = 0
        self.start_time = time.time()
        # fetch all files in the dirs
        self.existing_pcs = glob.glob(self.save_dir_pcs + "/*.npz")
        self.existing_annos = glob.glob(self.save_dir_annos + "/*.pickle")
        # truncate at the second _
        self.existing_pcs = [
            "_".join(os.path.basename(pc).split("_")[:2]) for pc in self.existing_pcs
        ]
        self.existing_annos = [
            "_".join(os.path.basename(anno).split("_")[:2])
            for anno in self.existing_annos
        ]

    def filter_pc_single(self, pc, camera_projectors, lidar_projector):
        idxes_all = []
        for cam_projector in camera_projectors:
            _, idxes = camera_projectors[cam_projector].project_lidar_to_image(
                pc[:, 0:3], lidar_projector=lidar_projector
            )
            idxes_all.append(idxes)
        idxes_all = np.concatenate(idxes_all)
        pc = pc[idxes_all]
        pc[:, :3] = lidar_projector.transform_to_reference(pc[:, :3])
        return pc

    # In these Orion projects, cuboids were only annotated in the region of the point cloud that was visible in the camera images.
    # so we need to filter the point cloud to only include the visible region.
    def filter_pcs(self, pc, calibrations, lidar_sensors, input_internal_id):
        camera_projectors = {}
        lidar_projectors = {}
        for sensor_name, calibration in calibrations.items():
            if isinstance(calibration, LidarCalibration):
                lidar_projectors[sensor_name] = LidarProjector(calibration)
            else:
                camera_projectors[sensor_name] = get_camera_projector(calibration)
        is_multi_lidar = len(lidar_sensors) > 1
        if is_multi_lidar:
            # split the point cloud into multiple point clouds, one for each lidar, stored in column 3 of the point cloud
            pcs = []
            lidar_indices = lidar_sensors.values()
            for idx in lidar_indices:
                pcs.append(pc[pc[:, 4] == idx])
        else:
            pcs = [pc]
        # project each point cloud to each camera image and keep only the points that are visible in at least one camera image
        pc_filtered = []
        for pc in pcs:
            # find correct lidar projector
            if is_multi_lidar:
                lidar_index = pc[0, 4]
                lidar_name = [k for k, v in lidar_sensors.items() if v == lidar_index][
                    0
                ]
                lidar_projector = lidar_projectors[lidar_name]
            else:
                lidar_projector = list(lidar_projectors.values())[0]
            pc = self.filter_pc_single(pc, camera_projectors, lidar_projector)
            pc_filtered.append(pc)
        pcs_filtered = np.concatenate(pc_filtered, axis=0)
        # remove duplicate points
        pcs_filtered = np.unique(pcs_filtered, axis=0)

        return pcs_filtered

    def read_pointcloud_from_bucket(
        self, file_id: FileId, is_multilidar: bool = False
    ) -> np.ndarray:
        num_retries = 3
        for i in range(num_retries):
            try:
                if is_multilidar:
                    attributes = [
                        PointAttributes.POSITION_CARTESIAN,
                        PointAttributes.INTENSITY,
                        PointAttributes.SOURCE_ID,
                    ]
                else:
                    attributes = [
                        PointAttributes.POSITION_CARTESIAN,
                        PointAttributes.INTENSITY,
                    ]
                return potree_fs.get_potree_content(file_id, attributes)
            except ProtocolError as e:
                if i == num_retries - 1:
                    raise Exception(
                        f"Failed to read point cloud from fid: {file_id}"
                    ) from e
            except MissingAttributeError as e:
                raise MissingAttributeError(
                    f"Failed to read attributes {attributes} in potree from fid: {file_id}"
                ) from e
            except Exception as e:
                raise Exception(
                    f"Failed to read point cloud from fid: {file_id}"
                ) from e

    def pc_split_and_save(self, pc, filename, is_multilidar):
        try:
            if is_multilidar:
                # separate into multiple pcs based on the source_id in the fifth column
                for source_id in np.unique(pc[:, 4]):
                    sensor_pc = pc[pc[:, 4] == source_id]
                    # remove sensor_id column
                    sensor_pc = sensor_pc[:, :4]
                    # save new pcs
                    new_filename = f"{filename}_{source_id}"
                    np.savez_compressed(
                        os.path.join(self.save_dir_pcs, f"{new_filename}.npy"),
                        sensor_pc,
                    )
            else:
                np.savez_compressed(
                    os.path.join(self.save_dir_pcs, f"{filename}_None.npy"), pc
                )
        except:
            print(f"Failed to save point cloud for {filename}")
            return False
        return True

    def download_single_input(self, item):
        judgement_id, data = item
        input_internal_id = data["input_internal_id"]
        is_multilidar = data["is_multilidar"]
        try:
            lidar_sensors = self.get_lidar_sensors(input_internal_id)
            lidar_projectors, calib = self.get_lidar_projectors(input_internal_id)
        except:
            print(f"Failed to get lidar sensors for {input_internal_id}")
            return
        for timestamp in data["timestamps"].keys():
            resource_id = data["timestamps"][timestamp]["resource_id"]
            filename = f"{judgement_id}_{timestamp}"
            pc_exists = filename in self.existing_pcs
            anno_exists = filename in self.existing_annos

            if pc_exists and anno_exists:
                print(f"Skipping {filename}, already exists")
                continue

            if not anno_exists:
                cuboids = self.get_cuboids_all_sensors(
                    data["timestamps"][timestamp]["sensors"],
                    lidar_projectors,
                    is_multilidar,
                    judgement_id,
                    timestamp,
                    lidar_sensors,
                )
            if len(cuboids) == 0:
                print(f"No cuboids found for {input_internal_id}")
                continue

            if not pc_exists:
                try:
                    pc = self.get_pc(resource_id, is_multilidar)

                except Exception:
                    print(f"Failed to download point cloud for {input_internal_id}")
                    continue

                if self.filter_pc:
                    pc = self.filter_pcs(pc, calib, lidar_sensors, input_internal_id)

                self.pc_split_and_save(pc, filename, is_multilidar)
            # save cuboids
            self.cuboids_split_and_save(cuboids, filename)

    def cuboids_split_and_save(self, cuboids, filename):
        sensor_cuboids = {}
        for cuboid in cuboids:
            sensor_cuboids.setdefault(cuboid["sensor"], []).append(cuboid)
        for sensor_name, cuboids in sensor_cuboids.items():
            with open(
                os.path.join(
                    self.save_dir_annos,
                    f"{filename}_{int(sensor_name) if sensor_name is not None else None}.pickle",
                ),
                "wb",
            ) as handle:
                pickle.dump(cuboids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_cuboids_all_sensors(
        self,
        sensors,
        lidar_projectors,
        is_multilidar,
        judgement_id,
        timestamp,
        lidar_sensors,
    ):
        cuboids = []
        for sensor_name, shapes in sensors.items():
            if is_multilidar:
                lidar_projector = lidar_projectors[sensor_name]
                lidar_sensor = lidar_sensors[sensor_name]
            else:
                lidar_projector = list(lidar_projectors.values())[0]
                lidar_sensor = None
            for shape_class, shapes in shapes.items():
                for _, shape in shapes.items():
                    geo = shape["geometry"]
                    cuboid = Cube3D(
                        scale=geo["scale"],
                        coordinates=geo["coordinates"],
                        rotation=geo["rotation"],
                    )
                    new_coords = cuboid.coordinates
                    new_quat = cuboid.rotation
                    if self.transform2LCS:
                        new_coords = (
                            lidar_projector.transform_matrix
                            @ np.concatenate([cuboid.coordinates, [1]])
                        )[:3]
                        new_rot_mat = (
                            lidar_projector.transform_matrix[:3, :3]
                            @ cuboid.rotation_matrix
                        )
                        new_quat = Rotation.from_matrix(new_rot_mat).as_quat()
                    cuboids.append(
                        {
                            "scale": geo["scale"],
                            "coordinates": new_coords,
                            "rotation": new_quat,
                            "class": shape_class,
                            "sensor": lidar_sensor,
                            "judgement_id": judgement_id,
                            "timestamp": timestamp,
                        }
                    )
        return cuboids

    def get_cuboids(self, geometries, lidar_projector):
        cuboids = []
        for obj in geometries:
            geo = json.loads(obj["shape_details"])
            cuboid = Cube3D(
                scale=geo["scale"],
                coordinates=geo["coordinates"],
                rotation=geo["rotation"],
            )
            new_coords = (
                lidar_projector.transform_matrix
                @ np.concatenate([cuboid.coordinates, [1]])
            )[:3]
            new_rot_mat = (
                lidar_projector.transform_matrix[:3, :3] @ cuboid.rotation_matrix
            )
            new_quat = Rotation.from_matrix(new_rot_mat).as_quat()
            # new_cuboid = Cube3D(scale=geo['scale'], coordinates=new_coords, rotation=new_quat, feature_id=obj['shape_class'])
            cuboids.append(
                {
                    "scale": geo["scale"],
                    "coordinates": new_coords,
                    "rotation": new_quat,
                    "class": obj["shape_class"],
                }
            )
        return cuboids

    def get_pc(self, resource_id, is_multilidar: bool = False):
        file_id = parse_file_id(resource_id)
        pc = self.read_pointcloud_from_bucket(file_id, is_multilidar)
        return pc

    def get_calibration(self, input_internal_id):
        calib = self.scene.get_calibration(input_internal_id)
        return calib

    def get_lidar_sensors(self, sceneUUID):
        scene_resolve = self.scene.get_full_scene(sceneUUID)
        sensors = scene_resolve.lidar_resources[0].sensors
        # map name-->id
        return {sensor.name: sensor.id for sensor in sensors}

    def get_lidar_projectors(self, scene_uuid) -> List[LidarProjector]:
        calibrations = self.get_calibration(scene_uuid)
        lidar_projectors = {}
        for sensor_name, calibration in calibrations.items():
            if isinstance(calibration, LidarCalibration):
                lidar_projectors[sensor_name] = LidarProjector(calibration)
        return lidar_projectors, calibrations

    def create_splits(self):
        np.random.seed(42)
        all_annotations = []
        pc_files = np.random.permutation(glob.glob(self.save_dir_pcs + "/*.npz"))

        for pc_file in tqdm(pc_files, desc="Creating splits"):
            annotation_file = pc_file.replace("pcs", "annos").replace(
                ".npy.npz", ".pickle"
            )
            if os.path.exists(annotation_file):
                with open(annotation_file, "rb") as handle:
                    annotation = pickle.load(handle)
                all_annotations.append(annotation)
            else:
                print(f"Annotation file for {pc_file} not found.")

        split_idx = int(len(all_annotations) * self.config["train_split"])
        train_annotations = all_annotations[:split_idx]
        val_annotations = all_annotations[split_idx:]

        with open(os.path.join(self.save_dir, "train.pickle"), "wb") as f:
            pickle.dump(train_annotations, f)
        with open(os.path.join(self.save_dir, "val.pickle"), "wb") as f:
            pickle.dump(val_annotations, f)
        total_time = time.time() - self.start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f"Created train/val split with {len(train_annotations)} train and {len(val_annotations)} val samples"
        )
        print(
            f"Total time: {int(hours):02}:{int(minutes):02}:{int(seconds):02} for {self.count} items"
        )

    def copy_config(self):
        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(self.config_path, "r") as src, open(config_path, "w") as dst:
            dst.write(src.read())

    def arrange_data(self, item):
        judgement_id = item[0]
        input_internal_id = item[1]
        resource_id = item[3]
        geometries = item[4]
        is_multilidar = item[6] > 1
        to_download = {}
        for shape in geometries:
            primarySensor = shape["primarySensor_value"]
            timestamp = shape["shape_timestamp"]
            geometry = shape["shape_details"]
            shape_id = shape["shape_id"]
            shape_class = shape["shape_class"]

            to_download.setdefault(judgement_id, {}).setdefault(
                "timestamps", {}
            ).setdefault(timestamp, {}).setdefault("sensors", {}).setdefault(
                primarySensor, {}
            ).setdefault(shape_class, {}).setdefault(shape_id, {})[
                "geometry"
            ] = geometry

            to_download[judgement_id]["input_internal_id"] = input_internal_id
            to_download[judgement_id]["timestamps"][timestamp]["resource_id"] = (
                resource_id
            )
            to_download[judgement_id]["is_multilidar"] = is_multilidar
        return to_download

    def flatten(self, dct, element):
        # takes 2 dictionaries and merges them
        # if a key is present in both dictionaries, and values are dictionaries, it will merge them recursively
        # if a key is present in both dictionaries, and values are lists, it will append the lists
        # if a key is present in both dictionaries, and values are not lists or dictionaries, it will overwrite the value and print a warning
        if not isinstance(dct, dict):
            raise ValueError("dct should be a dictionary")
        if not isinstance(element, dict):
            raise ValueError("element should be a dictionary")
        for key in element:
            if key in dct:
                if isinstance(dct[key], dict) and isinstance(element[key], dict):
                    dct[key] = self.flatten(dct[key], element[key])
                elif isinstance(dct[key], list) and isinstance(element[key], list):
                    dct[key].extend(element[key])
                elif dct[key] != element[key]:
                    print(
                        f"Warning: overwriting {key} of {dct[key]} with {element[key]}"
                    )
                    dct[key] = element[key]
            else:
                dct[key] = element[key]
        return dct

    def download_dataset(self, datatable):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir_pcs, exist_ok=True)
        os.makedirs(self.save_dir_annos, exist_ok=True)
        # sort the cuboids into judgement_id, timestamp, and sensor_name
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self.arrange_data,
                        datatable.values.tolist(),
                    ),
                    total=len(datatable),
                    desc="Arranging data",
                )
            )
        to_download = {}
        for result in results:
            to_download = self.flatten(to_download, result)
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self.download_single_input,
                        to_download.items(),
                    ),
                    total=len(to_download.items()),
                    desc="Downloading point clouds",
                )
            )

        # Create a basic train/val split
        if self.config["create_splits"]:
            self.create_splits()
        # Copy config.yaml to the new dataset folder
        self.copy_config()


if __name__ == "__main__":
    # get file list in /mnt/bfd/datasets/autobaans/3dod/test/annos
    annos = os.listdir("/mnt/bfd/datasets/autobaans/3dod/cosmos_proj_178/annos")
    for anno in annos:
        with open(
            f"/mnt/bfd/datasets/autobaans/3dod/cosmos_proj_178/annos/{anno}", "rb"
        ) as f:
            data = pickle.load(f)
        print("File:", anno, "Data:", data)
