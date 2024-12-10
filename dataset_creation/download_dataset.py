from concurrent.futures import ThreadPoolExecutor
import glob
from itertools import repeat
import json
import os
import pickle
import time
from typing import List, Optional

import numpy as np
from annotell.apiclients.scene_input_api_client import SceneInputApiClient
from kognic.filestorage.filestorage import FileId, FileStorage
from kognic.filestorage.resource_parser import parse_file_id
from kognic.io.model.calibration.lidar.lidar_calibration import LidarCalibration
from kognic.judgement_shapes.cube_3d import Cube3D
from kognic.potree.potree import PointAttributes
from kognic.potree_filestorage.potree_filestorage import (
    MissingAttributeError,
    PotreeFileStorage,
)
from kognic.projection.projection import LidarProjector, get_camera_projector
from p_tqdm import p_umap
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from urllib3.exceptions import ProtocolError
import yaml

from kognic.soleng.perception_expert_cli.tools import pprint_dict
from annotell.datamodel.scene_input_model import SingleLidarResource
from annotell.apiclients.scene_input_api_client import SceneInputApiClient


potree_fs = PotreeFileStorage()
LIDAR_INFO = {}


class DatasetLoader:
    def __init__(self, config):
        self.config_path = config
        self.scene = SceneInputApiClient(env="production")
        with open(config) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.save_dir = os.path.join(
            self.config["dataset_root"], self.config["dataset_name"]
        )
        self.save_dir_pcs = os.path.join(self.save_dir, "pcs")
        self.save_dir_annos = os.path.join(self.save_dir, "annos")
        self.filter_pc = self.config.get("filter_pc", False)
        self.count = 0
        self.pc_time = 0
        self.pc_transform_time = 0
        self.cuboid_time = 0
        self.save_time = 0
        self.start_time = time.time()

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
    def filter_pcs(self, pc, calibrations, lidar_sensors):
        camera_projectors = {}
        lidar_projectors = {}
        for sensor_name, calibration in calibrations.items():
            if isinstance(calibration, LidarCalibration):
                lidar_projectors[sensor_name] = LidarProjector(calibration)
            else:
                camera_projectors[sensor_name] = get_camera_projector(calibration)
        is_multi_lidar = len(lidar_projectors) > 1
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

    def check_is_multi_lidar(self, scene_uuid: str) -> bool:
        if scene_uuid in LIDAR_INFO:
            return LIDAR_INFO[scene_uuid]
        scene = self.scene.get_scene(scene_uuid)
        if len(scene.lidar_resources) == 0 or isinstance(
            scene.lidar_resources[0], SingleLidarResource
        ):
            LIDAR_INFO[scene_uuid] = False
            return False
        LIDAR_INFO[scene_uuid] = len(scene.lidar_resources[0].sensors) > 1
        return LIDAR_INFO[scene_uuid]

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

    def download_single_input(self, item):
        judgement_id, data = item
        input_internal_id = data["input_internal_id"]
        is_multilidar = data["is_multilidar"]
        for timestamp in data["timestamps"].keys():
            resource_id = data["timestamps"][timestamp]["resource_id"]
            filename = data["timestamps"][timestamp]["filename"]
            if os.path.exists(os.path.join(self.save_dir_pcs, f"{filename}.npy.npz")):
                if os.path.exists(
                    os.path.join(self.save_dir_annos, f"{filename}.pickle")
                ):
                    continue
            # print(f"Downloading {filename}")
            start_time = time.time()
            pc = self.get_pc(resource_id, is_multilidar)
            self.pc_time += time.time() - start_time
            start_time = time.time()
            lidar_sensors = self.get_lidar_sensors(input_internal_id)
            lidar_projectors, calib = self.get_lidar_projectors(input_internal_id)
            if self.filter_pc:
                pc = self.filter_pcs(pc, calib, lidar_sensors)
            self.pc_transform_time += time.time() - start_time
            start_time = time.time()

            cuboids = self.get_cuboids_all_sensors(
                data["timestamps"][timestamp]["sensors"],
                lidar_projectors,
                is_multilidar,
            )
            self.cuboid_time += time.time() - start_time
            start_time = time.time()

            if len(cuboids) > 0:
                np.savez_compressed(
                    os.path.join(self.save_dir_pcs, f"{filename}.npy"), pc
                )
                with open(
                    os.path.join(self.save_dir_annos, f"{filename}.pickle"), "wb"
                ) as handle:
                    pickle.dump(cuboids, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.save_time += time.time() - start_time
            self.count += 1
            print(
                "PC time:",
                self.pc_time / self.count,
                "PC transform time:",
                self.pc_transform_time / self.count,
                "Cuboid time:",
                self.cuboid_time / self.count,
                "Save time:",
                self.save_time / self.count,
                "Seconds per item:",
                (time.time() - self.start_time) / self.count,
                end="\r",
            )

    def get_cuboids_all_sensors(self, sensors, lidar_projectors, is_multilidar):
        cuboids = []
        for sensor_name, shapes in sensors.items():
            if is_multilidar:
                lidar_projector = lidar_projectors[sensor_name]
            else:
                lidar_projector = list(lidar_projectors.values())[0]
            for shape_class, shapes in shapes.items():
                for _, shape in shapes.items():
                    geo = shape["geometry"]
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
        print("Val annotations:", val_annotations)
        print("Val annotation count:", len(val_annotations))
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
        is_multilidar = self.check_is_multi_lidar(input_internal_id)
        to_download = {}
        for shape in geometries:
            primarySensor = shape["primarySensor_value"]
            timestamp = shape["shape_timestamp"]
            geometry = shape["shape_details"]
            shape_id = shape["shape_id"]
            shape_class = shape["shape_class"]

            filename = f"{judgement_id}_{timestamp}"
            if os.path.exists(os.path.join(self.save_dir_pcs, f"{filename}.npy.npz")):
                if os.path.exists(
                    os.path.join(self.save_dir_annos, f"{filename}.pickle")
                ):
                    continue
            to_download.setdefault(judgement_id, {}).setdefault(
                "timestamps", {}
            ).setdefault(timestamp, {}).setdefault("sensors", {}).setdefault(
                primarySensor, {}
            ).setdefault(
                shape_class, {}
            ).setdefault(
                shape_id, {}
            )[
                "geometry"
            ] = geometry

            to_download[judgement_id]["input_internal_id"] = input_internal_id
            to_download[judgement_id]["timestamps"][timestamp][
                "resource_id"
            ] = resource_id
            to_download[judgement_id]["timestamps"][timestamp]["filename"] = filename
            to_download[judgement_id]["is_multilidar"] = is_multilidar
        return to_download

    def download_dataset(self, datatable):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir_pcs, exist_ok=True)
        os.makedirs(self.save_dir_annos, exist_ok=True)
        to_download = {}
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
        # flatten the results
        def flatten(dct, element):
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
                        dct[key] = flatten(dct[key], element[key])
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

        for result in results:
            to_download = flatten(to_download, result)

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
