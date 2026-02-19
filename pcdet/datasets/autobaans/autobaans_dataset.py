import copy
import gc
import glob
import os
import pickle
import time
import traceback

import numpy as np
import torch

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from scipy.spatial.transform import Rotation
import sys


class AutobaansDataset(DatasetTemplate):
    lidars_loaded = 0

    def __init__(
        self,
        dataset_cfg=None,
        training=True,
        root_path=None,
        logger=None,
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=dataset_cfg.CLASS_NAMES,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.custom_infos = []
        if self.split == "train":
            with open(os.path.join(str(self.root_path), "train.pickle"), "rb") as f:
                self.custom_infos = pickle.load(f)
        elif self.split == "val":
            print("Val root path: ", str(self.root_path))
            with open(os.path.join(str(self.root_path), "val.pickle"), "rb") as f:
                self.custom_infos = pickle.load(f)
        elif self.split == "test":
            with open(os.path.join(str(self.root_path), "val.pickle"), "rb") as f:
                self.custom_infos = pickle.load(f)

        self.sample_id_list = [i for i, _ in enumerate(self.custom_infos)]
        self._anno_index = self._build_anno_index()

    def _build_anno_index(self):
        """Build lookup from base annotation ID to full file path.

        Annotation files may have extra suffixes (sensor name, shape type)
        beyond the base ID stored in train.pickle. E.g.:
          train.pickle ID: "12967050_4000"
          file on disk: "12967050_4000_None_Cube3D.pickle"
        """
        annos_dir = os.path.join(str(self.root_path), "annos")
        if not os.path.isdir(annos_dir):
            return {}
        index = {}
        for f in os.listdir(annos_dir):
            if not f.endswith(".pickle"):
                continue
            # Extract base ID: first 2 underscore-separated parts
            parts = f.replace(".pickle", "").split("_")
            if len(parts) >= 2:
                base_id = "_".join(parts[:2])
                # Prefer shorter filenames (compat names) over longer ones
                if base_id not in index or len(f) < len(os.path.basename(index[base_id])):
                    index[base_id] = os.path.join(annos_dir, f)
        print(f"[AutobaansDataset] Built annotation index: {len(index)} entries from {annos_dir}", flush=True)
        return index

    def _find_anno_path(self, anno_id):
        """Find annotation file path for a given base annotation ID."""
        # Fast path: use pre-built index
        if anno_id in self._anno_index:
            return self._anno_index[anno_id]
        # Fallback: try exact match patterns
        annos_dir = os.path.join(str(self.root_path), "annos")
        for suffix in [".pickle", "_Cube3D.pickle"]:
            path = os.path.join(annos_dir, anno_id + suffix)
            if os.path.exists(path):
                return path
        # Last resort: glob
        matches = glob.glob(os.path.join(annos_dir, anno_id + "*.pickle"))
        if matches:
            return matches[0]
        return None

    def get_label(self, idx):
        label = self.custom_infos[idx][1]
        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = label["gt_boxes_lidar"]
        gt_names = label["name"]
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger,
        )
        self.split = split

        if self.split == "train":
            with open(os.path.join(str(self.root_path), "train.pickle"), "rb") as f:
                self.custom_infos = pickle.load(f)
        elif self.split == "val":
            with open(os.path.join(str(self.root_path), "val.pickle"), "rb") as f:
                self.custom_infos = pickle.load(f)
        self.sample_id_list = [i for i, _ in enumerate(self.custom_infos)]

    def get_lidar(self, pc_path):
        try:
            with np.load(
                pc_path, allow_pickle=True, mmap_mode="r"
            ) as data:  # Use "with" to auto-close
                pointcloud = data["arr_0"]
            pointcloud = np.c_[
                pointcloud[:, 0],
                pointcloud[:, 1],
                pointcloud[:, 2],
                pointcloud[:, 3] / 2**16,
            ]
            self.lidars_loaded += 1
            # print("Lidars loaded:", self.lidars_loaded)
            return pointcloud
        except Exception as e:
            print("Error loading pointcloud from: ", pc_path, "ERR:", e, flush=True)
            return None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs
        return len(self.custom_infos)

    def convert_annotations(self, path):
        try:
            with open(path, "rb") as f:
                judgement = pickle.load(f)
        except (EOFError, FileNotFoundError, Exception) as e:
            print(f"Error loading annotations from: {path} ERR: {e}", flush=True)
            return None
        annotations = {"name": [], "dimensions": [], "location": [], "rotation_y": []}

        for i in range(0, len(judgement)):
            curr_obj = judgement[i]
            if curr_obj["class"] not in self.dataset_cfg["CLASS_ADJUSTMENTS"]:
                continue
            # Skip non-box annotations (e.g. ExtremePointBox) that lack 3D box keys
            if not all(k in curr_obj for k in ("coordinates", "scale", "rotation")):
                continue
            obj_class = self.dataset_cfg["CLASS_ADJUSTMENTS"][curr_obj["class"]]
            coordinates = curr_obj["coordinates"]
            wid, le, hei = curr_obj["scale"]
            rotation = Rotation.from_quat(curr_obj["rotation"]).as_euler("xyz")[2]

            annotations["name"].append(obj_class)
            annotations["dimensions"].append(np.array([le, wid, hei]))
            annotations["location"].append(np.array(coordinates))
            annotations["rotation_y"].append(rotation)

        if len(annotations["name"]) > 0:
            annotations["name"] = np.array(annotations["name"])
            annotations["rotation_y"] = np.array(annotations["rotation_y"])
            annotations["dimensions"] = np.array(annotations["dimensions"])
            annotations["location"] = np.concatenate(
                [loc.reshape(1, 3) for loc in annotations["location"]], axis=0
            )

            loc = annotations["location"]
            dims = annotations["dimensions"]
            rots = annotations["rotation_y"]
            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            annotations["gt_boxes_lidar"] = gt_boxes_lidar
        return annotations

    def __getitem__(self, index):
        max_retries = 200
        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])
        pointcloud = None
        annotations = None

        for attempt in range(max_retries):
            if self._merge_all_iters_to_one_epoch:
                index = index % len(self.custom_infos)

            anno_id = self.custom_infos[index]
            pc_filename = anno_id + ".npy.npz"
            pc_path = os.path.join(str(self.root_path), "pcs", pc_filename)
            anno_path = self._find_anno_path(anno_id)

            pointcloud = self.get_lidar(pc_path)
            annotations = self.convert_annotations(anno_path) if anno_path else None

            if pointcloud is not None and annotations is not None and "gt_boxes_lidar" in annotations:
                break
            # PC not downloaded yet or annotation empty — wait briefly then try another sample
            if attempt < max_retries - 1:
                if attempt % 50 == 0:
                    pc_dir = os.path.join(str(self.root_path), "pcs")
                    n_pcs = len(os.listdir(pc_dir)) if os.path.isdir(pc_dir) else 0
                    print(f"[getitem] attempt={attempt}, index={index}, pc={pointcloud is not None}, "
                          f"anno={annotations is not None}, "
                          f"has_boxes={'gt_boxes_lidar' in annotations if annotations else 'N/A'}, "
                          f"pcs_on_disk={n_pcs}", flush=True)
                time.sleep(0.5)
                index = np.random.randint(0, len(self.custom_infos))
            gc.collect()

        if pointcloud is None or annotations is None or "gt_boxes_lidar" not in annotations:
            print(f"[getitem] FAILED after {max_retries} retries. "
                  f"pc={pointcloud is not None}, anno={annotations is not None}, "
                  f"has_boxes={'gt_boxes_lidar' in annotations if annotations else 'N/A'}", flush=True)
            raise RuntimeError(f"Could not load sample after {max_retries} retries")

        input_dict = {"frame_id": index}
        if "points" in get_item_list:
            input_dict["points"] = pointcloud

        input_dict.update(
            {"gt_names": annotations["name"], "gt_boxes": annotations["gt_boxes_lidar"]}
        )
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def get_infos(
        self,
        class_names,
        num_workers=4,
        has_label=True,
        sample_id_list=None,
        num_features=4,
    ):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print("%s sample_idx: %s" % (self.split, sample_idx))
            info = {}
            pc_info = {"num_features": num_features, "lidar_idx": sample_idx}
            info["point_cloud"] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations["name"] = name
                annotations["gt_boxes_lidar"] = gt_boxes_lidar[:, :7]
                info["annos"] = annotations

            return info

        sample_id_list = (
            sample_id_list if sample_id_list is not None else self.sample_id_list
        )

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(
        self, info_path=None, used_classes=None, split="train"
    ):
        import torch

        database_save_path = Path(self.root_path) / (
            "gt_database" if split == "train" else ("gt_database_%s" % split)
        )
        db_info_save_path = Path(self.root_path) / ("autobaans_dbinfos_%s.pkl" % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, "rb") as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print("gt_database sample: %d/%d" % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info["point_cloud"]["lidar_idx"]
            points = self.get_lidar(sample_idx)
            annos = info["annos"]
            names = annos["name"]
            gt_boxes = annos["gt_boxes_lidar"]

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = "%s_%s_%d.bin" % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(
                        filepath.relative_to(self.root_path)
                    )  # gt_database/xxxxx.bin
                    db_info = {
                        "name": names[i],
                        "path": db_path,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))

        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(
        class_names, gt_names, gt_boxes, save_label_path
    ):
        with open(save_label_path, "w") as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0],
                    y=boxes[1],
                    z=(boxes[2]),
                    l=boxes[3],
                    w=boxes[4],
                    h=boxes[5],
                    angle=boxes[6],
                    name=name,
                )
                f.write(line)


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = AutobaansDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        training=False,
        logger=common_utils.create_logger(),
    )
    train_split, val_split = "train", "val"
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ("custom_infos_%s.pkl" % train_split)
    val_filename = save_path / ("custom_infos_%s.pkl" % val_split)

    print(
        "------------------------Start to generate data infos------------------------"
    )

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, "wb") as f:
        pickle.dump(custom_infos_train, f)
    print("Custom info train file is saved to %s" % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, "wb") as f:
        pickle.dump(custom_infos_val, f)
    print("Custom info train file is saved to %s" % val_filename)

    print(
        "------------------------Start create groundtruth database for data augmentation------------------------"
    )
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print("------------------------Data preparation done------------------------")


if __name__ == "__main__":
    if sys.argv.__len__() > 1 and sys.argv[1] == "create_custom_infos":
        from pathlib import Path

        import yaml
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = Path("/root/OpenPCDet/data/autobaans/")
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=["Medium", "Large", "VeryLarge"],
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
        )
