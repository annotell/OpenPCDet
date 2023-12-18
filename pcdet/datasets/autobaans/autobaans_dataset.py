import copy
import os.path
import pickle

import numpy as np

from ..dataset import DatasetTemplate


class AutobaansDataset(DatasetTemplate):
    def __init__(
            self,
            dataset_cfg=None,
            class_names=None,
            training=True,
            root_path=None,
            logger=None,
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        if training:
            with open(os.path.join(str(self.root_path), 'train.pickle'), 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            with open(os.path.join(str(self.root_path), 'val.pickle'), 'rb') as f:
                self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        pc_path = os.path.join(str(self.root_path), self.annotations[index][0])
        pointcloud = np.load(pc_path, allow_pickle=True)
        pointcloud = pointcloud['arr_0']
        pointcloud = np.c_[pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], pointcloud[:, 3]/2**16, pointcloud[:, 3]/2**16]
        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])
        annotations = self.annotations[index][1]

        input_dict = {"frame_id": index}
        if "points" in get_item_list:
            input_dict["points"] = pointcloud

        input_dict.update({"gt_names": annotations["name"], "gt_boxes": annotations["gt_boxes_lidar"]})
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

# import glob

# import numpy as np

# from ..dataset import DatasetTemplate

# class AutobaansDatasetEval(DatasetTemplate):
#     def __init__(
#             self,
#             dataset_cfg=None,
#             class_names=None,
#             training=True,
#             root_path=None,
#             logger=None,
#     ):
#         super().__init__(
#             dataset_cfg=dataset_cfg,
#             class_names=class_names,
#             training=training,
#             root_path=root_path,
#             logger=logger,
#         )

#         self.lidar_files = sorted(glob.glob(str(self.root_path) + '/*.npy'))

#     def __len__(self):
#         return len(self.lidar_files)

#     def __getitem__(self, index):
#         pointcloud = np.load(self.lidar_files[index], allow_pickle=True)
#         pointcloud = pointcloud['arr_0']
#         pointcloud = np.c_[pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], pointcloud[:, 3]/2**16]
#         get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])
#         input_dict = {"frame_id": index}
#         if "points" in get_item_list:
#             input_dict["points"] = pointcloud
#         data_dict = self.prepare_data(data_dict=input_dict)

#         return data_dict
