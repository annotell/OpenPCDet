import numpy as np
import glob
from ..dataset import DatasetTemplate


class JupyterDataset(DatasetTemplate):
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

        self.lidar_files = sorted(glob.glob(str(self.root_path) + '/*/*.npy'))

    def __len__(self):
        return len(self.lidar_files)

    def __getitem__(self, index):
        pointcloud = np.load(self.lidar_files[index], allow_pickle=True)
        pointcloud = np.c_[pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2] - 1.2, pointcloud[:, 3]/2**16]
        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])
        input_dict = {"frame_id": index}
        if "points" in get_item_list:
            input_dict["points"] = pointcloud
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
