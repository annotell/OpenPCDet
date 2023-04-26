import copy
import os.path
import numpy as np
from ..dataset import DatasetTemplate
import glob
import pandas as pd
from scipy.spatial.transform import Rotation


class ObjectCentricTrainDataset(DatasetTemplate):
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
            self.annotations = glob.glob(os.path.join(str(self.root_path), '**/*.parquet'), recursive=True)
        else:
            self.annotations = glob.glob(os.path.join(str(self.root_path), '**/*.parquet'), recursive=True)


    @staticmethod
    def generate_prediction_dicts(
            batch_dict, pred_dicts, class_names, output_path=None
    ):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])

            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            annos.append(single_pred_dict)

        return annos

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def get_annotation_from_parquet(path):
        annotation = {"name": ['Object'], "dimensions": [], "location": [], "rotation_y": []}
        df = pd.read_parquet(path)
        w, l, h = df.Cube3D_scale[0]
        rot = df.Cube3D_rotation[0]
        coord = df.Cube3D_coordinates[0]

        offset = np.array([coord[1], -coord[0], coord[2]])
        new_coord = np.array([0,0,0])
        new_scale = np.array([l, w, h])
        new_rot = np.array(Rotation.from_quat(rot).as_euler('xyz')[2] - np.pi / 2)

        gt_boxes_lidar = np.concatenate([new_coord, new_scale, new_rot[..., np.newaxis]])

        annotation["dimensions"].append(new_scale)
        annotation["location"].append(new_coord)
        annotation["rotation_y"].append(new_rot)
        annotation["dimensions"] = np.array(annotation["dimensions"])
        annotation["location"] = np.array(annotation["location"])
        annotation["rotation_y"] = np.array(annotation["rotation_y"])[..., np.newaxis]
        annotation["gt_boxes_lidar"] = gt_boxes_lidar[np.newaxis, ...]
        annotation["name"] = np.array(annotation["name"])

        return annotation, offset

    @staticmethod
    def filter_pointcloud(pointcloud):
        # check how many points in pointcloud are within a cuboid of 6m x 6m x 6m
        mask = np.logical_and(pointcloud[:, 0] >= -6, pointcloud[:, 0] <= 6)
        mask = np.logical_and(mask, pointcloud[:, 1] >= -6)
        mask = np.logical_and(mask, pointcloud[:, 1] <= 6)
        mask = np.logical_and(mask, pointcloud[:, 2] >= -2)
        mask = np.logical_and(mask, pointcloud[:, 2] <= 2)
        pointcloud = pointcloud[mask]
        return pointcloud

    def __getitem__(self, index):
        pc_path = self.annotations[index].replace('.parquet', '.npy')
        pointcloud = np.load(pc_path)
        pointcloud = np.c_[pointcloud[:, 1], -pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 3]/2**16]
        annotations, new_coord = self.get_annotation_from_parquet(self.annotations[index])
        pointcloud[:, :3] -= new_coord

        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])

        pointcloud = self.filter_pointcloud(pointcloud)
        while len(pointcloud) <= 2:
            # sample a new random index
            index = np.random.randint(0, len(self.annotations))
            pc_path = self.annotations[index].replace('.parquet', '.npy')
            pointcloud = np.load(pc_path)
            pointcloud = np.c_[pointcloud[:, 1], -pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 3] / 2 ** 16]
            annotations, new_coord = self.get_annotation_from_parquet(self.annotations[index])
            pointcloud[:, :3] -= new_coord
            pointcloud = self.filter_pointcloud(pointcloud)

        input_dict = {"frame_id": index}
        if "points" in get_item_list:
            input_dict["points"] = pointcloud

        input_dict.update({"gt_names": annotations["name"], "gt_boxes": annotations["gt_boxes_lidar"]})
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, eval_det_annos, class_names):
        from .eval import get_official_eval_result
        from .object_centric_utils import transform_annotations_to_kitti_format
        eval_gt_annos = [copy.deepcopy(self.annotations[det['frame_id']][1]) for det in eval_det_annos]
        transform_annotations_to_kitti_format(eval_det_annos)
        transform_annotations_to_kitti_format(eval_gt_annos, info_with_fakelidar=self.dataset_cfg.get(
            'INFO_WITH_FAKELIDAR', False))

        ap_result_str, ap_dict = get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=class_names)
        return ap_result_str, ap_dict
