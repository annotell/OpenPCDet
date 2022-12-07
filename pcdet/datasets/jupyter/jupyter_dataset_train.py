import copy
import os.path
import numpy as np
import pickle
from ..dataset import DatasetTemplate


class JupyterDatasetTrain(DatasetTemplate):
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

    def __getitem__(self, index):
        pc_path = os.path.join(str(self.root_path), self.annotations[index][0])
        pointcloud = np.load(pc_path, allow_pickle=True)
        pointcloud = np.c_[pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], pointcloud[:, 3] / 2 ** 16]
        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])
        annotations = self.annotations[index][1]

        input_dict = {"frame_id": index}
        if "points" in get_item_list:
            input_dict["points"] = pointcloud

        input_dict.update({"gt_names": annotations["name"], "gt_boxes": annotations["gt_boxes_lidar"]})
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, eval_det_annos, class_names):
        from .eval import get_official_eval_result
        from .jupuyter_utils import transform_annotations_to_kitti_format
        eval_gt_annos = [copy.deepcopy(self.annotations[det['frame_id']][1]) for det in eval_det_annos]
        transform_annotations_to_kitti_format(eval_det_annos)
        transform_annotations_to_kitti_format(eval_gt_annos, info_with_fakelidar=self.dataset_cfg.get(
            'INFO_WITH_FAKELIDAR', False))

        ap_result_str, ap_dict = get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=class_names)
        return ap_result_str, ap_dict
