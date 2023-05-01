import copy
import glob
import os.path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ..dataset import DatasetTemplate


class Cube3D:
    def __init__(self, scale: np.ndarray, rotation: np.ndarray, coordinates: np.ndarray):
        # Quaternions with order: (x, y, z, w)
        self.rotation = np.asarray(rotation)
        self.coordinates = np.asarray(coordinates)  # X, Y, Z
        self.scale = np.asarray(scale)  # W, L, H
        self.rotation_matrix = Rotation.from_quat(self.rotation).as_matrix()
        self.inverse_rotation_matrix = np.linalg.inv(self.rotation_matrix)

    @property
    def width(self):
        return self.scale[0]

    @property
    def length(self):
        return self.scale[1]

    @property
    def height(self):
        return self.scale[2]

    def get_scale(self):
        return self.scale

    def get_coordinates(self):
        return self.coordinates

    def get_rotation(self):
        return self.rotation

    def calculate_relative_position(self, shape_2):
        return (self.get_coordinates() - shape_2.get_coordinates()).reshape(3, -1)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.scale * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        corners = np.zeros((3, 8))
        corners[0, :] = w / 2 * \
            np.array([1, -1, -1, 1, 1, -1, -1, 1])  # x_corners
        corners[1, :] = l / 2 * \
            np.array([1, 1, 1, 1, -1, -1, -1, -1])  # y_corners
        corners[2, :] = h / 2 * \
            np.array([1, 1, -1, -1, 1, 1, -1, -1])  # z_corners

        # Rotate & Translate
        corners = self.rotation_matrix @ corners + \
            self.coordinates.reshape((3, -1))

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def calculate_intersection(self, shape_2):
        # From Isak
        if np.linalg.norm(self.calculate_relative_position(shape_2)) >= (self.calculate_diagonal() +
                                                                         shape_2.calculate_diagonal()) / 2:
            return 0  # if it is possible for the boxes to intersect
        rotobj = Rotation.from_quat(self.get_rotation())
        R1 = rotobj.as_matrix()

        rotobj = Rotation.from_quat(shape_2.get_rotation())
        R2 = rotobj.as_matrix()
        R2_inv = np.linalg.inv(R2)

        # sample points randomly in box 1
        num_samples = 10000
        points_in_box_1 = np.random.uniform(-0.5, 0.5, size=(
            3, num_samples)) * self.get_scale().reshape(3, -1)

        # transformation from box_1 reference system to box_2 reference system
        point_in_box_2 = R2_inv @ (R1 @ points_in_box_1 +
                                   self.calculate_relative_position(shape_2=shape_2))

        box_2_boundaries = shape_2.get_scale().reshape((3, -1)) / 2

        # caclulate boxes both in box_1 and box_2
        A1 = point_in_box_2 <= box_2_boundaries
        A2 = point_in_box_2 >= -box_2_boundaries
        A = np.logical_and(A1, A2)
        is_inside = np.prod(A, axis=0)
        # fractions of box 1 in box 2
        fraction_overlay = np.sum(is_inside) / is_inside.size
        intersection_volume = fraction_overlay * self.calculate_volume()
        return intersection_volume

    def calculate_union(self, shape_2):
        return self.calculate_volume() + shape_2.calculate_volume() - self.calculate_intersection(shape_2=shape_2)

    def calculate_iou(self, shape_2):
        intersection_volume = self.calculate_intersection(shape_2=shape_2)
        if intersection_volume == 0:
            return 0
        union_volume = self.calculate_volume() + shape_2.calculate_volume() - \
            intersection_volume
        return intersection_volume / union_volume

    def calculate_diagonal(self):
        return np.linalg.norm(self.get_scale())

    def calculate_volume(self):
        return np.prod(self.get_scale())

    def transform_points_to_box_coordinate_system(self, points: np.ndarray):
        """

        :param points: N x 3 matrix of points in lidar coordinate system
        :return: N x 3 matrix of points in this box´s local coordinate system
        """
        points_box = self.inverse_rotation_matrix @ (
            points.T - self.coordinates.reshape((3, -1)))
        return points_box.T

    def points_inside(self, points_box):
        """

        :param points_box: N x 3 matrix in box coordinates, eg points_to_box_coordinate_system(points)
        :return: N boolean array
        """
        n = points_box.shape[0]
        condition = np.ones(n).astype(bool)
        for i in range(3):
            c1 = points_box[:, i] >= - self.scale[i] / 2
            c2 = points_box[:, i] <= self.scale[i] / 2
            condition = condition * c1 * c2
        return condition


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
            self.annotations = glob.glob(os.path.join(
                str(self.root_path), '**/*.parquet'), recursive=True)
        else:
            self.annotations = glob.glob(os.path.join(
                str(self.root_path), '**/*.parquet'), recursive=True)

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
    def points_in_cuboid(cuboid: Cube3D, points3D: np.ndarray, wlh_factor: float = 1.0):
        """
        Checks whether points are inside the box.

        Picks one corner as reference (p1) and computes the vector to a target point (v).
        Then for each of the 3 axes, project v onto the axis and compare the length.
        :param cuboid: The cuboid to check.
        :param points3D: Nx3 array
        :param wlh_factor: Inflates or deflates the box.
        :return: <np.bool: n, >.
        """
        corners = cuboid.corners(wlh_factor=wlh_factor)

        p1 = corners[:, 0]
        p_x = corners[:, 4]
        p_y = corners[:, 1]
        p_z = corners[:, 3]

        i = p_x - p1
        j = p_y - p1
        k = p_z - p1

        v = (points3D - p1).T

        iv = i @ v
        jv = j @ v
        kv = k @ v

        mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
        mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
        mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

        return mask

    def get_cuboid_points(self, cuboid, points):
        wlh_factor = np.array([1., 1., 1.])
        pts_mask = self.points_in_cuboid(
            cuboid, points[:, [0, 1, 2]], wlh_factor=wlh_factor)
        points = points[pts_mask]
        # add column to points and fill with zeros
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
        return points

    @staticmethod
    def filter_pointcloud(pointcloud):
        # check how many points in pointcloud are within a cuboid of 6m x 6m x 6m
        mask = np.logical_and(pointcloud[:, 0] > -6, pointcloud[:, 0] < 6)
        mask = np.logical_and(mask, pointcloud[:, 1] > -6)
        mask = np.logical_and(mask, pointcloud[:, 1] < 6)
        mask = np.logical_and(mask, pointcloud[:, 2] > -3)
        mask = np.logical_and(mask, pointcloud[:, 2] < 3)
        pointcloud = pointcloud[mask]
        return pointcloud

    def select_points(self, points):
        points = self.filter_pointcloud(points)
        return points

    @staticmethod
    def random_box(cuboid):
        coordinates = cuboid.coordinates
        scale = cuboid.scale
        rotation = cuboid.rotation
        # sample uniform random point between coordinate - scale and coordinate + scale
        delta_coordinates = np.random.uniform(
            coordinates - scale / 2, coordinates + scale / 2)
        # add noise to quaternion rotation
        delta_rotation = np.random.normal(rotation, 0.1)
        noise_cuboid = Cube3D(coordinates=delta_coordinates,
                              rotation=delta_rotation, scale=scale)

        return noise_cuboid

    @staticmethod
    def generate_points_cuboid(height, width, length, spacing=15):
        points = []
        num_points_per_face = int(height / (spacing / 100))
        xy_points = np.meshgrid(np.linspace(-width/2, width/2, num_points_per_face),
                                np.linspace(-length/2, length/2,
                                            num_points_per_face),
                                indexing='ij')
        xz_points = np.meshgrid(np.linspace(-width/2, width/2, num_points_per_face),
                                np.linspace(-height/2, height/2,
                                            num_points_per_face),
                                indexing='ij')
        yz_points = np.meshgrid(np.linspace(-length/2, length/2, num_points_per_face),
                                np.linspace(-height/2, height/2,
                                            num_points_per_face),
                                indexing='ij')
        points.append(np.vstack([xy_points[0].flatten(),
                                xy_points[1].flatten(),
                                np.full(num_points_per_face**2, -height/2)]).T)
        points.append(np.vstack([xy_points[0].flatten(),
                                xy_points[1].flatten(),
                                np.full(num_points_per_face**2, height/2)]).T)
        points.append(np.vstack([xz_points[0].flatten(),
                                np.full(num_points_per_face**2, -length/2),
                                xz_points[1].flatten()]).T)
        points.append(np.vstack([xz_points[0].flatten(),
                                np.full(num_points_per_face**2, length/2),
                                xz_points[1].flatten()]).T)
        points.append(np.vstack([np.full(num_points_per_face**2, -width/2),
                                yz_points[0].flatten(),
                                yz_points[1].flatten()]).T)
        points.append(np.vstack([np.full(num_points_per_face**2, width/2),
                                yz_points[0].flatten(),
                                yz_points[1].flatten()]).T)
        points = np.vstack(points)
        points = np.hstack((points, 1*np.ones((points.shape[0], 1))))
        return points

    def get_noise_pc_and_box(self, cuboid, points):
        w, l, h = cuboid.scale
        cuboid_points = self.generate_points_cuboid(h, w, l)
        cuboid_points = np.hstack(
            (cuboid_points, np.zeros((cuboid_points.shape[0], 1))))

        noise_cuboid = self.random_box(cuboid)
        points[:, :3] = noise_cuboid.transform_points_to_box_coordinate_system(
            points[:, :3])
        inv_quat = Rotation.from_matrix(
            noise_cuboid.inverse_rotation_matrix).as_quat()
        inv_coord = -(noise_cuboid.inverse_rotation_matrix @
                      noise_cuboid.coordinates).T
        cuboid_corrected = Cube3D(noise_cuboid.scale, inv_quat, inv_coord)

        # add column to noise_pc and fill with ones
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        # stack points with cuboid_points
        points = np.vstack((points, cuboid_points))

        return points, cuboid_corrected

    def generate_sample(self, path):
        df = pd.read_parquet(path)
        points = np.load(path.replace('parquet', 'npy'))
        cuboid = Cube3D(
            df.Cube3D_scale[0], df.Cube3D_rotation[0], df.Cube3D_coordinates[0])
        cuboid_local = Cube3D(df.Cube3D_scale[0], np.array(
            [0, 0, 0, 1]), np.array([0, 0, 0]))
        points[:, :3] = cuboid.transform_points_to_box_coordinate_system(
            points[:, :3])
        points, correct_cuboid = self.get_noise_pc_and_box(
            cuboid_local, points)
        annotation = self.get_annotation_from_cuboid(correct_cuboid)
        points = self.select_points(points)
        points = np.c_[points[:, 1], -points[:, 0],
                       points[:, 2], points[:, 3] / 2 ** 16, points[:, 4]]
        return points, annotation

    @staticmethod
    def get_annotation_from_cuboid(correct_cuboid):
        annotation = {"name": ['Object'], "dimensions": [],
                      "location": [], "rotation_y": []}
        w, l, h = correct_cuboid.scale
        rot = correct_cuboid.rotation
        coord = correct_cuboid.coordinates

        new_coord = np.array([coord[1], -coord[0], coord[2]])
        new_scale = np.array([l, w, h])
        new_rot = np.array(Rotation.from_quat(
            rot).as_euler('xyz')[2] - np.pi / 2)

        gt_boxes_lidar = np.concatenate(
            [new_coord, new_scale, new_rot[..., np.newaxis]])

        annotation["dimensions"].append(new_scale)
        annotation["location"].append(new_coord)
        annotation["rotation_y"].append(new_rot)
        annotation["dimensions"] = np.array(annotation["dimensions"])
        annotation["location"] = np.array(annotation["location"])
        annotation["rotation_y"] = np.array(
            annotation["rotation_y"])[..., np.newaxis]
        annotation["gt_boxes_lidar"] = gt_boxes_lidar[np.newaxis, ...]
        annotation["name"] = np.array(annotation["name"])

        return annotation

    def __getitem__(self, index):
        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])

        pointcloud, annotation = self.generate_sample(self.annotations[index])
        while len(pointcloud) < 5:
            index = np.random.randint(0, len(self.annotations))
            pointcloud, annotation = self.generate_sample(
                self.annotations[index])

        input_dict = {"frame_id": index}
        if "points" in get_item_list:
            input_dict["points"] = pointcloud

        input_dict.update(
            {"gt_names": annotation["name"], "gt_boxes": annotation["gt_boxes_lidar"]})
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, eval_det_annos, class_names):
        from .eval import get_official_eval_result
        from .object_centric_utils import transform_annotations_to_kitti_format
        eval_gt_annos = [copy.deepcopy(
            self.annotations[det['frame_id']][1]) for det in eval_det_annos]
        transform_annotations_to_kitti_format(eval_det_annos)
        transform_annotations_to_kitti_format(eval_gt_annos, info_with_fakelidar=self.dataset_cfg.get(
            'INFO_WITH_FAKELIDAR', False))

        ap_result_str, ap_dict = get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=class_names)
        return ap_result_str, ap_dict
