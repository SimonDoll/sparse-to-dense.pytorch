import os
import os.path
import numpy as np
import pathlib
import random


import cv2
import tqdm
import torch.utils.data as data
import h5py
import dataloaders.transforms as transforms
from dataset_3d.data_loaders import dataset_loader
from dataset_3d.utils.projection_utils import LidarToCameraProjector
import dataset_3d.utils.loading_utils as loading_utils
import dataset_3d.utils.visualization_utils as visualization_utils


class CarlaDataset(data.Dataset):
    SEED = 42

    _modality_names = ['rgb', 'rgbd']

    def __init__(self, root, type, modality="rgbd", camera_rgb_name="cam_front", camera_depth_name="cam_front_depth", lidar_name="lidar_top", ego_pose_sensor_name="imu_perfect"):

        random.seed(self.SEED)

        assert type == "val" or type == "train", "unsupported dataset type {}".format(
            type)

        root = pathlib.Path(root).joinpath(type)

        self._loader = dataset_loader.DatasetLoader(root)
        self._loader.setup()

        self._projector = LidarToCameraProjector(
            self._loader, camera_depth_name, lidar_name, ego_pose_sensor_name)

        self._camera_rgb_sensor, _ = loading_utils.load_sensor_with_calib(
            self._loader, camera_rgb_name)

        self._camera_depth_sensor, _ = loading_utils.load_sensor_with_calib(
            self._loader, camera_depth_name)

        self._data_entries = self.prepare_dataset()

        # if type == 'train':
        #     self.transform = self.train_transform
        # elif type == 'val':
        #     self.transform = self.val_transform
        # else:
        #     raise (RuntimeError("Invalid dataset type: " + type + "\n"
        #                         "Supported dataset types are: train, val"))

        assert (modality in self._modality_names), "Invalid modality type: " + modality + "\n" + \
            "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def _sensor_data_to_full_filepath(self, s_data):
        path = s_data.file
        path = str(self._loader.dataset_root.joinpath(path))
        return path

    def prepare_dataset(self,):
        """
        We simply store all samples of the dataset in a random order.
        """
        data_entries = []
        for scene_token in tqdm.tqdm(self._loader.scene_tokens):
            scene = self._loader.get_scene(scene_token)
            sample_token = scene.first_sample_token
            # next token is none if scene ends
            while sample_token != None:

                sample = self._loader.get_sample(sample_token)
                data_entries.append(sample)
                sample_token = sample.next_token

        # shuffle
        random.shuffle(data_entries)
        return data_entries

    def load_data(self, sample):

        depth_map_lidar = self._projector.lidar2depth_map(sample)

        # non set values are nan -> set to 0.0 for training
        depth_map_lidar = np.nan_to_num(depth_map_lidar, nan=0)

        cam_rgb_img = loading_utils.load_camera_image(
            self._loader, sample, self._camera_rgb_sensor)

        cam_depth_img = loading_utils.load_camera_image(
            self._loader, sample, self._camera_depth_sensor)
        # convert to single channel float img
        cam_depth_img = loading_utils.rgb_encoded_depth_to_float(
            cam_depth_img)

        return cam_rgb_img, depth_map_lidar, cam_depth_img

    def visualize(self, sample, vis_dir="/workspace/visualization"):
        # TODO
        vis_dir = pathlib.Path(vis_dir)
        depth_map_lidar = self._projector.lidar2depth_map(sample)

        # for vis replace nan with 0.0
        depth_map_lidar = np.nan_to_num(depth_map_lidar)

        depth_map_lidar = visualization_utils.depth_to_img(
            depth_map_lidar)

        depth__lidar_path = str(vis_dir.joinpath(
            "depth_lidar_{}.jpg".format(i)))
        cv2.imwrite(depth__lidar_path, depth_map_lidar)

        cam_rgb_img = loading_utils.load_camera_image(
            self._loader, sample, self._camera_rgb_sensor)

        rgb_path = str(vis_dir.joinpath(
            "rgb_{}.jpg".format(i)))
        cv2.imwrite(rgb_path, cam_rgb_img)

        cam_depth_img = loading_utils.load_camera_image(
            self._loader, sample, self._camera_depth_sensor)

        cam_depth_img = loading_utils.rgb_encoded_depth_to_float(
            cam_depth_img)
        cam_depth_img = visualization_utils.depth_to_img(cam_depth_img)

        depth__gt_path = str(vis_dir.joinpath(
            "depth_gt_{}.jpg".format(i)))
        cv2.imwrite(depth__gt_path, cam_depth_img)

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth_lidar, depth_gt) the raw data.
        """
        sample = self._data_entries[index]
        return self.load_data(sample)

    def __getitem__(self, index):
        rgb, depth_lidar, depth_gt = self.__getraw__(index)

        input_img = None

        # make single channel h x w x 1
        depth_gt = np.expand_dims(depth_gt, axis=-1)
        # transpose to 1 x h x w
        depth_gt = np.transpose(depth_gt, (2, 0, 1))

        if self.modality == 'rgb':
            input_img = rgb
        elif self.modality == 'rgbd':
            # add depth as 4th channel
            depth_lidar = np.expand_dims(depth_lidar, axis=-1)
            input_img = np.concatenate([rgb, depth_lidar], axis=2)

        # convert to channels x h x w
        input_img = np.transpose(input_img,  (2, 0, 1))
        return input_img, depth_gt

    def __len__(self):
        return len(self._data_entries)
