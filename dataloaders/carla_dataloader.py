import os
import os.path
import numpy as np
import pathlib
import random

import torch
import cv2
import tqdm
import torch.utils.data as data
import h5py
import dataloaders.transforms as transforms


from dataset_3d.data_loaders import dataset_loader
from dataset_3d.utils.projection_utils import LidarToCameraProjector
import dataset_3d.utils.loading_utils as loading_utils
import dataset_3d.utils.visualization_utils as visualization_utils


to_tensor = transforms.ToTensor()


class CarlaDataset(data.Dataset):
    seed = 42

    output_size = (228, 912)
    _modality_names = ['rgb', 'rgbd']
    _color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    # as the crop is hardcoded we assert a given image size to prevent errors
    _input_width = 1600
    _input_height = 900

    _crop_upper_x = 0
    _crop_upper_y = 150

    _crop_width = 1600
    _crop_height = 750

    _road_crop = (_crop_upper_y, _crop_upper_x, _crop_height, _crop_width)

    def __init__(self, root, type, modality="rgbd", camera_rgb_name="cam_front", camera_depth_name="cam_front_depth", lidar_name="lidar_top", ego_pose_sensor_name="imu_perfect"):

        random.seed(self.seed)

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

        # check image sizes
        assert self._camera_rgb_sensor.meta['image_size_x'] == self._input_width, "crop does not match input images, pls adapt."
        assert self._camera_rgb_sensor.meta['image_size_y'] == self._input_height, "crop does not match input images, pls adapt."

        assert self._camera_depth_sensor.meta['image_size_x'] == self._input_width, "crop does not match input images, pls adapt."
        assert self._camera_depth_sensor.meta['image_size_y'] == self._input_height, "crop does not match input images, pls adapt."

        self._data_entries = self.prepare_dataset()

        if type == 'train':
            self._transform = self._train_transform
        elif type == 'val':
            self._transform = self._val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))

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
        # random.shuffle(data_entries)
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

    def visualize(self, sample, vis_dir="/workspace/visualization", i=0):
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

        # if self.modality == 'rgb':
        #     input_img = rgb
        # elif self.modality == 'rgbd':
        #     # add depth as 4th channel
        #     depth_lidar = np.expand_dims(depth_lidar, axis=-1)
        #     input_img = np.concatenate([rgb, depth_lidar], axis=2)

        # apply transforms (for data augmentation)
        if self._transform is not None:
            input_img, sparse_depth, depth_gt = self._transform(
                rgb, depth_lidar, depth_gt)
        else:
            raise(RuntimeError("transform not defined"))

        # convert to torch and flip channels in front
        input_img = to_tensor(input_img)
        depth_gt = to_tensor(depth_gt)

        return input_img, depth_gt

    def __len__(self):
        return len(self._data_entries)

    def _train_transform(self, rgb, sparse_depth, depth_gt):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_gt = depth_gt / s

        # TODO critical why is the input not scaled in original implementation?
        sparse_depth = sparse_depth / s

        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        # TODO critical adjust sizes
        transform = transforms.Compose([
            transforms.Crop(*self._road_crop),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])

        print("orig shape =", rgb.shape)
        rgb = transform(rgb)
        print("crop shape =", rgb.shape)
        sparse_depth = transform(sparse_depth)

        # TODO needed?
        # Scipy affine_transform produced RuntimeError when the depth map was
        # given as a 'numpy.ndarray'
        depth_gt = np.asfarray(depth_gt, dtype='float32')
        depth_gt = transform(depth_gt)

        rgb = self._color_jitter(rgb)  # random color jittering

        cv2.imwrite("/workspace/visualization/rgb.png", rgb)

        # convert color [0,255] -> [0.0, 1.0] floats
        rgb = np.asfarray(rgb, dtype='float') / 255

        return rgb, sparse_depth, depth_gt

    def _val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np
