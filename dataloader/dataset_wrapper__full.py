import itertools

import numba as nb
import numpy as np
import torch
from torch.utils import data

from dataloader.dataset import ImagePoint_NuScenes_withFeatures, ImagePoint_NuScenes_withFeatures_openseg
from dataloader.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)


class Voxelize():
    def __init__(self, max_volume_space, min_volume_space, grid_size):
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.grid_size = grid_size

        self.max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        self.min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5

        # get grid index
        self.crop_range = self.max_bound - self.min_bound

        # TODO: intervals should not minus one.
        # intervals = crop_range / (cur_grid_size - 1)
        self.intervals = self.crop_range / (self.grid_size)

        if (self.intervals == 0).any():
            print("Zero interval!")

    def __call__(self, xyz):
        grid_ind_float = (np.clip(xyz, self.min_bound, self.max_bound - 1e-3) - self.min_bound) / self.intervals
        grid_ind_float = grid_ind_float.astype(float)

        return grid_ind_float


class DatasetWrapper_NuScenes_LinearGT(data.Dataset):
    def __init__(self, in_dataset, grid_size, fill_label=0, ignore_label=255,
                 fixed_volume_space=False, max_volume_space=[51.2, 51.2, 3],
                 min_volume_space=[-51.2, -51.2, -5], phase='train', scale_rate=1, gt_mode='tpvformer',
                 linear_probe=False, unique_features=False,
                 max_features=None,
                 **kwargs):
        'Initialization'
        self.imagepoint_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.fill_label_gt = 17
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.gt_mode = gt_mode
        self.linear_probe = linear_probe
        self.unique_features = unique_features
        self.max_features = max_features  # max allowed number of returned features

        self.voxelization = Voxelize(self.max_volume_space, self.min_volume_space, self.grid_size)

    def __len__(self):
        return len(self.imagepoint_dataset)

    def __getitem__(self, index):
        data = self.imagepoint_dataset[index]
        return self.getitem_feats(data)
        if isinstance(self.imagepoint_dataset, ImagePoint_NuScenes_withFeatures):
            return self.getitem_feats(data)
        imgs, img_metas, xyz, labels, labels_cls = data

        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']

        assert self.fixed_volume_space
        # TODO: grid_ind_float should actually be returned.
        # grid_ind_float = (np.clip(xyz, min_bound, max_bound - 1e-3) - min_bound) / intervals
        grid_ind_float = self.voxelization(xyz)
        grid_ind = np.floor(grid_ind_float).astype(int)
        # print(f'grid_ind_float {np.unique(grid_ind_float)} grid_ind {np.unique(grid_ind)}')

        # process labels
        if self.gt_mode == 'also':
            processed_label, labels, grid_ind_float = self.prepare_gt_occupancy_also(xyz, min_bound, max_bound,
                                                                                     intervals, labels)
        else:
            processed_label = self.prepare_gt_occupancy_tpvformer(grid_ind, labels)
        data_tuple = (imgs, img_metas, processed_label)

        # process SEMANTIC labels
        processed_label_cls = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label_gt
        label_voxel_pair_cls = np.concatenate([grid_ind, labels_cls], axis=1)
        label_voxel_pair_cls = label_voxel_pair_cls[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label_cls = nb_process_label(np.copy(processed_label_cls), label_voxel_pair_cls)

        # data_tuple += (grid_ind, labels)
        data_tuple += (grid_ind_float, labels, processed_label_cls)

        return data_tuple

    def prepare_gt_occupancy_tpvformer(self, grid_ind, semantic_labels):
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
        label_voxel_pair = np.concatenate([grid_ind, semantic_labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        return processed_label

    def getitem_feats(self, data):
        imgs, img_metas, xyz, semantic_labels, semantic_labels_cls, features, matching_points, nonexistent = data

        assert self.fixed_volume_space
        grid_ind_float = self.voxelization(xyz)
        grid_ind = np.floor(grid_ind_float).astype(int)

        # grid for matching points with features
        grid_ind_float_features = self.voxelization(xyz[matching_points])
        grid_ind_features = np.floor(grid_ind_float_features).astype(int)

        # process labels
        processed_label = self.prepare_gt_occupancy_tpvformer(grid_ind, semantic_labels)
        # data_tuple = (imgs, img_metas, processed_label)

        # process SEMANTIC labels
        # processed_label_cls = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label_gt
        # label_voxel_pair_cls = np.concatenate([grid_ind, semantic_labels_cls], axis=1)
        # label_voxel_pair_cls = label_voxel_pair_cls[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # processed_label_cls = nb_process_label(np.copy(processed_label_cls), label_voxel_pair_cls)

        label_feature_pair_cls = np.concatenate([grid_ind_features, features], axis=1, dtype=features.dtype)
        label_feature_pair_cls = label_feature_pair_cls[np.lexsort(
            (grid_ind_features[:, 0], grid_ind_features[:, 1], grid_ind_features[:, 2])), :]
        label_feature_pair_idx = label_feature_pair_cls[:, :3].astype(int)
        label_feature_pair_fts = label_feature_pair_cls[:, 3:]
        processed_features = np.zeros_like(features)
        processed_feature_locations = np.zeros_like(grid_ind_features)
        # processed_features, processed_feature_locations, last_ptr = nb_process_features(
        #     processed_features, processed_feature_locations, label_feature_pair_idx, label_feature_pair_fts,
        #     features.shape[-1])

        processed_feature_locations_float = np.zeros_like(grid_ind_float_features).astype(float)
        label_feature_pair_loc_fts = np.zeros((grid_ind_float_features.shape[0], 3), dtype=features.dtype)
        processed_features, processed_feature_locations, processed_feature_locations_float, last_ptr = nb_process_features(
            processed_features, processed_feature_locations, processed_feature_locations_float,
            label_feature_pair_idx, label_feature_pair_loc_fts, label_feature_pair_fts, features.shape[-1])

        processed_features = processed_features[:last_ptr + 1]
        processed_feature_locations = processed_feature_locations[:last_ptr + 1]
        if self.unique_features:
            unq_ids = np.unique(processed_feature_locations.astype(int), return_index=True)[1]
            processed_feature_locations = processed_feature_locations[unq_ids]
        processed_feature_locations_T = processed_feature_locations.T
        processed_features_labels = processed_label[
            processed_feature_locations_T[0], processed_feature_locations_T[1], processed_feature_locations_T[2]
        ]

        # if self.linear_probe:
        #     grid_ind_features = grid_ind_features.T
        #     semantic_labels = processed_label_cls[grid_ind_features[0], grid_ind_features[1], grid_ind_features[2]]
        #
        # data_tuple += (grid_ind_float, semantic_labels, processed_label_cls)
        # data_tuple += (grid_ind_float_features, features, nonexistent)

        data_tuple = (processed_features, processed_features_labels)

        return data_tuple


class DatasetWrapper_NuScenes(data.Dataset):
    def __init__(self, in_dataset, grid_size, fill_label=0, fill_label_gt=None, ignore_label=255,
                 fixed_volume_space=False, max_volume_space=[51.2, 51.2, 3],
                 min_volume_space=[-51.2, -51.2, -5], phase='train', scale_rate=1, gt_mode='tpvformer',
                 linear_probe=False, with_projections=False, unique_features=False, eval_mode=False,
                 semantic_points=False, max_features=None): 
        self.imagepoint_dataset = in_dataset
        try:
            self.dino_features = self.imagepoint_dataset.dino_features
        except:
            self.dino_features = False
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.fill_label_gt = 17 if fill_label_gt is None else fill_label_gt
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.gt_mode = gt_mode
        self.linear_probe = linear_probe
        self.with_projections = with_projections
        self.semantic_points = semantic_points
        self.unique_features = unique_features
        self.eval_mode = eval_mode
        self.max_features = max_features  # max allowed number of returned features

        self.voxelization = Voxelize(self.max_volume_space, self.min_volume_space, self.grid_size)

        if scale_rate != 1:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
        else:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
        self.transforms = transforms

    def __len__(self):
        return len(self.imagepoint_dataset)

    def __getitem__(self, index):
        data = self.imagepoint_dataset[index]
        # print(index)
        if isinstance(self.imagepoint_dataset, (ImagePoint_NuScenes_withFeatures,
                                                ImagePoint_NuScenes_withFeatures_openseg)):
            if self.eval_mode:
                return self.getitem_feats_eval(data)
            else:
                return self.getitem_feats(data)
        imgs, img_metas, xyz, labels, labels_cls = data

        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']

        assert self.fixed_volume_space
        grid_ind_float = self.voxelization(xyz)
        grid_ind = np.floor(grid_ind_float).astype(int)
        # print(f'grid_ind_float {np.unique(grid_ind_float)} grid_ind {np.unique(grid_ind)}')

        # process labels
        if self.gt_mode == 'also':
            processed_label, labels, grid_ind_float = self.prepare_gt_occupancy_also(xyz, self.voxelization.min_bound,
                                                                                     self.voxelization.max_bound,
                                                                                     labels)
        else:
            processed_label = self.prepare_gt_occupancy_tpvformer(grid_ind, labels)
        data_tuple = (imgs, img_metas, processed_label)

        # process SEMANTIC labels
        processed_label_cls = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label_gt
        try:
            label_voxel_pair_cls = np.concatenate([grid_ind, labels_cls], axis=1)
        except:
            label_voxel_pair_cls = np.concatenate([grid_ind, labels_cls[:, None]], axis=1)
        label_voxel_pair_cls = label_voxel_pair_cls[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label_cls = nb_process_label(np.copy(processed_label_cls), label_voxel_pair_cls)

        processed_label_cls_val, semantic_labels_cls_val, _ = self.prepare_gt_occupancy_also(
            xyz, self.voxelization.min_bound, self.voxelization.max_bound,
            labels_cls, ignore_label=255, freespace_label=self.fill_label_gt, ignore_label_semantics=0)

        # data_tuple += (grid_ind, labels)
        data_tuple += (grid_ind_float, labels, processed_label_cls, processed_label_cls_val)

        return data_tuple

    def getitem_feats(self, data):
        if self.dino_features:
            imgs, img_metas, xyz, semantic_labels, semantic_labels_cls, features, matching_points, nonexistent, \
            features_dino = data
        else:
            imgs, img_metas, xyz, semantic_labels, semantic_labels_cls, features, matching_points, nonexistent = data
            features_dino = None

        if self.max_features is not None and len(matching_points) > self.max_features:
            selected = np.zeros_like(matching_points, dtype=bool)
            selected_idx = np.random.choice(len(matching_points), int(self.max_features), replace=False)
            selected[selected_idx] = True

            # use selected indices
            matching_points = matching_points[selected_idx]
            features = features[selected_idx]
            if features_dino is not None:
                features_dino = features_dino[selected_idx]

        s_im_aug = time.time()
        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']
        t_im_aug = time.time() - s_im_aug

        s_im_aug = time.time()
        assert self.fixed_volume_space
        grid_ind_float = self.voxelization(xyz)
        grid_ind = np.floor(grid_ind_float).astype(int)

        # grid for matching points with features
        grid_ind_float_features = self.voxelization(xyz[matching_points])
        grid_ind_features = np.floor(grid_ind_float_features).astype(int)
        t_im_aug = time.time() - s_im_aug

        s_gt_occ = time.time()
        # process labels
        if self.gt_mode == 'also':
            processed_label, semantic_labels, grid_ind_float, grid_ind_float_features = \
                self.prepare_gt_occupancy_also(xyz, self.voxelization.min_bound, self.voxelization.max_bound,
                                               semantic_labels, xyz[matching_points])
        else:
            processed_label = self.prepare_gt_occupancy_tpvformer(grid_ind, semantic_labels)
        data_tuple = (imgs, img_metas, processed_label)
        t_gt_occ = time.time() - s_gt_occ

        s_sem = time.time()
        # process SEMANTIC labels
        processed_label_cls = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label_gt
        label_voxel_pair_cls = np.concatenate([grid_ind, semantic_labels_cls], axis=1)
        label_voxel_pair_cls = label_voxel_pair_cls[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label_cls = nb_process_label(np.copy(processed_label_cls), label_voxel_pair_cls)
        t_sem = time.time() - s_sem

        s_gt_occ = time.time()
        processed_label_cls_val, semantic_labels_cls_val, _ = self.prepare_gt_occupancy_also(
            xyz, self.voxelization.min_bound, self.voxelization.max_bound,
            semantic_labels_cls, ignore_label=255, freespace_label=self.fill_label_gt, ignore_label_semantics=0)
        t_gt_occ = time.time() - s_gt_occ

        if self.with_projections or self.unique_features:
            assert False, "Not debugged properly."
            label_feature_pair_cls = np.concatenate(
                [grid_ind_features, features, np.arange(grid_ind_features.shape[0])[:, None]],
                axis=1, dtype=features.dtype
            )
            label_feature_pair_cls = label_feature_pair_cls[np.lexsort(
                (grid_ind_features[:, 0], grid_ind_features[:, 1], grid_ind_features[:, 2])), :]
            label_feature_pair_loc = label_feature_pair_cls[:, :3].astype(int)
            label_feature_pair_fts = label_feature_pair_cls[:, 3:-1]
            label_feature_pair_idx = label_feature_pair_cls[:, -1].astype(int)
            label_feature_pair_loc_fts = grid_ind_float_features[label_feature_pair_idx].astype(features.dtype)
            processed_features = np.zeros_like(features)
            processed_feature_locations = np.zeros_like(grid_ind_features)
            processed_feature_locations_float = np.zeros_like(grid_ind_float_features).astype(float)
            processed_features, processed_feature_locations, processed_feature_locations_float, last_ptr = nb_process_features(
                processed_features, processed_feature_locations, processed_feature_locations_float,
                label_feature_pair_loc, label_feature_pair_loc_fts, label_feature_pair_fts, features.shape[-1])
            processed_features = processed_features[:last_ptr + 1]
            processed_feature_locations = processed_feature_locations[:last_ptr + 1]
            processed_feature_locations_float = processed_feature_locations_float[:last_ptr + 1]
            if self.unique_features:
                grid_ind_float_features = processed_feature_locations_float
                features = processed_features

        if self.linear_probe:
            grid_ind_features = grid_ind_features.T
            semantic_labels = processed_label_cls[grid_ind_features[0], grid_ind_features[1], grid_ind_features[2]]

        data_tuple += (grid_ind_float, semantic_labels, processed_label_cls)
        data_tuple += (grid_ind_float_features, features, processed_label_cls_val, matching_points, nonexistent)

        if self.semantic_points:
            data_tuple += (semantic_labels_cls,)

        if self.with_projections:
            data_tuple += (processed_feature_locations,)

        if self.dino_features:
            data_tuple += (features_dino,)

        return data_tuple

    def getitem_feats_eval(self, data):
        imgs, img_metas, xyz, semantic_labels, semantic_labels_cls, features, matching_points, nonexistent = data

        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']

        assert self.fixed_volume_space
        grid_ind_float = self.voxelization(xyz)
        grid_ind = np.floor(grid_ind_float).astype(int)

        # grid for matching points with features
        grid_ind_float_features = self.voxelization(xyz[matching_points])
        grid_ind_features = np.floor(grid_ind_float_features).astype(int)

        # process labels
        processed_label, semantic_labels, grid_ind_float, grid_ind_float_features = \
            self.prepare_gt_occupancy_also(xyz, self.voxelization.min_bound, self.voxelization.max_bound,
                                           semantic_labels, xyz[matching_points])
        data_tuple = (imgs, img_metas, processed_label)

        # process SEMANTIC labels
        # processed_label_cls = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label_gt
        # label_voxel_pair_cls = np.concatenate([grid_ind, semantic_labels_cls], axis=1)
        # label_voxel_pair_cls = label_voxel_pair_cls[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # processed_label_cls = nb_process_label(np.copy(processed_label_cls), label_voxel_pair_cls)
        processed_label_cls, semantic_labels_cls, _ = self.prepare_gt_occupancy_also(
            xyz, self.voxelization.min_bound, self.voxelization.max_bound,
            semantic_labels_cls, ignore_label=255, freespace_label=self.fill_label_gt, ignore_label_semantics=0)

        if self.with_projections or self.unique_features:
            label_feature_pair_cls = np.concatenate(
                [grid_ind_features, features, np.arange(grid_ind_features.shape[0])[:, None]],
                axis=1, dtype=features.dtype
            )
            label_feature_pair_cls = label_feature_pair_cls[np.lexsort(
                (grid_ind_features[:, 0], grid_ind_features[:, 1], grid_ind_features[:, 2])), :]
            label_feature_pair_loc = label_feature_pair_cls[:, :3].astype(int)
            label_feature_pair_fts = label_feature_pair_cls[:, 3:-1]
            label_feature_pair_idx = label_feature_pair_cls[:, -1].astype(int)
            label_feature_pair_loc_fts = grid_ind_float_features[label_feature_pair_idx].astype(features.dtype)
            processed_features = np.zeros_like(features)
            processed_feature_locations = np.zeros_like(grid_ind_features)
            processed_feature_locations_float = np.zeros_like(grid_ind_float_features).astype(float)
            processed_features, processed_feature_locations, processed_feature_locations_float, last_ptr = nb_process_features(
                processed_features, processed_feature_locations, processed_feature_locations_float,
                label_feature_pair_loc, label_feature_pair_loc_fts, label_feature_pair_fts, features.shape[-1])
            processed_features = processed_features[:last_ptr + 1]
            processed_feature_locations = processed_feature_locations[:last_ptr + 1]
            processed_feature_locations_float = processed_feature_locations_float[:last_ptr + 1]
            if self.unique_features:
                grid_ind_float_features = processed_feature_locations_float
                features = processed_features

        if self.linear_probe:
            grid_ind_features = grid_ind_features.T
            semantic_labels = processed_label_cls[grid_ind_features[0], grid_ind_features[1], grid_ind_features[2]]

        data_tuple += (grid_ind_float, semantic_labels, processed_label_cls)
        data_tuple += (grid_ind_float_features, features, matching_points, nonexistent)

        if self.with_projections:
            data_tuple += (processed_feature_locations,)

        return data_tuple

    def prepare_gt_occupancy_tpvformer(self, grid_ind, semantic_labels):
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
        # print(f'grid_ind.shape: {grid_ind.shape}, semantic_labels.shape: {semantic_labels.shape}')
        try:
            label_voxel_pair = np.concatenate([grid_ind, semantic_labels], axis=1)
        except:
            print(f'Didnt work with grid_ind.shape: {grid_ind.shape}, semantic_labels.shape: {semantic_labels.shape}')
            label_voxel_pair = np.concatenate([grid_ind, semantic_labels[:, None]], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        return processed_label

    def prepare_gt_occupancy_also(self, xyz, min_bound, max_bound, labels, xyz_features=None,
                                  ignore_label=None, freespace_label=None, ignore_label_semantics=None):
        if ignore_label is None:
            ignore_label = self.ignore_label
        if freespace_label is None:
            freespace_label = self.fill_label

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * ignore_label

        # step 1: filter points that are outside of the range
        mask = np.ones(xyz.shape[0], dtype=bool)
        mask = np.bitwise_and(mask, xyz[:, 0] > min_bound[0])
        mask = np.bitwise_and(mask, xyz[:, 1] > min_bound[1])
        mask = np.bitwise_and(mask, xyz[:, 2] > min_bound[2])
        mask = np.bitwise_and(mask, xyz[:, 0] < max_bound[0])
        mask = np.bitwise_and(mask, xyz[:, 1] < max_bound[1])
        mask = np.bitwise_and(mask, xyz[:, 2] < max_bound[2])
        xyz = xyz[mask]
        labels = labels[mask]
        if ignore_label_semantics is not None:
            labels[labels == ignore_label_semantics] = ignore_label

        # shoot rays to every point
        # get 100 free points
        xyz_free = (np.linspace(0, 1, 100)[..., None] * xyz[:, None, :]).reshape(-1, 3)
        free_ind_float = self.voxelization(xyz_free)
        free_ind_int = np.floor(free_ind_float).astype(int)

        # # get unique points in the voxel grid
        # _, ind_f, pt2vox = np.unique(
        #     free_ind_int, return_index=True, axis=0, return_inverse=True
        # )
        # free_ind_int = free_ind_int[ind_f, :]
        free_ind_int_T = free_ind_int.T

        occ_ind_float = self.voxelization(xyz)
        occ_ind_int = np.floor(occ_ind_float).astype(int)
        occ_ind_int_T = occ_ind_int.T

        # assign free-space label
        processed_label[free_ind_int_T[0], free_ind_int_T[1], free_ind_int_T[2]] = freespace_label

        try:
            label_voxel_pair_cls = np.concatenate([occ_ind_int, labels], axis=1)
        except:
            label_voxel_pair_cls = np.concatenate([occ_ind_int, labels[:, None]], axis=1)
        label_voxel_pair_cls = label_voxel_pair_cls[
                               np.lexsort((occ_ind_int[:, 0], occ_ind_int[:, 1], occ_ind_int[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair_cls)

        if xyz_features is None:
            return processed_label, labels, occ_ind_float
        else:
            # filter points that are outside of the range
            mask = np.ones(xyz_features.shape[0], dtype=bool)
            mask = np.bitwise_and(mask, xyz_features[:, 0] > min_bound[0])
            mask = np.bitwise_and(mask, xyz_features[:, 1] > min_bound[1])
            mask = np.bitwise_and(mask, xyz_features[:, 2] > min_bound[2])
            mask = np.bitwise_and(mask, xyz_features[:, 0] < max_bound[0])
            mask = np.bitwise_and(mask, xyz_features[:, 1] < max_bound[1])
            mask = np.bitwise_and(mask, xyz_features[:, 2] < max_bound[2])
            xyz_features = xyz_features[mask]
            grid_ind_float_features = self.voxelization(xyz_features)
            # grid_ind_features = np.floor(grid_ind_float_features).astype(int)
            return processed_label, labels, occ_ind_float, grid_ind_float_features


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


@nb.jit(
    'Tuple((float32[:,:],int64[:,:],float32[:,:],int64))(float32[:,:],int64[:,:],float32[:,:],int64[:,:],float32[:,:],float32[:,:],int64)',
    nopython=True, cache=True, parallel=False)
def nb_process_features(processed_features, processed_feature_locations, processed_feature_locations_float,
                        position_int, position_float, features, feature_size):
    feature_sum = features[0]
    num_features = 1
    cur_sear_ind = position_int[0]
    cur_sear_ind_float = position_float[0]
    cur_pos_float_sum = position_float[0]
    pointer_cur = 0
    processed_feature_locations[pointer_cur] = cur_sear_ind
    processed_feature_locations_float[pointer_cur] = cur_sear_ind_float
    for i in range(1, features.shape[0]):
        cur_ind = position_int[i]
        cur_ind_float = position_float[i]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            # new location, we need to
            mean_feature = feature_sum / num_features
            mean_pos_float = cur_pos_float_sum / num_features
            processed_features[pointer_cur] = mean_feature
            processed_feature_locations[pointer_cur] = cur_sear_ind
            processed_feature_locations_float[pointer_cur] = mean_pos_float
            feature_sum = np.zeros((feature_size,), dtype=processed_features.dtype)
            cur_pos_float_sum = np.zeros((3,), dtype=processed_feature_locations_float.dtype)
            cur_sear_ind = cur_ind
            num_features = 0
            pointer_cur += 1
        feature_sum += features[i]
        cur_pos_float_sum += cur_ind_float
        num_features += 1
    processed_features[pointer_cur] = feature_sum / num_features
    processed_feature_locations[pointer_cur] = cur_sear_ind
    processed_feature_locations_float[pointer_cur] = cur_pos_float_sum / num_features
    # features_out = np.stack(features_list)
    # return features_out
    return processed_features, processed_feature_locations, processed_feature_locations_float, pointer_cur


def custom_collate_fn(data):
    img2stack = np.stack([d[0] for d in data]).astype(np.float32)
    meta2stack = [d[1] for d in data]
    label2stack = np.stack([d[2] for d in data]).astype(int)
    # because we use a batch size of 1, so we can stack these tensor together.
    grid_ind_stack = np.stack([d[3] for d in data]).astype(float)
    point_label = np.stack([d[4] for d in data]).astype(int)
    point_label_cls = np.stack([d[5] for d in data]).astype(int)

    if len(data[0]) == 7:
        vox_labels_eval = np.stack([d[6] for d in data]).astype(int)
        return torch.from_numpy(img2stack), \
               meta2stack, \
               torch.from_numpy(label2stack), \
               torch.from_numpy(grid_ind_stack), \
               torch.from_numpy(point_label), \
               torch.from_numpy(point_label_cls), torch.from_numpy(vox_labels_eval)

    elif len(data[0]) > 6:
        grid_ind_ft_stack = np.stack([d[6] for d in data]).astype(float)
        point_ft = np.stack([d[7] for d in data]).astype(float)
        vox_labels_eval = np.stack([d[8] for d in data]).astype(int)
        list_of_list_matching_points = [d[9] for d in data]
        matching_points = list(itertools.chain(list_of_list_matching_points))
        list_of_list_nonexistent = [d[10] for d in data]
        nonexistent = list(itertools.chain(list_of_list_nonexistent))
        return_tuple = (
            torch.from_numpy(img2stack), meta2stack, torch.from_numpy(label2stack), torch.from_numpy(grid_ind_stack),
            torch.from_numpy(point_label), torch.from_numpy(point_label_cls), torch.from_numpy(grid_ind_ft_stack),
            torch.from_numpy(point_ft), torch.from_numpy(vox_labels_eval), matching_points, nonexistent
        )
        if len(data[0]) == 12:
            return_tuple += (torch.from_numpy(np.stack([d[11] for d in data]).astype(int)),)
        if len(data[0]) == 13:
            return_tuple += (torch.from_numpy(np.stack([d[12] for d in data]).astype(int)),)
        return return_tuple
    else:
        return torch.from_numpy(img2stack), \
               meta2stack, \
               torch.from_numpy(label2stack), \
               torch.from_numpy(grid_ind_stack), \
               torch.from_numpy(point_label), \
               torch.from_numpy(point_label_cls)


def custom_collate_fn_dino(data):
    img2stack = np.stack([d[0] for d in data]).astype(float)
    meta2stack = [d[1] for d in data]
    label2stack = np.stack([d[2] for d in data]).astype(int)
    # because we use a batch size of 1, so we can stack these tensor together.
    grid_ind_stack = np.stack([d[3] for d in data]).astype(float)
    point_label = np.stack([d[4] for d in data]).astype(int)
    point_label_cls = np.stack([d[5] for d in data]).astype(int)

    if len(data[0]) == 7:
        vox_labels_eval = np.stack([d[6] for d in data]).astype(int)
        return torch.from_numpy(img2stack), \
               meta2stack, \
               torch.from_numpy(label2stack), \
               torch.from_numpy(grid_ind_stack), \
               torch.from_numpy(point_label), \
               torch.from_numpy(point_label_cls), torch.from_numpy(vox_labels_eval)

    elif len(data[0]) > 6:
        grid_ind_ft_stack = np.stack([d[6] for d in data]).astype(float)
        point_ft = np.stack([d[7] for d in data]).astype(float)
        vox_labels_eval = np.stack([d[8] for d in data]).astype(int)
        list_of_list_matching_points = [d[9] for d in data]
        matching_points = list(itertools.chain(list_of_list_matching_points))
        list_of_list_nonexistent = [d[10] for d in data]
        nonexistent = list(itertools.chain(list_of_list_nonexistent))
        point_ft_dino = np.stack([d[11] for d in data]).astype(float)
        return_tuple = (
            torch.from_numpy(img2stack), meta2stack, torch.from_numpy(label2stack), torch.from_numpy(grid_ind_stack),
            torch.from_numpy(point_label), torch.from_numpy(point_label_cls), torch.from_numpy(grid_ind_ft_stack),
            torch.from_numpy(point_ft), torch.from_numpy(vox_labels_eval), torch.from_numpy(point_ft_dino),
            matching_points, nonexistent
        )
        if len(data[0]) == 12:
            return_tuple += (torch.from_numpy(np.stack([d[11] for d in data]).astype(int)),)
        if len(data[0]) == 13:
            return_tuple += (torch.from_numpy(np.stack([d[12] for d in data]).astype(int)),)
        return return_tuple
    else:
        return torch.from_numpy(img2stack), \
               meta2stack, \
               torch.from_numpy(label2stack), \
               torch.from_numpy(grid_ind_stack), \
               torch.from_numpy(point_label), \
               torch.from_numpy(point_label_cls)


def custom_collate_fn_linear_gt(data):
    features = torch.from_numpy(np.concatenate([d[0] for d in data])).T.unsqueeze(0)
    labels = torch.from_numpy(np.concatenate([d[1] for d in data])).unsqueeze(0).long()
    # labels = torch.from_numpy(np.concatenate([d[4] for d in data])).unsqueeze(0).long()
    # features = torch.from_numpy(np.concatenate([d[7] for d in data])).T.unsqueeze(0)
    return features, labels
