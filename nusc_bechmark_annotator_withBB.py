import argparse
import copy
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, filter_colors
from nuscenes.panoptic.panoptic_utils import generate_panoptic_colors
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from benchmark_retrieval import panoptic2semAndInst

MIN_DIST = 1.

NAME2COLOR = {  # RGB.
    "noise": (0, 0, 0),  # Black.
    "animal": (70, 130, 180),  # Steelblue
    "human.pedestrian.adult": (0, 0, 230),  # Blue
    "human.pedestrian.child": (135, 206, 235),  # Skyblue,
    "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
    "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
    "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
    "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
    "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
    "movable_object.barrier": (112, 128, 144),  # Slategrey
    "movable_object.debris": (210, 105, 30),  # Chocolate
    "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
    "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
    "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
    "vehicle.bicycle": (220, 20, 60),  # Crimson
    "vehicle.bus.bendy": (255, 127, 80),  # Coral
    "vehicle.bus.rigid": (255, 69, 0),  # Orangered
    "vehicle.car": (255, 158, 0),  # Orange
    "vehicle.construction": (233, 150, 70),  # Darksalmon
    "vehicle.emergency.ambulance": (255, 83, 0),
    "vehicle.emergency.police": (255, 215, 0),  # Gold
    "vehicle.motorcycle": (255, 61, 99),  # Red
    "vehicle.trailer": (255, 140, 0),  # Darkorange
    "vehicle.truck": (255, 99, 71),  # Tomato
    "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
    "flat.other": (175, 0, 75),
    "flat.sidewalk": (75, 0, 75),
    "flat.terrain": (112, 180, 60),
    "static.manmade": (222, 184, 135),  # Burlywood
    "static.other": (255, 228, 196),  # Bisque
    "static.vegetation": (0, 175, 0),  # Green
    "vehicle.ego": (255, 240, 245)
}

NAME2IDX = {'noise': 0, 'animal': 1, 'human.pedestrian.adult': 2, 'human.pedestrian.child': 3,
            'human.pedestrian.construction_worker': 4, 'human.pedestrian.personal_mobility': 5,
            'human.pedestrian.police_officer': 6, 'human.pedestrian.stroller': 7, 'human.pedestrian.wheelchair': 8,
            'movable_object.barrier': 9, 'movable_object.debris': 10, 'movable_object.pushable_pullable': 11,
            'movable_object.trafficcone': 12, 'static_object.bicycle_rack': 13, 'vehicle.bicycle': 14,
            'vehicle.bus.bendy': 15, 'vehicle.bus.rigid': 16, 'vehicle.car': 17, 'vehicle.construction': 18,
            'vehicle.emergency.ambulance': 19, 'vehicle.emergency.police': 20, 'vehicle.motorcycle': 21,
            'vehicle.trailer': 22, 'vehicle.truck': 23, 'flat.driveable_surface': 24, 'flat.other': 25,
            'flat.sidewalk': 26, 'flat.terrain': 27, 'static.manmade': 28, 'static.other': 29, 'static.vegetation': 30,
            'vehicle.ego': 31}
IDX2NAME = {idx: name for name, idx in NAME2IDX.items()}

IDX2COLOR = [NAME2COLOR[IDX2NAME[idx]] for idx in range(len(IDX2NAME))]


def paint_panop_points_label(panoptic_labels,
                             filter_panoptic_labels=None,
                             name2idx=NAME2IDX,
                             colormap=NAME2COLOR) -> np.ndarray:
    """
    Paint each label in a pointcloud with the corresponding RGB value, and set the opacity of the labels to
    be shown to 1 (the opacity of the rest will be set to 0); e.g.:
        [30, 5, 12, 34, ...] ------> [[R30, G30, B30, 0], [R5, G5, B5, 1], [R34, G34, B34, 1], ...]
    :param panoptic_labels: An array containing the labels.
    :param filter_panoptic_labels: The labels for which to set opacity to zero; this is to hide those points
                                   thereby preventing them from being displayed.
    :param name2idx: A dictionary containing the mapping from class names to class indices.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :return: A numpy array which has length equal to the number of points in the pointcloud, and each value is
             a RGBA array.
    """
    # Given a colormap (class name -> RGB color) and a mapping from class name to class index,
    # get an array of RGB values where each color sits at the index in the array corresponding
    # to the class index.
    colors = generate_panoptic_colors(colormap, name2idx)  # Shape: [num_instances, 3]

    if filter_panoptic_labels is not None:
        # Ensure that filter_panoptic_labels is an iterable.
        assert isinstance(filter_panoptic_labels, (list, np.ndarray)), \
            'Error: filter_panoptic_labels should be a list of class indices, eg. [9], [10, 21].'

        # Check that class indices in filter_panoptic_labels are valid.
        assert all([0 <= x < len(name2idx) for x in filter_panoptic_labels]), \
            f'All class indices in filter_panoptic_labels should be between 0 and {len(name2idx) - 1}'

        # Filter to get only the colors of the desired classes; this is done by setting the
        # alpha channel of the classes to be viewed to 1, and the rest to 0.
        colors = np.concatenate((colors, np.ones((colors.shape[0], 1))), 1)
        for id_i in np.unique(panoptic_labels):  # Shape: [num_class, 4]
            if id_i // 1000 not in filter_panoptic_labels:
                colors[id_i, -1] = 0.0

    coloring = colors[panoptic_labels]  # Shape: [num_points, 4]

    return coloring


def paint_semantic_label(points_label, filter_lidarseg_labels=None,
                         name2idx=NAME2IDX, colormap=NAME2COLOR) -> np.ndarray:
    """
    Paint each label in a pointcloud with the corresponding RGB value, and set the opacity of the labels to
    be shown to 1 (the opacity of the rest will be set to 0); e.g.:
        [30, 5, 12, 34, ...] ------> [[R30, G30, B30, 0], [R5, G5, B5, 1], [R34, G34, B34, 1], ...]
    :param lidarseg_labels_filename: Path to the .bin file containing the labels.
    :param filter_lidarseg_labels: The labels for which to set opacity to zero; this is to hide those points
                                   thereby preventing them from being displayed.
    :param name2idx: A dictionary containing the mapping from class names to class indices.
    :param colormap: A dictionary containing the mapping from class names to RGB values.
    :return: A numpy array which has length equal to the number of points in the pointcloud, and each value is
             a RGBA array.
    """
    # Given a colormap (class name -> RGB color) and a mapping from class name to class index,
    # get an array of RGB values where each color sits at the index in the array corresponding
    # to the class index.
    colors = colormap_to_colors(colormap, name2idx)  # Shape: [num_class, 3]

    if filter_lidarseg_labels is not None:
        # Ensure that filter_lidarseg_labels is an iterable.
        assert isinstance(filter_lidarseg_labels, (list, np.ndarray)), \
            'Error: filter_lidarseg_labels should be a list of class indices, eg. [9], [10, 21].'

        # Check that class indices in filter_lidarseg_labels are valid.
        assert all([0 <= x < len(name2idx) for x in filter_lidarseg_labels]), \
            'All class indices in filter_lidarseg_labels should ' \
            'be between 0 and {}'.format(len(name2idx) - 1)

        # Filter to get only the colors of the desired classes; this is done by setting the
        # alpha channel of the classes to be viewed to 1, and the rest to 0.
        colors = filter_colors(colors, filter_lidarseg_labels)  # Shape: [num_class, 4]

    # Paint each label with its respective RGBA value.
    coloring = colors[points_label]  # Shape: [num_points, 4]

    return coloring


def get_infos(splits):
    infos = []
    if isinstance(splits, str):
        splits = [splits]
    if 'train' in splits:
        with open('/home/vobecant/PhD/TPVFormer-OpenSet/data/nuscenes_infos_train_new.pkl', 'rb') as f:
            infos += pickle.load(f)
    elif 'val' in splits:
        with open('/home/vobecant/PhD/TPVFormer-OpenSet/data/nuscenes_infos_val_new.pkl', 'rb') as f:
            infos += pickle.load(f)
    elif 'test' in splits:
        with open('/home/vobecant/PhD/TPVFormer-OpenSet/data/nuscenes_infos_test.pkl', 'rb') as f:
            infos += pickle.load(f)
    return infos


def get_lidar_token_lut(nusc_root):
    # given token, return a lidar token
    start = time.time()
    with open(os.path.join(nusc_root, 'v1.0-trainval/sample.pkl'), 'rb') as f:
        sample_trainval = [('trainval', sample) for sample in pickle.load(f)]
    end1 = time.time()
    elapsed_trainval = end1 - start
    print('Loaded trainval data in {:.1f}s!'.format(elapsed_trainval))
    with open(os.path.join(nusc_root, 'v1.0-test/sample.pkl'), 'rb') as f:
        sample_test = [('test', sample) for sample in pickle.load(f)]
    elapsed_test = time.time() - end1
    print('Loaded test data in {:.1f}s!'.format(elapsed_test))

    # CALIBRATED SENSOR
    with open(os.path.join(nusc_root, 'v1.0-trainval/calibrated_sensor.json'), 'r') as f:
        calibrated_sensor_trainval = json.load(f)
    with open(os.path.join(nusc_root, 'v1.0-test/calibrated_sensor.json'), 'r') as f:
        calibrated_sensor_test = json.load(f)
    calibrated_sensor = calibrated_sensor_trainval + calibrated_sensor_test

    with open('/home/vobecant/PhD/TPVFormer-OpenSet/data/token2ind_calibrated_sensor_trainval.pkl', 'rb') as f:
        token2ind_calibrated_sensor_trainval = pickle.load(f)
    with open('/home/vobecant/PhD/TPVFormer-OpenSet/data/token2ind_calibrated_sensor_test.pkl', 'rb') as f:
        token2ind_calibrated_sensor_test = pickle.load(f)

    table = sample_trainval + sample_test
    token2ind = dict()

    for ind, member in enumerate(table):
        token2ind[member[1]['token']] = ind

    def lidar_token_lut(token):
        ind = token2ind[token]
        split, _data = table[ind]
        lidar_token = _data['data']['LIDAR_TOP']
        return split, lidar_token

    return lidar_token_lut


def lidarseg_lut(nusc_root, lidar_token, split):
    return os.path.join(nusc_root, f'lidarseg/v1.0-{split}/{lidar_token}_lidarseg.bin')


def panoptic_lut(nusc_root, lidar_token, split):
    return os.path.join(nusc_root, f'panoptic/v1.0-{split}/{lidar_token}_panoptic.npz')


def main(args):
    data_path = args.data_root
    infos = get_infos(args.splits)

    lidar_token_lut = get_lidar_token_lut(data_path)

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for ii, info in enumerate(infos):
        token = info['token']
        split, lidar_sd_token = lidar_token_lut(token)  # nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        if args.debug:
            print(f'Index {ii}, lidar token: {lidar_sd_token}, split: {split}')
        lidar_path = info['lidar_path']
        # points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
        pc_original = LidarPointCloud.from_file(lidar_path)
        pc_original.rotate(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
        pc_original.translate(np.array(info["lidar2ego_translation"]))
        pc_original.rotate(Quaternion(info["ego2global_rotation"]).rotation_matrix)
        pc_original.translate(np.array(info["ego2global_translation"]))

        # lidarseg_labels_filename = os.path.join(data_path, nusc.get('lidarseg', lidar_sd_token)['filename'])
        lidarseg_labels_filename = os.path.join(data_path, lidarseg_lut(args.data_root, lidar_sd_token, split))
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        if args.debug:
            print(f'Label statistics: {np.unique(points_label, return_counts=True)}')

        # panoptic_labels_filename = os.path.join(data_path, nusc.get('panoptic', lidar_sd_token)['filename'])
        panoptic_labels_filename = os.path.join(data_path, panoptic_lut(args.data_root, lidar_sd_token, split))
        panoptic_labels = load_bin_file(panoptic_labels_filename, type='panoptic')
        semantic, instance = panoptic2semAndInst(panoptic_labels)

        # denotes whether a point projects into some camera or not
        visibility_mask = np.zeros(len(semantic.squeeze()), dtype=bool)

        if args.show:
            f, axs = plt.subplots(6, 3, figsize=(16, 20))

        for cam_idx, (cam_name, cam_data) in enumerate(info['cams'].items()):
            pc = copy.deepcopy(pc_original)
            imgfile = cam_data['data_path'].replace('./data/nuscenes', args.data_root)
            img = np.asarray(Image.open(imgfile))
            imh, imw = img.shape[:2]

            pc.translate(-np.array(cam_data["ego2global_translation"]))
            pc.rotate(Quaternion(cam_data["ego2global_rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            pc.translate(-np.array(cam_data["sensor2ego_translation"]))
            pc.rotate(Quaternion(cam_data["sensor2ego_rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cam_data["cam_intrinsic"]),
                normalize=True,
            )

            points = points[:2].T

            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > MIN_DIST)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < imw - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < imh - 1)

            visibility_mask = np.logical_or(visibility_mask, mask)

            # Index of points with a matching pixel (size N)
            matching_points = np.where(mask)[0]

            # For points with a matching pixel, coordinates of that pixel (size N x 2)
            # Use flip for change for (x, y) to (row, column).
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)

            semantic_cam = semantic[matching_points]
            instance_cam = instance[matching_points]
            panoptic_cam = panoptic_labels[matching_points]

            if args.show:
                y, x = matching_pixels.T

                # f, axs = plt.subplots(2, 2, figsize=(12, 8))

                semantic_colors = paint_semantic_label(semantic_cam)
                panoptic_colors = paint_panop_points_label(panoptic_cam)

                axs[cam_idx, 0].imshow(img)
                axs[cam_idx, 0].set_title(cam_name)
                axs[cam_idx, 1].imshow(img)
                axs[cam_idx, 1].scatter(x, y, c=semantic_colors, s=2., cmap='tab20')
                axs[cam_idx, 1].set_title(f'classes: {np.unique(semantic_cam)}')
                axs[cam_idx, 2].imshow(img)
                axs[cam_idx, 2].scatter(x, y, c=panoptic_colors, s=2., cmap='tab20')

        if args.show:
            plt.suptitle(lidar_path)
            plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--data-root', default='/nfs/datasets/nuscenes', type=str)
    parser.add_argument('--splits', default=['val'], type=str, nargs='+')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
