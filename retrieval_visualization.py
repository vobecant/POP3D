import argparse
import copy
import os
import pickle

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from feature_extraction import res2txt
from train import assign_labels_clip
from visualization.training import set_axes_equal, CLASS_COLORS

IGNORE_LABEL_SEMANTIC = 0
CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
MIN_DIST = 1.
MODEL_NAME = 'ViT-B/16'
TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
    'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.',
    'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
    'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
    'this is the {} in the scene.', 'this is one {} in the scene.',
]

NUSC16 = [
    'barrier', 'bicycle', 'bus', 'car', 'constuction vehicle', 'motorcycle', 'pedestrian', 'traffic cone', 'trailer',
    'truck', 'driveable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]

try:
    with open('/home/vobecant/PhD/TPVFormer-OpenSet/data/info.pth', 'rb') as f:
        INFO = pickle.load(f)
except:
    pass


def project_points(pc_npy):
    projections = {}
    pc_original = LidarPointCloud(
        np.concatenate(
            (pc_npy, np.ones((1, pc_npy.shape[1])))
        )
    )
    for camera_name in CAM_NAMES:
        pc = copy.deepcopy(pc_original)
        im_path = INFO['cams'][camera_name]["data_path"]
        im = np.array(Image.open(im_path))
        imh, imw = im.shape[:2]

        im_name = os.path.split(im_path)[-1].split('.')[0]

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        camera_data = INFO['cams'][camera_name]
        # pc.translate(-np.array(camera_data["ego2global_translation"]))
        # pc.rotate(Quaternion(camera_data["ego2global_rotation"]).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        pc.translate(-np.array(camera_data["sensor2ego_translation"]))
        pc.rotate(Quaternion(camera_data["sensor2ego_rotation"]).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        # Take the actual picture
        # (matrix multiplication with camera-matrix + renormalization).
        points = view_points(
            pc.points[:3, :],
            np.array(camera_data["cam_intrinsic"]),
            normalize=True,
        )
        # assert False, "Check that all the points are mapped to the image and therefore we have a correct number of labels."

        # Remove points that are either outside or behind the camera.
        # Also make sure points are at least 1m in front of the camera to avoid
        # seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        points = points[:2].T
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > MIN_DIST)
        mask = np.logical_and(mask, points[:, 0] > 0)
        mask = np.logical_and(mask, points[:, 0] < imw - 1)
        mask = np.logical_and(mask, points[:, 1] > 0)
        mask = np.logical_and(mask, points[:, 1] < imh - 1)

        # Index of points with a matching pixel (size N)
        matching_points = np.where(mask)[0]
        # For points with a matching pixel, coordinates of that pixel (size N x 2)
        # Use flip for change for (x, y) to (row, column).
        matching_pixels = np.round(
            np.flip(points[matching_points], axis=1)
        ).astype(np.int64)

        points_img = points[mask]
        projections[camera_name] = points_img
    return projections


def zeroshot_classifier(classnames, templates=TEMPLATES, model_name=MODEL_NAME):
    model, preprocess = clip.load(model_name)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.T


class Vox2Real:
    def __init__(self, grid_size, point_cloud_range):
        max_bound = np.asarray(point_cloud_range[-3:])  # 51.2 51.2 3
        min_bound = np.asarray(point_cloud_range[:3])  # -51.2 -51.2 -5
        x_max, y_max, z_max = max_bound
        x_min, y_min, z_min = min_bound
        crop_range = max_bound - min_bound
        vox_size = crop_range / (grid_size)

        x_ = np.linspace(x_min + (vox_size[0] / 2), x_max - (vox_size[0] / 2), grid_size[0])
        y_ = np.linspace(y_min + (vox_size[1] / 2), y_max - (vox_size[1] / 2), grid_size[1])
        z_ = np.linspace(z_min + (vox_size[2] / 2), z_max - (vox_size[2] / 2), grid_size[2])

        self.X, self.Y, self.Z = np.meshgrid(x_, y_, z_, indexing='ij')

    def __call__(self, vox):
        vx, vy, vz = vox
        x, y, z = self.X[vx, vy, vz], self.Y[vx, vy, vz], self.Z[vx, vy, vz]
        xyz = np.stack((x, y, z))
        return xyz


def color_plot(ax, s, xyz, labels):
    x, y, z = xyz
    colors = np.zeros((len(x), 3))
    for cls in np.unique(labels):
        colors[labels == cls] = CLASS_COLORS[cls, :3] / 255.
    ax.scatter(x, y, z, c=colors, s=s, cmap='tab20')
    return colors


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--extra-classes', nargs='+', default=None, type=str)
    parser.add_argument('--scale', default=1, type=int)
    args = parser.parse_args()

    tgt_dir = '/home/vobecant/PhD/TPVFormer-OpenSet/out_qualitative__multi'
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    grid_size = np.array([100, 100, 8]) * args.scale
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    vox2real = Vox2Real(grid_size, point_cloud_range)

    # text_names = ['jeep', 'suv', 'truck', 'umbrella', 'cone', 'car wheel', 'wheel', 'window', 'shop window', 'doors',
    #               'building door', 'campaign van', 'van', 'campaign truck', 'tire', 'crossroad', 'intersection']
    num_extra = len(args.extra_classes)
    text_names = NUSC16 + args.extra_classes
    text_prompts = zeroshot_classifier(text_names).cuda()

    save_dir = args.save_dir

    dirs = [
        os.path.join(args.save_dir, d) for d in os.listdir(args.save_dir) if
        os.path.isdir(os.path.join(args.save_dir, d))
    ]

    dirs = dirs[-2:]

    for d in dirs:

        token = d.split(os.sep)[-1]

        feature_paths = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('ft.pth')]
        assert len(feature_paths) == 1
        feature_path = feature_paths[0]
        image_paths = sorted([os.readlink(os.path.join(d, f)) for f in os.listdir(d) if f.endswith('.jpg')])

        top_idx = [4, 3, 5]
        bottom_idx = [1, 0, 2]

        top_images = np.concatenate([np.array(Image.open(image_paths[idx])) for idx in top_idx], axis=1)
        bottom_images = [np.array(Image.open(image_paths[idx])) for idx in bottom_idx]
        bottom_images = [np.fliplr(im) for im in bottom_images]
        bottom_images = np.concatenate(bottom_images, axis=1)
        images = np.concatenate((top_images, bottom_images), axis=0)

        plt.figure(figsize=(10, 6))
        plt.imshow(images)
        plt.show()

        xyz_path = feature_path.replace('ft.pth', 'xyz.pth')
        xyz_path_gt = feature_path.replace('ft.pth', 'xyz_gt.pth')
        xyz_path_gt_lbl = feature_path.replace('ft.pth', 'xyz_gt_lbl.pth')

        features = torch.load(feature_path).cuda()
        xyz = torch.load(xyz_path)
        xyz = xyz.cpu()
        xyz_gt = torch.load(xyz_path_gt, map_location='cpu')
        xyz_gt_lbl = torch.load(xyz_path_gt_lbl, map_location='cpu')

        # 0) transfer the coordinates from xyz to real world
        xyz_real = vox2real(xyz)
        x, y, z = xyz_real
        pts_cam = project_points(xyz_real)

        xyz_gt_real = vox2real(xyz_gt)
        x_gt, y_gt, z_gt = xyz_gt_real

        # assign zero-shot labels
        text_features = torch.load('/home/vobecant/PhD/MaskCLIP/pretrain/nuscenes_subcategories_ViT16_clip_text.pth',
                                   map_location='cuda')
        if type(text_features) in [tuple, list]:
            text_features, class_mapping_clip = text_features
            learning_map_gt = {1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0, 9: 1,
                               14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7, 12: 8, 22: 9, 23: 10,
                               24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16, 32: 17}
            class_mapping_clip = torch.tensor([learning_map_gt[c.item()] for c in class_mapping_clip]).cuda()
            nonignored = class_mapping_clip != IGNORE_LABEL_SEMANTIC
            text_features_nonignore = text_features[:, nonignored]
            class_mapping_clip_nonignore = class_mapping_clip[nonignored]
        gt_clip_pred, logits = assign_labels_clip(
            features.float().cuda(), text_features_nonignore.T.float(), 1,
            maskclip=True, class_mapping_clip=class_mapping_clip_nonignore, ignore_label=0)
        logits = logits.cpu()

        # 1) zero-shot segmentation
        # compute similarity
        sim = F.conv1d(features.permute(0, 2, 1), text_prompts[[-1], :, None].float()).cpu()
        sim_all = torch.cat((logits, sim), axis=1)
        zero_shot_all = sim_all.argmax(1).squeeze()
        zero_shot_16 = sim_all[:, :16].argmax(1).squeeze()
        fig = plt.figure(figsize=(21, 7))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        s = 0.01 * np.ones(len(zero_shot_all))  # / (args.scale ** 3)
        ax.scatter(x, y, z, c=zero_shot_all, s=s, cmap='tab20')
        set_axes_equal(ax)
        ax.set_title('zero-shot + extra')
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        s = 0.01 * np.ones(len(zero_shot_16))  # / (args.scale ** 3)
        zero_shot_16_colors = color_plot(ax, s, (x, y, z), zero_shot_16 + 1)
        set_axes_equal(ax)
        ax.set_title('zero-shot 16')
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        s = np.ones(len(xyz_gt_lbl))  # / ((args.scale ** 2))
        color_plot(ax, s, (x_gt, y_gt, z_gt), xyz_gt_lbl)
        set_axes_equal(ax)
        ax.set_title('GT labels')
        plt.show()

        out_path = os.path.join(d, f'sim_{token}.txt')
        res2txt(xyz_real, sim[0], out_path=out_path)

        out_path = os.path.join(d, f'zero_shot_all_{token}.txt')
        res2txt(xyz_real, zero_shot_all, out_path=out_path)

        out_path = os.path.join(d, f'zero_shot_16_{token}.txt')
        res2txt(xyz_real, zero_shot_16, out_path=out_path, colors=CLASS_COLORS)

        sim_softmax = sim_all.softmax(1).squeeze()
        sim_softmax_retreived = sim_softmax[0]
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # s = np.ones(len(sim_softmax_retreived)) / args.scale
        # ax.scatter(x, y, z, c=sim_softmax_retreived, s=s, cmap='bwr')
        # set_axes_equal(ax)
        # plt.title('after softmax, retrieved class')
        # plt.show()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        s = np.ones(len(sim_softmax_retreived)) / args.scale
        ax.scatter(x, y, z, c=sim[0], s=s, cmap='bwr')
        set_axes_equal(ax)
        plt.title('retrieved class similarity')
        plt.show()

        out_path = os.path.join(d, f'softmax_{token}.txt')
        res2txt(xyz_real, sim_softmax_retreived, out_path=out_path)

        # 2) single-query retrieval
        for text_prompt, name in zip(text_prompts[:3], text_names[:3]):
            continue
            text_prompt = text_prompt.unsqueeze(0).float()

            # compute similarity
            sim = F.conv1d(features.permute(0, 2, 1), text_prompt[:, :, None])

            out_path = os.path.join(tgt_dir, f'{name}.txt')
            res2txt(xyz, sim, out_path=out_path)
            # print(f'Saved to {out_path}')
            # continue

            sim = sim.cpu()

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            s = np.ones(len(sim)) / args.scale
            ax.scatter(x, y, z, c=sim, s=s)
            set_axes_equal(ax)
            plt.title(name)
            plt.show()
