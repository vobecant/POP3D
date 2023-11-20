import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mmcv import Config
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from tqdm import tqdm

from benchmark_retrieval import get_args, SET2IMAGES, limit_dataloader, get_img2scene_lut, get_dataloader, \
    panoptic2semAndInst

MIN_DIST = 1.


def limit_infos(new_infos_loader, image_names, img2scene_lut, lut_infos):
    indices = [img2scene_lut.get(img_name.split('.')[0], None) for img_name in image_names]
    indices = [i for i in indices if i is not None]
    new_infos_loader = [new_infos_loader[i] for i in indices]
    new_infos_loader_lut = [lut_infos[i] for i in indices]
    return new_infos_loader


def main(args, cfg):
    # new_infos, lut, lut_split = get_img2scene_lut()
    new_infos, lut = get_img2scene_lut(splits=args.split)
    _, val_dataloader = get_dataloader(cfg, retrieval=True, no_features=args.no_features, no_nusc=args.no_nusc)
    val_dataloader.dataset.imagepoint_dataset.class_agnostic = False

    set_name = ''
    image_paths = None
    if not args.no_retrieval and args.set_name is not None:
        set_name = args.set_name.lower()
        args.text_embeddings_path = f'/home/vobecant/PhD/MaskCLIP/pretrain/{set_name}_ViT16_clip_text.pth'
        image_names = SET2IMAGES[set_name]  # ['n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119612404.jpg']
        val_dataloader, image_paths = limit_dataloader(val_dataloader, image_names, lut, new_infos)
        new_infos_loader = val_dataloader.dataset.imagepoint_dataset.nusc_infos
    nusc = val_dataloader.dataset.imagepoint_dataset.nusc
    data_path = nusc.dataroot

    lidar_sd_token = nusc.get('sample', new_infos[0]['token'])['data']['LIDAR_TOP']

    indices = []
    cameras = []
    if args.cls_num is not None and args.set_name is None:
        for ii, info in tqdm(enumerate(new_infos)):
            if ii == 986: continue
            lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(data_path, nusc.get('lidarseg', lidar_sd_token)['filename'])
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            bool_cls = (points_label == args.cls_num)
            num_cls_pts = (bool_cls).sum()
            if num_cls_pts >= args.min_num_pts:
                print(f'index {ii}, {num_cls_pts} points of class {args.cls_num}, LiDAR token: {lidar_sd_token}')
                lidar_path = info['lidar_path']
                # xyz = LidarPointCloud.from_file(lidar_path).points
                # lidar_path_name = os.path.split(lidar_path)[1]
                # show3d(xyz, plt.figure(), 1, 1, 1, labels=bool_cls, cmap_name='bwr', s=0.002, title=lidar_path_name)
                # plt.show()
                indices.append(ii)
                print(f'Index {ii}, image_name: {info["cams"]["CAM_FRONT"]["data_path"]}')
            del points_label
        indices = set(indices)
        new_infos_loader = [new_infos[ii] for ii in indices]

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    semantic_idx = [18]
    instance_idx = [11]
    # assert len(new_infos_loader) == len(semantic_idx) == len(instance_idx)
    for ii, info in enumerate(new_infos_loader):
        lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        print(f'Index {ii}, lidar token: {lidar_sd_token}')
        lidar_path = info['lidar_path']
        # points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
        pc_original = LidarPointCloud.from_file(lidar_path)
        pc_original.rotate(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
        pc_original.translate(np.array(info["lidar2ego_translation"]))
        pc_original.rotate(Quaternion(info["ego2global_rotation"]).rotation_matrix)
        pc_original.translate(np.array(info["ego2global_translation"]))

        lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(data_path, nusc.get('lidarseg', lidar_sd_token)['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        print(f'Label statistics: {np.unique(points_label, return_counts=True)}')
        panoptic_labels_filename = os.path.join(data_path, nusc.get('panoptic', lidar_sd_token)['filename'])
        panoptic_labels = load_bin_file(panoptic_labels_filename, type='panoptic')
        semantic, instance = panoptic2semAndInst(panoptic_labels)

        whr_sem = semantic == args.cls_num  # semantic_idx[ii]
        whr_both = whr_sem  # np.bitwise_and(whr_sem, whr_inst)
        # whr_sem = semantic == semantic_idx[ii]
        # whr_inst = instance == instance_idx[ii]
        # whr_both = np.bitwise_and(whr_sem, whr_inst)
        retrieval_tgts = whr_both.astype(np.uint8)

        if args.save_dir is not None and (
                args.set_name is not None or args.set_name_save is not None) and args.cls_num is not None:
            name = args.set_name if args.set_name is not None else args.set_name_save
            save_dir = args.save_dir
            tgt_path = os.path.join(save_dir, f'{name}{ii}_tgt.npy')
            np.save(tgt_path, retrieval_tgts)
            print(f'Saved targets to {tgt_path}')

        for cam_name, cam_data in info['cams'].items():
            pc = copy.deepcopy(pc_original)
            imgfile = cam_data['data_path'].replace('./data', args.data_root)
            # img = cv2.imread(imgfile)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

            # Index of points with a matching pixel (size N)
            matching_points = np.where(mask)[0]
            # For points with a matching pixel, coordinates of that pixel (size N x 2)
            # Use flip for change for (x, y) to (row, column).
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)

            semantic_cam = semantic[matching_points]
            instance_cam = instance[matching_points]
            retrieval_cam = retrieval_tgts[matching_points]

            title = None
            print(f'[{cam_name}] args.cls_num in semantic_cam: {args.cls_num in semantic_cam}')
            if args.cls_num in semantic_cam:
                instance_cam_cls = instance_cam[semantic_cam == args.cls_num]
                idx, count = np.unique(instance_cam_cls, return_counts=True)
                title = ','.join([f'{_idx}: {c}' for _idx, c in zip(idx, count)])
                print(title)

            if args.show:
                y, x = matching_pixels.T

                f, axs = plt.subplots(2, 2, figsize=(12, 8))

                axs[0, 0].imshow(img)
                axs[0, 1].imshow(img)
                axs[0, 1].scatter(x, y, c=semantic_cam, s=2., cmap='tab20')
                axs[0, 1].set_title(f'classes: {np.unique(semantic_cam)}')
                axs[1, 0].imshow(img)
                axs[1, 0].scatter(x, y, c=instance_cam, s=2., cmap='tab20')
                if title is not None:
                    axs[1, 0].set_title(title)

                if args.cls_num is not None:
                    axs[1, 1].imshow(img)
                    num_anns_cam = retrieval_cam.sum()
                    axs[1, 1].scatter(x, y, c=retrieval_cam, s=2., cmap='gray')
                    axs[1, 1].set_title(num_anns_cam)
                plt.suptitle(lidar_path + '\n' + imgfile)
                plt.show()

            # print()


if __name__ == '__main__':
    args = get_args()
    cfg = Config.fromfile(args.py_config)
    cfg.resume_from = args.resume_from
    main(args, cfg)
