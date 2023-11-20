import argparse
import copy
import os
import pickle

import numpy as np
import torch
from PIL import Image
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from tqdm import tqdm

FEATURES_TYPE = 'k'
MODEL_SPEC = 'vit_small_8'
VERSION = "mini"  # "trainval"
# VERSION = "trainval"
CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
DEBUG = True


def prepare_pc2scene(pc2scene_path, args):
    savepath = pc2scene_path.replace('.pkl', f'_{args.version}.pkl')
    if os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            pc2scene = pickle.load(f)
        return pc2scene

    nusc = NuScenes(
        version=f"v1.0-{args.version}",
        dataroot=args.nusc_root,
        verbose=True
    )

    pc2scene = {}
    for scene in tqdm(nusc.scene):
        name = scene['name']
        current_sample_token = scene["first_sample_token"]
        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)
            data = current_sample["data"]
            current_sample_token = current_sample["next"]

            # Extract point cloud
            pointsensor = nusc.get("sample_data", data["LIDAR_TOP"])
            pcl_path = pointsensor["filename"]
            pc2scene[pcl_path] = name
    with open(savepath, 'wb') as f:
        pickle.dump(pc2scene, f)
    return pc2scene


def main(args, pc2scene, infos, features_type, model_spec, min_dist: float = 1.0, projections_only=False):
    max_volume_space = [51.2, 51.2, 3]
    min_volume_space = [-51.2, -51.2, -5]
    tpv_h_ = 100
    tpv_w_ = 100
    tpv_z_ = 8
    scale_h = 1
    scale_w = 1
    scale_z = 1
    grid_size = [tpv_h_ * scale_h, tpv_w_ * scale_w, tpv_z_ * scale_z]
    max_bound = np.asarray(max_volume_space)  # 51.2 51.2 3
    min_bound = np.asarray(min_volume_space)  # -51.2 -51.2 -5
    # get grid index
    crop_range = max_bound - min_bound
    cur_grid_size = np.asarray(grid_size)  # 200, 200, 16
    # TODO: intervals should not minus one.
    intervals = crop_range / (cur_grid_size - 1)

    for info in tqdm(infos):
        try:
            lidar_path = info['lidar_path']
            points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
            points = points[:, :3]  # take just X,Y,Z

            pc_original = LidarPointCloud.from_file(lidar_path)

            lidar_path_key = lidar_path.replace('./data/nuscenes/', '')
            scene = pc2scene[lidar_path_key]

            # --- Map points to images

            # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            pc_original = copy.deepcopy(pc_original)
            # cs_record = info["point"]["cs_record"]
            pc_original.rotate(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
            pc_original.translate(np.array(info["lidar2ego_translation"]))
            # Second step: transform from ego to the global frame.
            # poserecord = info["point"]["poserecord"]
            pc_original.rotate(Quaternion(info["ego2global_rotation"]).rotation_matrix)
            pc_original.translate(np.array(info["ego2global_translation"]))

            # Load each image and project
            camera_list = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
                           "CAM_FRONT_LEFT"]
            for idx_camera, camera_name in enumerate(camera_list):
                pc = copy.deepcopy(pc_original)
                im_path = info['cams'][camera_name]["data_path"]
                im = np.array(Image.open(im_path))
                imh, imw = im.shape[:2]

                im_name = os.path.split(im_path)[-1].split('.')[0]

                fts_name = f"{im_name}__{features_type}.pth"
                save_path = os.path.join(args.nusc_root_fts, model_spec, 'matched', camera_name, fts_name)
                projections_dir = os.path.join(args.nusc_root_fts, 'projections', camera_name)
                points_pth = os.path.join(projections_dir, f"{im_name}__points.npy")
                pixels_pth = os.path.join(projections_dir, f"{im_name}__pixels.npy")
                pixels_pth_vox_unq = os.path.join(projections_dir, f"{im_name}__pixels_vox_unq.npy")

                if not DEBUG and all([os.path.exists(pth) for pth in [save_path, points_pth, pixels_pth]]):
                    continue

                # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
                camera_data = info['cams'][camera_name]
                pc.translate(-np.array(camera_data["ego2global_translation"]))
                pc.rotate(Quaternion(camera_data["ego2global_rotation"]).rotation_matrix.T)

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
                mask = np.logical_and(mask, depths > min_dist)
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

                # TODO: Get the matching pixels, but only unique for voxels
                #   1) voxelize matching_points
                matching_points_xyz = pc.points[:3, matching_points].T
                voxel_sizes = (max_bound - min_bound) / (np.array(grid_size))

                # voxelize and take unique voxels
                # ind... mapping voxel_idx -> point_idx
                # inv_ind... mapping point_idx -> voxel_idx
                unq_xyz, ind, pt2vox = np.unique(
                    (matching_points_xyz / voxel_sizes).astype("int"), return_index=True, axis=0, return_inverse=True
                )
                unq_pix = []
                for idx in range(len(ind)):
                    unq_pix.append(np.mean(matching_pixels[pt2vox == idx], axis=0).astype(int))
                unq_pix = np.asarray(unq_pix)
                np.save(pixels_pth_vox_unq, unq_pix)
                # print(f'Saved to {pixels_pth_vox_unq}')
                #   3) get the corresponding matching_pixels
                # unq_pix = matching_pixels[ind]

                if not os.path.exists(projections_dir):
                    os.makedirs(projections_dir)
                np.save(points_pth, matching_points)
                np.save(pixels_pth, matching_pixels)
                if projections_only:
                    continue

                matching_pixels = np.concatenate(
                    (
                        np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * idx_camera,
                        matching_pixels
                    ),
                    axis=1
                )

                if os.path.exists(save_path):
                    continue

                # TODO: get the corresponding DINO feature map for this image
                fts_path = os.path.join(args.nusc_root_fts, model_spec, scene, fts_name)
                dino_feats = torch.load(fts_path, map_location='cpu')
                # TODO: get the correct features by interpolation
                # 1) interpolate DINO feature map into the size of the image
                dino_feats = torch.nn.functional.interpolate(dino_feats, (imh, imw))
                # 2) get the projections
                bi, rows, cols = matching_pixels.T
                dino_feats_paired = dino_feats[0, :, rows, cols].T
                # 3) save
                _d = os.path.split(save_path)[0]
                if not os.path.exists(_d):
                    os.makedirs(_d)
                torch.save(dino_feats_paired, save_path)
                del dino_feats, dino_feats_paired
        except:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--projections-only', action='store_true', help='Save only projections, not features.')
    parser.add_argument('--base-infos-dir', type=str,
                        default='/home/vobecant/PhD/TPVFormer-OpenSet/data')  # /scratch/project/open-26-3/vobecant/projects/TPVFormer-OpenSet/data
    parser.add_argument('--features-type', type=str, default=FEATURES_TYPE)
    parser.add_argument('--model-spec', type=str, default=MODEL_SPEC)
    parser.add_argument('--version', type=str, default=VERSION)
    parser.add_argument('--nusc-root', type=str,
                        default="/nfs/datasets/nuscenes")  # /scratch/project/open-26-3/vobecant/datasets/nuscenes
    parser.add_argument('--nusc-root-fts', type=str,
                        default="/nfs/datasets/nuscenes/features")  # /mnt/proj1/open-26-3/datasets/nuscenes/features
    parser.add_argument('--pc2scene-path', type=str,
                        default='/nfs/datasets/nuscenes/pc2scene.pkl')  # /scratch/project/open-26-3/vobecant/datasets/nuscenes/pc2scene.pkl
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    base_infos_dir = args.base_infos_dir
    info_files_trainval = ['nuscenes_infos_train.pkl', 'nuscenes_infos_val.pkl']
    info_files_mini = ['nuscenes_infos_train_mini.pkl', 'nuscenes_infos_val_mini.pkl']

    info_files = info_files_mini if args.version == 'mini' else info_files_trainval
    infos = []
    for info_file in info_files:
        info_file = os.path.join(base_infos_dir, info_file)
        with open(info_file, 'rb') as f:
            infos_cur = pickle.load(f)['infos']
        infos.extend(infos_cur)

    start = args.start
    end = args.end
    if start is not None and end is not None:
        print(f'Limiting infos to {start}-{end} out of {len(infos)}.')
        infos = infos[start:end]
        print(f'Use {len(infos)} info files.')

    pc2scene = prepare_pc2scene(args.pc2scene_path, args)

    if False:
        camera_list = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
                       "CAM_FRONT_LEFT"]
        image_path = '/nfs/datasets/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg'
        infos_new = []
        for info in tqdm(infos):
            for idx_camera, camera_name in enumerate(camera_list):
                im_path = info['cams'][camera_name]["data_path"]
                im_path = im_path.replace('./data/nuscenes/', '/nfs/datasets/nuscenes/')
                if im_path == image_path:
                    infos_new.append(info)
                    break
            if len(infos_new) > 0:
                break
        infos = infos_new

    main(args, pc2scene, infos, features_type=args.features_type, model_spec=args.model_spec,
         projections_only=args.projections_only)
