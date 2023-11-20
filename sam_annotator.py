import copy
import os

import numpy as np
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm


class SAMPredictor(nn.Module):
    def __init__(self, sam=None, sam_model=None, sam_path=None, multimask_output=False, soft_masks=False,
                 mobile_sam=False):
        super(SAMPredictor, self).__init__()
        if sam is None:
            sam = sam_model_registry[sam_model](checkpoint=sam_path).cuda()
        else:
            self.predictor = SamPredictor(sam)
        self.multimask_output = multimask_output
        self.soft_masks = soft_masks

    def sam_single_image(self, image, points):
        self.predictor.set_image(image)
        pred_masks, scores = [], []
        points = points[:, None, [1, 0]]
        n_queries = points.shape[0]
        for point in points:
            masks_cur, confs, _ = self.predictor.predict(point_coords=point, point_labels=np.array([1]),
                                                         multimask_output=self.multimask_output,
                                                         return_logits=self.soft_masks
                                                         )

            pred_masks.extend(masks_cur)
            scores.extend(confs)

        return pred_masks, scores


def get_sam_model(args):
    from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
    sam_model = "vit_h"
    sam_path = "/home/vobecant/PhD/weights/sam_vit_h_4b8939.pth"
    multimask_output = args.multimask_output
    sam_predictor = SAMPredictor(sam_model=sam_model, sam_path=sam_path, multimask_output=multimask_output,
                                 soft_masks=args.soft_masks)
    sam = sam_model_registry[sam_model](checkpoint=sam_path).cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)

    return sam_predictor, mask_generator


def extract_sam_masks(args, image_paths, infos, lidar_paths):
    sam_predictor, sam_mask_generator = get_sam_model(args)
    print('TODO: Try other hyperparameters!!!! Maybe explore the LiDAR guidance!!!')
    os.makedirs(args.save_dir, exist_ok=True)
    print("Extracting SAM masks...")
    num_scenes = len(image_paths)
    print(f'Number of scenes: {num_scenes}, number of images: {num_scenes * 6}')
    min_dist = 1.
    for info in tqdm(infos):

        paths = []
        for cam_name, cam_info in info['cams'].items():
            imgfile = cam_info['data_path'].replace('./data', args.data_root)
            single_mask = '' if args.multimask_output else '_singleMask'
            path = os.path.join(
                args.save_dir,
                os.path.splitext(os.path.basename(imgfile))[0] + f"{single_mask}.pkl",
            )
            paths.append(path)
        if not args.debug and all([os.path.exists(p) for p in paths]):
            # pass
            continue

        lidar_path = info['lidar_path'].replace('./data', args.data_root)
        pc_original = LidarPointCloud.from_file(lidar_path)
        pc_original_npy = pc_original.points[:3].T

        pc_original.rotate(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
        pc_original.translate(np.array(info["lidar2ego_translation"]))
        pc_original.rotate(Quaternion(info["ego2global_rotation"]).rotation_matrix)
        pc_original.translate(np.array(info["ego2global_translation"]))

        for cam_name, cam_info in info['cams'].items():
            print(f'{cam_name}')
            pc = copy.deepcopy(pc_original)
            imgfile = cam_info['data_path'].replace('./data', args.data_root)
            # img = cv2.imread(imgfile)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(Image.open(imgfile))
            imh, imw = img.shape[:2]

            # load points
            img_name = os.path.split(imgfile)[-1].split('.')[0]
            points_path = os.path.join(args.projections_path, cam_name, f'{img_name}__pixels.npy')
            points = np.load(points_path)
            # print(f'{len(points)} point projections')

            pc.translate(-np.array(cam_info["ego2global_translation"]))
            pc.rotate(Quaternion(cam_info["ego2global_rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            pc.translate(-np.array(cam_info["sensor2ego_translation"]))
            pc.rotate(Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cam_info["cam_intrinsic"]),
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

            matching_pixels_y, matching_pixels_x = matching_pixels.T
            matching_points_xyz = pc.points[:3, matching_points].T

            # PREDICT
            picked_points = ...
            masks, confs = sam_predictor.sam_single_image(img, picked_points)

            single_mask = '' if args.multimask_output else '_singleMask'
            _savefile = os.path.join(
                args.save_dir,
                os.path.splitext(os.path.basename(imgfile))[0] + f"{single_mask}.pkl",
            )

    return image_paths
