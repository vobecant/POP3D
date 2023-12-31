import os
import copy
import argparse
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm

CAMERA_LIST = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        nusc_root,
        version="test",
        shuffle=False,
        save_dir=None,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle

        self.H, self.W = None, None

        assert save_dir is not None
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            camera_list = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_FRONT_LEFT",
            ]
            for cam_name in camera_list:
                os.makedirs(os.path.join(self.save_dir, cam_name))


        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version=f"v1.0-{version}", dataroot=nusc_root, verbose=True
            )
        # print(f'NuScenes loaded with {len(self.nusc)} samples.')

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)

        pairing_points = np.empty(0, dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]


        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):

            
            cam = self.nusc.get("sample_data", data[camera_name])
            spec = os.path.split(cam['filename'])[-1].split('.')[0]
            matching_points_path = os.path.join(self.save_dir, camera_name, f'{spec}__points.npy')
            matching_pixels_path = os.path.join(self.save_dir, camera_name, f'{spec}__pixels.npy')
            if os.path.exists(matching_points_path) and os.path.exists(matching_pixels_path):
                print(f'{matching_points_path} exists for spec {spec}')
                continue


            pc = copy.deepcopy(pc_original)
            if self.H is None and self.W is None:
                im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
                self.H, self.W = im.shape[:2]

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < self.W - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < self.H - 1)
            matching_points = np.where(mask)[0]
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)
            pairing_points = np.concatenate((pairing_points, matching_points))
        
            np.save(matching_points_path, matching_points)
            np.save(matching_pixels_path, matching_pixels)



    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        self.map_pointcloud_to_image(self.list_keyframes[idx])
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nusc_root', type=str, default='./data/nuscenes')
    parser.add_argument('--proj-dir', type=str, default='./data/nuscenes/features/projections')

    args = parser.parse_args()

    nusc_root = args.nusc_root
    projections_root = args.proj_dir #os.path.join(nusc_root, 'features', 'projections')
    if not os.path.exists(projections_root):
        os.makedirs(projections_root)
    
    for cam_name in CAMERA_LIST:
        cam_dir = os.path.join(projections_root, cam_name)
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)

    for phase in ['train','val','test']:
        print(f'\n\nGenerating projection files for "{phase}" split.')
        dataset = NuScenesMatchDataset(phase=phase, nusc_root=nusc_root, save_dir=projections_root, version='test' if phase=='test' else 'trainval')
        for _ in tqdm(dataset):
            pass