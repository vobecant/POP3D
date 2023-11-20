import os
import pickle

import numpy as np
import torch
import yaml
from mmcv.image.io import imread
from torch.utils import data


class ImagePoint_NuScenes(data.Dataset):
    def __init__(self, data_path, imageset='train', label_mapping="nuscenes.yaml",
                 label_mapping_gt="./config/label_mapping/nuscenes-noIgnore.yaml",
                 nusc=None, class_agnostic=False, retrieval=False, **kwargs):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        with open(label_mapping_gt, 'r') as stream:
            nuscenesyaml_gt = yaml.safe_load(stream)
        self.learning_map_gt = nuscenesyaml_gt['learning_map']

        if isinstance(data, dict):
            self.nusc_infos = data['infos']
        else:
            self.nusc_infos = data
        self.data_path = data_path
        self.nusc = nusc
        self.class_agnostic = class_agnostic
        self.retrieval = retrieval
        print(f'self.class_agnostic: {self.class_agnostic}')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def merge(self, dataset):
        num_new = len(dataset.nusc_infos)
        self.nusc_infos += dataset.nusc_infos
        print(f'Added {num_new} new infos. Resulting dataset has {len(self.nusc_infos)} infos.')

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        imgs_info = self.get_data_info(info)
        img_metas = {
            'lidar2img': imgs_info['lidar2img'], 'token': info['token']
        }
        # read 6 cams
        imgs = []
        for filename in imgs_info['img_filename']:
            imgs.append(
                imread(filename, 'unchanged').astype(np.float32)
            )
            # print(filename)

        lidar_path = info['lidar_path']
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
        num_pts = points.shape[0]

        print(f'self.retrieval: {self.retrieval}')
        if not self.retrieval and 'lidar_path_labels' in info.keys():
            lidarseg_labels_filename = info['lidar_path_labels']
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        elif self.nusc is not None:
            if not self.retrieval:
                lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                lidarseg_labels_filename = os.path.join(self.data_path,
                                                        self.nusc.get('lidarseg', lidar_sd_token)['filename'])
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            else:
                points_label = points_label_cls = info['retrieval']
                print(f'Loading retrieval info! Labels shape: {points_label.shape}, sum: {sum(points_label)}')

        else:
            # probably a test example, have empty segmentation labels
            points_label = points_label_cls = np.zeros((num_pts, 1), dtype=np.uint8)

        if not self.retrieval:
            points_label_cls = np.vectorize(self.learning_map_gt.__getitem__)(np.copy(points_label))
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

        if self.class_agnostic:
            points_label[:] = 1

        data_tuple = (imgs, img_metas, points[:, :3], points_label.astype(np.uint8), points_label_cls.astype(np.uint8))
        return data_tuple

    def get_data_info(self, info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
        """
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        return input_dict


class ImagePoint_NuScenes_withFeatures(ImagePoint_NuScenes):
    def __init__(self, data_path, features_path, features_type, projections_path, imageset='train',
                 label_mapping="nuscenes.yaml", nusc=None, class_agnostic=False, linear_probe=False,
                 label_mapping_gt="./config/label_mapping/nuscenes-noIgnore.yaml",
                 dino_features=False,
                 features_path_dino=None, features_type_dino=None, projections_path_dino=None,
                 retrieval=False, **kwargs):
        super(ImagePoint_NuScenes_withFeatures, self).__init__(data_path, imageset=imageset,
                                                               label_mapping=label_mapping, nusc=nusc,
                                                               label_mapping_gt=label_mapping_gt, retrieval=retrieval)
        self.class_agnostic = class_agnostic
        self.projections_path = projections_path
        self.features_path = features_path
        self.features_type = features_type
        self.dino_features = dino_features
        self.projections_path_dino = projections_path_dino
        self.features_path_dino = features_path_dino
        self.features_type_dino = features_type_dino
        self.linear_probe = linear_probe
        assert self.features_type in ['fts', 'q', 'k', 'v', 'clip']
        self.clip_features = self.features_type == 'clip'
        self.retrieval = retrieval
        self.first_iter = True
        print(f'self.class_agnostic: {self.class_agnostic}')

        self.missing_paths_file = '/home/vobecant/PhD/MaskCLIP/paths_file.txt'
        if os.path.exists(self.missing_paths_file):
            os.remove(self.missing_paths_file)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        # sample_idx = info['token'],
        # pts_filename = info['lidar_path']
        # print(f'sample idx: {sample_idx}, pts_filename: {pts_filename}')
        imgs_info = self.get_data_info(info)
        img_metas = {
            'lidar2img': imgs_info['lidar2img'], 'token': info['token']
        }

        # read 6 cams
        imgs = []
        feats = []
        if self.dino_features: feats_dino = []
        matching_points = []
        nonexistent = []
        pth_npy_path = None
        for filename in imgs_info['img_filename']:
            # print(filename)
            if not self.linear_probe:
                imgs.append(
                    imread(filename, 'unchanged').astype(np.float32)
                )
            im_name = os.path.split(filename)[-1].split('.')[0] 

            camera_name = im_name.split('__')[1]
           
            # features
            if self.clip_features:
                fts_name = im_name
                fts_path = os.path.join(self.features_path, camera_name, fts_name)
                if not os.path.exists(fts_path):
                    fts_name = f"{im_name}.pth.npy"
                pth_npy_path = os.path.join(self.features_path, camera_name, fts_name)
                fts_path = os.path.join(self.features_path, camera_name, fts_name)
                if not os.path.exists(fts_path):
                    fts_name = f"{im_name}.pth"
                fts_path = os.path.join(self.features_path, camera_name, fts_name)
                if not os.path.exists(fts_path):
                    fts_name = f"{im_name}_fts.pth"
                fts_path = os.path.join(self.features_path, camera_name, fts_name)
            else:
                fts_name = f"{im_name}__{self.features_type}.pth"
                fts_path = os.path.join(self.features_path, camera_name, fts_name)

            if self.dino_features:
                fts_name_dino = f"{im_name}__{self.features_type_dino}.pth"
                fts_path_dino = os.path.join(self.features_path_dino, camera_name, fts_name_dino)

            if self.first_iter:
                print(f'fts_path: {fts_path}')
                self.first_iter = False


            # corresponding points
            matching_points_cam_path = os.path.join(self.projections_path, camera_name, f"{im_name}__points.npy")
            try:
                matching_points_cam = np.load(matching_points_cam_path)
            except:
                raise Exception(f'Counldnt load {matching_points_cam_path}!')
            matching_points.append(matching_points_cam)

            if not os.path.exists(fts_path):
                nonexistent.append(fts_path)
                fts_cam = None
                continue
            else:
                try:
                    fts_cam = torch.load(fts_path, map_location='cpu')
                    if 'odise' in fts_path:
                        fts_cam = fts_cam.T
                except:
                    fts_cam = np.load(fts_path)

            if self.dino_features:
                try:
                    fts_cam_dino = torch.load(fts_path_dino, map_location='cpu')
                except:
                    fts_cam_dino = np.load(fts_path_dino)
                feats_dino.append(fts_cam_dino)

            if 'ovseg' in fts_path:
                fts_cam = fts_cam.T
            if fts_cam is not None:
                feats.append(fts_cam)

        # Load LiDAR points.
        lidar_path = info['lidar_path']
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])

        try:
            lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(self.data_path,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        except:
            if 'lidar_path_labels' in info:
                lidarseg_labels_filename = info['lidar_path_labels']
            else:
                lidarseg_labels_filename = None
        if not self.retrieval:
            if lidarseg_labels_filename is not None:
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            else:
                points_label = np.zeros((points.shape[0],1), dtype=np.uint8).reshape([-1, 1])
            points_label_cls = np.vectorize(self.learning_map_gt.__getitem__)(np.copy(points_label))
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        else:
            if 'retrieval' in info.keys():
                points_label = info['retrieval']
            else:
                if lidarseg_labels_filename is not None:
                    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                else:
                    points_label = np.zeros((points.shape[0],1), dtype=np.uint8).reshape([-1, 1])
            if lidarseg_labels_filename is not None:
                points_label_sem = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            else:
                points_label_sem = np.zeros((points.shape[0],1), dtype=np.uint8).reshape([-1, 1])
            points_label_cls = np.vectorize(self.learning_map_gt.__getitem__)(np.copy(points_label_sem))

        # print(points_label.shape, points_label_cls.shape)

        if self.class_agnostic:
            points_label[:] = 1

        try:
            feats = np.concatenate(feats)
        except:
            print(f'\nCouldnt load features for {info["token"]}, e.g., {pth_npy_path}')
            with open(self.missing_paths_file,'a') as f:
                for filename in imgs_info['img_filename']:
                    im_name = os.path.split(filename)[-1].split('.')[0]
                    tmp=im_name+'.jpg'
                    print(tmp)
                    f.write(tmp+'\n')
                # assert False
            print()
        try:
            matching_points = np.concatenate(matching_points)
        except:
            matching_points = []
            assert False

        if points.shape[0]!=points_label.shape[0]:
            print('number of points and targets is not the same!')

        data_tuple = (
            imgs, img_metas, points[:, :3], points_label.astype(np.uint8), points_label_cls.astype(np.uint8), feats,
            matching_points, nonexistent)

        if self.dino_features:
            feats_dino = np.concatenate(feats_dino)
            data_tuple += (feats_dino,)

        return data_tuple


class ImagePoint_NuScenes_withFeatures_openseg(ImagePoint_NuScenes):
    def __init__(self, data_path, features_path, features_type, projections_path, imageset='train',
                 label_mapping="nuscenes.yaml", nusc=None, class_agnostic=False, linear_probe=False,
                 label_mapping_gt="./config/label_mapping/nuscenes-noIgnore.yaml",
                 dino_features=False, retrieval=False, cam2tok_path=None, split=None, labels_path=None, **kwargs):
        super(ImagePoint_NuScenes_withFeatures_openseg, self).__init__(data_path, imageset=imageset,
                                                                       label_mapping=label_mapping, nusc=nusc,
                                                                       label_mapping_gt=label_mapping_gt,
                                                                       retrieval=retrieval)
        self.class_agnostic = class_agnostic
        self.projections_path = projections_path
        self.features_path = features_path
        self.features_type = features_type
        self.dino_features = dino_features
        self.linear_probe = linear_probe
        assert self.features_type in ['fts', 'q', 'k', 'v', 'clip']
        self.clip_features = self.features_type == 'clip'
        self.retrieval = retrieval
        self.first_iter = True
        self.cam2tok = None
        if cam2tok_path is not None:
            with open(cam2tok_path, 'rb') as f:
                self.cam2tok = pickle.load(f)
        print(f'self.class_agnostic: {self.class_agnostic}')
        self.split = split
        self.labels_path = labels_path

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        imgs_info = self.get_data_info(info)
        img_metas = {
            'lidar2img': imgs_info['lidar2img'], 'token': info['token']
        }
        token = info['token']

        # from tqdm import tqdm
        # for index in tqdm(range(6018)):
        #     info = self.nusc_infos[index]
        #     token = info['token']
        #     path = os.path.join(self.features_path, f'{token}.pt')
        #     try:
        #         torch.load(path)
        #     except:
        #         print(path)
        # exit(0)

        # read 6 cams
        imgs = []
        feats, points_loc = torch.load(os.path.join(self.features_path, f'{token}.pt'))
        matching_points = torch.arange(max(feats.shape))
        nonexistent = []
        for filename in imgs_info['img_filename']:
            # print(filename)
            if not self.linear_probe:
                imgs.append(
                    imread(filename, 'unchanged').astype(np.float32)
                )

        try:
            # lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
            # lidarseg_labels_filename = os.path.join(self.data_path,
            #                                         self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            lidarseg_labels_filename = os.path.join(self.labels_path, f'{token}.pt')
        except:
            lidarseg_labels_filename = None
        if not self.retrieval:
            if lidarseg_labels_filename is not None:
                points_label = torch.load(lidarseg_labels_filename).reshape([-1, 1])
            else:
                points_label = np.zeros((points.shape[0],1), dtype=np.uint8).reshape([-1, 1])
            points_label_cls = points_label.copy() + 1
            # points_label_cls = np.vectorize(self.learning_map_gt.__getitem__)(np.copy(points_label))
            # points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        else:
            if 'retrieval' in info.keys():
                points_label = info['retrieval']
            else:
                if lidarseg_labels_filename is not None:
                    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                else:
                    points_label = np.zeros((points.shape[0],1), dtype=np.uint8).reshape([-1, 1])
            if lidarseg_labels_filename is not None:
                points_label_sem = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            else:
                points_label_sem = np.zeros((points.shape[0],1), dtype=np.uint8).reshape([-1, 1])
            points_label_cls = np.vectorize(self.learning_map_gt.__getitem__)(np.copy(points_label_sem))

        if self.class_agnostic:
            points_label[:] = 1

        data_tuple = (
            imgs, img_metas, points_loc, points_label.astype(np.uint8), points_label_cls.astype(np.uint8), feats,
            matching_points, nonexistent)


        return data_tuple


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config',
                        default='config/tpv04_occupancy_100_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc.py')
    parser.add_argument('--resume-from', type=str,
                        default='/home/vobecant/TPVFormer-OpenSet/trained_models/RN101_100_maskclip_8gpu_6ep_fullRes_2occ2ft_2decOcc_512hidOcc_2decFt_1024hidFt_noClsW_16052023_090608/epoch_12.pt')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--text-embeddings-path', type=str, default=None)
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--set-name', default=None, type=str)
    parser.add_argument('--num-classes', default=None, type=int)
    parser.add_argument('--scale', default=None, type=int)
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--normalized-cosine', action='store_true')
    parser.add_argument('--nusc-dir', type=str, default='/nfs/datasets/nuscenes')
    args = parser.parse_args()

    