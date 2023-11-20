import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from mmcv.image.io import imread


SPLITS = [
    'train', 'val', 'test', 'eval', 'all'
]
SPLITS_LUT = {
    'train':['train'], 'val':['val'], 'test':['test'], 'eval':['val','test'], 'all':['train','val','test']
}


def load_infos(infos_dir, splits, tokens):

    filtered_infos = {}

    for split in splits:

        # load camera2tkoken LUT for the current split
        cam2token_path = f"{infos_dir}/nuscenes_cam2token_{split}.pkl"
        with open(cam2token_path, 'rb') as f:
            cam2token_cur=pickle.load(f)
        
        # load info files for the current split
        with open(f'./data/nuscenes_infos_{split}_new.pkl', 'rb') as f:
            cur_infos = pickle.load(f)
            if isinstance(cur_infos, dict):
                cur_infos = cur_infos['infos']
        
        for info in cur_infos:
            token = info['token']
            if not token in tokens:
                continue
            filtered_infos[token] = info
        
    assert all([token in filtered_infos.keys() for token in tokens])

    return filtered_infos



class RetrievalDataset(Dataset):

    def __init__(self, retrieval_dir, infos_dir, nusc_dir, split):
        self.retrieval_dir = retrieval_dir
        self.infos_dir = infos_dir
        self.nusc_dir = nusc_dir
        self.split = split.lower()
        
        self.ann_dir = os.path.join(self.retrieval_dir, 'annotations')
        self.matching_points_dir = os.path.join(self.retrieval_dir, 'matching_points')
        
        # Load retrieval data.
        self.key2idx = {}
        self.idx2key = []
        self.idx2token = []
        self.samples = {}
        self.csv_path = os.path.join(self.retrieval_dir, f'retrieval_anns_{self.split}.csv')
        for idx,line in enumerate(open(self.csv_path,'r').readlines()):
            line = line.strip()
            token, split, ann_path, pts_path, query = line.split(';')
            key = (token, query)
            self.samples[key] = dict(token=token, split=split, ann_path=os.path.join(self.retrieval_dir, ann_path), visible_pts_path=os.path.join(self.retrieval_dir, pts_path), query=query)
            self.key2idx[key] = idx
            self.idx2key.append(key)
            self.idx2token.append(token)

        # Load info files.
        self.token2info = load_infos(self.infos_dir, SPLITS_LUT[self.split], self.idx2token)



    def __len__(self):
        return len(self.samples)


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

    def __getitem__(self, idx):
        key = self.idx2key[idx]
        sample = self.samples[key]
        token, query = key
        info = self.token2info[token]

        # get images info
        imgs_info = self.get_data_info(info)

        # read 6 cams
        imgs = []
        for filename in imgs_info['img_filename']:
            imgs.append(
                imread(filename, 'unchanged').astype(np.float32)
            )

        # read LiDAR points
        lidar_path = info['lidar_path']
        xyz_points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:,:3]

        ann_all = np.load(sample['ann_path'])
        visible_pts = np.load(sample['visible_pts_path'])

        ann_visible = np.zeros(len(visible_pts),dtype=ann_all.dtype)
        ann_visible = ann_all[visible_pts]

        sample.update(dict(ann_visible=ann_visible, ann_all=ann_all, visible_pts=visible_pts, imgs=imgs, xyz_points=xyz_points))

        return sample

    def get(self, token):
        sample = self.samples[token]

        ann = np.load(sample['ann_path'])
        visible_pts = np.load(sample['visible_pts_path'])

        sample.update(dict(ann=ann, visible_pts=visible_pts))

        return sample


if __name__ == '__main__':
    retrieval_dir = '/nfs/datasets/nuscenes/retrieval_benchmark_release'
    infos_dir = '/home/vobecant/PhD/TPVFormer-OpenSet/data'
    nusc_dir = '/nfs/datasets/nuscenes'
    split = 'test'
    dataset = RetrievalDataset(retrieval_dir=retrieval_dir, split=split, infos_dir=infos_dir, nusc_dir=nusc_dir)

    for sample in dataset:
        print(sample.keys())