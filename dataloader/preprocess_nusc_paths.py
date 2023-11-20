import os
import pickle

from nuscenes import NuScenes
from tqdm import tqdm


def preprocess_test():
    version = 'v1.0-test'
    imageset_test_path = "./data/nuscenes_infos_test.pkl"
    data_path = "./data/nuscenes/"

    imageset_test = []

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    lidar2ego_translation = [0.985793, 0.0, 1.84019]
    lidar2ego_rotation = [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719]

    with open(os.path.join(data_path, 'cam_infos.pkl'), 'rb') as f:
        cam_infos = pickle.load(f)

    for my_sample in nusc.sample:
        lidar_path = os.path.join(data_path, nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])['filename'])
        token = my_sample['token']
        nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])

        cams = {}
        for cam_name in cam_names:
            data = nusc.get('sample_data', my_sample['data'][cam_name])
            cur_dict = {
                'data_path': os.path.join(data_path, data['filename']),
                'type': data['channel']
            }
            for k, v in cam_infos[cam_name].items():
                cur_dict[k] = v

            cams[cam_name] = cur_dict

        imageset_test.append(
            {
                'lidar_path': lidar_path, 'token': token, 'cams': cams, 'lidar2ego_translation': lidar2ego_translation,
                'lidar2ego_rotation': lidar2ego_rotation, 'sweeps': None, 'ego2global_translation': None,
                'ego2global_rotation': None,
            }
        )

    with open(imageset_test_path, 'wb') as f:
        pickle.dump(imageset_test, f)
    print(f'Saved to {imageset_test_path}')


if __name__ == '__main__':
    preprocess_test()
    version = 'v1.0-mini'
    if 'trainval' in version:
        imageset_train = "./data/nuscenes_infos_train.pkl"
        imageset_val = "./data/nuscenes_infos_val.pkl"
        extra = ''
    else:
        imageset_train = "./data/nuscenes_infos_train_mini.pkl"
        imageset_val = "./data/nuscenes_infos_val_mini.pkl"
        extra = '_mini'
    data_path = "data/nuscenes/"

    with open(imageset_train, 'rb') as f:
        train_infos = pickle.load(f)['infos']

    with open(imageset_val, 'rb') as f:
        val_infos = pickle.load(f)['infos']

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    for infos, name in zip([train_infos, val_infos], ['train', 'val']):
        infos_new = []
        for info in tqdm(infos):
            lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(data_path, nusc.get('lidarseg', lidar_sd_token)['filename'])
            info['lidar_path_labels'] = lidarseg_labels_filename
            infos_new.append(info)

        save_path = f"./data/nuscenes_infos_{name}{extra}_new.pkl"
        with open(save_path, 'wb')as f:
            pickle.dump(infos_new, f)
        print(f'Saved to {save_path}')
