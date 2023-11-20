import argparse
import os
import pickle

from nuscenes import NuScenes
from tqdm import tqdm


def main(args):
    nusc = NuScenes(version=args.version,
                    dataroot=args.nusc_root,
                    verbose=True)

def preprocess(args):
    version = args.version
    if 'trainval' in version:
        imageset_train = "./data/nuscenes_infos_train.pkl"
        imageset_val = "./data/nuscenes_infos_val.pkl"
        extra = ''
    else:
        imageset_train = "./data/nuscenes_infos_train_mini.pkl"
        imageset_val = "./data/nuscenes_infos_val_mini.pkl"
        extra = '_mini'
    data_path = args.nusc_root

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--version', default='v1.0-mini', type=str, help='version of the nuScenes dataset')
    parser.add_argument('--nusc_root', default='/path/to/nuscenes', type=str,
                        help='path to the directory with the dataset')
    args = parser.parse_args()

    preprocess(args)

    main(args)