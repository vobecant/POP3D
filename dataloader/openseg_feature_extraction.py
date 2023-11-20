"""
Greatly inspired from https://github.com/pengsongyou/openscene/blob/main/scripts/feature_fusion/nuscenes_openseg.py
"""

import argparse
import os
import pickle
from os.path import join

import numpy as np
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import torch
from tensorflow import io
from tqdm import tqdm


def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on nuScenes.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')
    parser.add_argument('--dataset-root', type=str, default=None, help='Path to dir with datasets.')
    parser.add_argument('--nusc-root-fts', type=str, default=None, help='Path to dir with nusc features.')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''

    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes


def extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=None, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    # load RGB image
    np_image_string = read_bytes(img_dir)
    # run OpenSeg
    results = openseg_model.signatures['serving_default'](
        inp_image_bytes=tf.convert_to_tensor(np_image_string),
        inp_text_emb=text_emb)
    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]
    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
    if img_size is not None:
        feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
            image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
    else:
        feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()

    feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)

    return feat_2d


def process_one_image(image_path, out_dir, args):
    '''Process one image.'''

    # short hand
    img_size = args.img_dim
    openseg_model = args.openseg_model
    text_emb = args.text_emb

    device = torch.device('cpu')

    d, im_name = os.path.split(image_path)
    im_name = im_name.split('.')[0]
    camera_name = d.split(os.path.sep)[-1]
    projections_dir = os.path.join(args.nusc_root_fts, 'projections', camera_name)
    pixels_pth = os.path.join(projections_dir, f"{im_name}__pixels.npy")
    matching_pixels = np.load(pixels_pth)

    feat_2d = extract_openseg_img_feature(
        image_path, openseg_model, text_emb, img_size=[img_size[1], img_size[0]]).to(device)

    # 2) get the projections
    rows, cols = matching_pixels.T
    feats2d_paired = feat_2d[:, rows, cols].T

    out_dir_cam = os.path.join(out_dir,camera_name)
    if not os.path.exists(out_dir_cam):
        os.makedirs(out_dir_cam)
    save_path = os.path.join(out_dir_cam, f'{im_name}.pth')
    torch.save(feats2d_paired, save_path)


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #### Dataset specific parameters #####
    # img_dim = (800, 450)
    img_dim = (1600, 900)
    ######################################

    args.cut_num_pixel_boundary = 5  # do not use the features on the image boundary
    args.feat_dim = 768  # CLIP feature dimension
    args.img_dim = img_dim

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if args.openseg_model != '':
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                                                  tags=[tf.saved_model.tag_constants.SERVING], )
        args.text_emb = tf.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None

    # get the paths from .pkl files
    pkl_path = f'./data/nuscenes_infos_{args.split}.pkl'
    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)['infos']
    image_paths = []
    for info in infos:
        for cam_name, cam_info in info['cams'].items():
            image_path = cam_info['data_path'].replace('./data', args.dataset_root)
            image_paths.append(image_path)
    print(f'Loaded {len(image_paths)} paths.')

    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]
        image_paths = image_paths[id_range[0]:id_range[1]]
        print(f'Process only images with IDs in [{id_range[0]}, {id_range[1]}), #images: {len(image_paths)}')

    for image_path in tqdm(image_paths):
        process_one_image(image_path, out_dir, args)


if __name__ == '__main__':
    args = get_args()
    main(args)
