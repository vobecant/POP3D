import argparse
import os.path
import pickle

import numpy as np
import torch
# from matplotlib import pyplot as plt
from mmcv import Config
from tqdm import tqdm

from dataloader.dataset import get_nuScenes_label_name
from eval_maskclip import assign_clip_labels, NCLS2CONFIG, EMPTY_SEMANTIC_LABEL
from retrieval_visualization import Vox2Real
from utils.load_save_util import revise_ckpt_linear_probe
from visualization.vis_frame import pca_rgb_projection

EXCAVATOR = [
    'n015-2018-09-25-11-10-38+0800__CAM_FRONT_RIGHT__1537845410620339.jpg'
    'n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151864528113.jpg',
    'n015-2018-09-25-11-10-38+0800__CAM_FRONT_RIGHT__1537845398670339.jpg',
]
TRASH_BIN = [
    'n008-2018-09-18-14-35-12-0400__CAM_FRONT_LEFT__1537295886604799.jpg',
    'n008-2018-07-27-12-07-38-0400__CAM_FRONT_RIGHT__1532708693170482.jpg',
]
HORSE = [
    'n008-2018-08-31-11-37-23-0400__CAM_BACK_LEFT__1535730425547405.jpg',
    'n008-2018-08-31-11-37-23-0400__CAM_BACK_LEFT__1535730426547405.jpg',
    'n008-2018-08-31-11-37-23-0400__CAM_FRONT_LEFT__1535730422504799.jpg'
]

SAMPLE = [
    'n015-2018-09-25-13-17-43+0800__CAM_BACK_LEFT__1537852964197423.jpg'
]

SET2IMAGES = {'trash_bin': TRASH_BIN, 'horse': HORSE, 'sample': SAMPLE}

ID2COLOR = np.array(
    [
        [0, 0, 0, 255],  # ignore
        [255, 120, 50, 255],  # barrier              orange
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [255, 127, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],  # ego car
        [0, 191, 255, 255]  # ego car
    ]
).astype(np.uint8)


def load_network(cfg):
    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model)
    my_model = my_model.cuda()

    print('resume from: ', cfg.resume_from)
    map_location = 'cpu'
    ckpt = torch.load(cfg.resume_from, map_location=map_location)
    revise_fnc = revise_ckpt_linear_probe
    print(my_model.load_state_dict(revise_fnc(ckpt['state_dict'], ddp=False), strict=True))

    epoch = ckpt['epoch']
    print(f'successfully resumed from epoch {epoch}')

    my_model.eval()

    return my_model


def get_dataloader(cfg):
    dataset_config = cfg.dataset_params
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader
    grid_size = cfg.grid_size

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=False,
            scale_rate=cfg.get('scale_rate', 1),
            num_workers=0,
            dataset_type='ImagePoint_NuScenes'
        )

    return train_dataset_loader, val_dataset_loader


def get_img2scene_lut():
    infos = []
    for split in ['train', 'val']:
        with open(f'./data/nuscenes_infos_{split}_new.pkl', 'rb') as f:
            infos.extend(pickle.load(f))

    # test split
    with open('./data/nuscenes_infos_test.pkl', 'rb') as f:
        infos.extend(pickle.load(f))

    lut = {}
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    for idx, info in enumerate(infos):
        for cam_name in cam_names:
            img_path = info['cams'][cam_name]['data_path']
            img_name = os.path.split(img_path)[-1].split('.')[0]
            lut[img_name] = idx

    return infos, lut


@torch.no_grad()
def get_features(dataloader, index, model, save_dir, class_mapping_clip, text_features, colors, vox2real=None,
                 vox2real_gt=None):
    for i, loaded_data in enumerate(tqdm(dataloader)):
        if index is not None and i != index: continue
        imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs_agnostic, val_vox_label_cls, val_vox_label_cls_val, *_ = loaded_data

        token = dataloader.dataset.imagepoint_dataset.nusc_infos[i]['token']
        cur_image_paths = [v['data_path'] for v in dataloader.dataset.imagepoint_dataset.nusc_infos[i]['cams'].values()]

        save_dir_cur = os.path.join(save_dir, token)
        if not os.path.exists(save_dir_cur):
            os.makedirs(save_dir_cur)

        imgs = imgs.cuda()
        val_grid_float = val_grid.to(torch.float32).cuda()
        dummy_fts_loc = torch.zeros((1, 1, 3), dtype=torch.float32).cuda()

        predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
        predict_fts_vox_dino, predict_fts_pts_dino = model(img=imgs, img_metas=img_metas,
                                                           points=val_grid_float,
                                                           features=dummy_fts_loc)

        occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_occupancy.argmax(1) == 1))

        # predict features at those positions
        tgt_shape = (
            predict_fts_vox.shape[0], predict_fts_vox.shape[2], predict_fts_vox.shape[3], predict_fts_vox.shape[4])
        outputs_vox_clip_all = torch.ones(tgt_shape, device='cuda').long() * EMPTY_SEMANTIC_LABEL

        n_occ = occupied_voxels_loc.shape[1]
        predicted_features_occupied_vox = None
        if n_occ > 0:
            predicted_features_occupied_vox = predict_fts_vox[occupied_voxels_loc[0], :,
                                              occupied_voxels_loc[1], occupied_voxels_loc[2],
                                              occupied_voxels_loc[3]].unsqueeze(0)

        # assign labels
        ignore_label = 0
        if args.set_name is not None:
            ignore_label = None
        _outputs_vox_clip_predOcc = assign_clip_labels(
            args, class_mapping_clip, None, None, predicted_features_occupied_vox,
            text_features, None, None, assignment_only=True, ignore_label=0)
        outputs_vox_clip_all[occupied_voxels_loc[0], occupied_voxels_loc[1], occupied_voxels_loc[2],
                             occupied_voxels_loc[3]] = _outputs_vox_clip_predOcc

        xyz_pred = torch.stack(torch.where(outputs_vox_clip_all.cpu() != EMPTY_SEMANTIC_LABEL))[1:]
        labels_pred = _outputs_vox_clip_predOcc

        xyz_pred = vox2real(xyz_pred)
        out_path = os.path.join(save_dir_cur, f'pca.txt')
        rgb = pca_rgb_projection(predicted_features_occupied_vox[0])[0]
        res2txt(xyz_pred, rgb, colors, out_path=out_path, verbose=False)
        print(f'Saved to {out_path}')

        tgt_dir = save_dir_cur
        if tgt_dir is not None:
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            out_path = os.path.join(tgt_dir, f'{token}.txt')
            xyz_pred = vox2real(xyz_pred)
            res2txt(xyz_pred, labels_pred, colors, out_path=out_path, verbose=False)

            for path in cur_image_paths:
                im_fname = os.path.split(path)[-1]
                tgt = os.path.join(tgt_dir, im_fname)
                try:
                    os.symlink(path, tgt)
                except:
                    pass

            # save GT
            occupied_voxels_loc_gt = torch.stack(torch.where(val_vox_label_cls[0] != 17))
            idx_x, idx_y, idx_z = occupied_voxels_loc_gt
            occupied_voxels_label_gt = val_vox_label_cls[0][idx_x, idx_y, idx_z].squeeze()
            out_path = os.path.join(tgt_dir, f'{token}_gt.txt')
            if vox2real_gt is not None:
                occupied_voxels_loc_gt = vox2real_gt(occupied_voxels_loc_gt)
            res2txt(occupied_voxels_loc_gt, occupied_voxels_label_gt, ID2COLOR, out_path, verbose=False)

        torch.cuda.empty_cache()
        del predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
            predict_fts_vox_dino, predict_fts_pts_dino, occupied_voxels_loc, outputs_vox_clip_all, predicted_features_occupied_vox, xyz_pred, labels_pred


def get_clip_utils(args, dataloader, no_feats = False):
    if args.num_classes is None:
        unique_label_clip = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        SemKITTI_label_name_clip = get_nuScenes_label_name("./config/label_mapping/nuscenes.yaml")
        unique_label_str_clip = [SemKITTI_label_name_clip[x] for x in unique_label_clip]
    else:
        unique_label_clip = [i for i in range(1, args.num_classes + 1)]
        SemKITTI_label_name_clip = get_nuScenes_label_name(NCLS2CONFIG[args.num_classes])
        unique_label_str_clip = [SemKITTI_label_name_clip[x] for x in unique_label_clip]
    colors = [] if 'noise' in unique_label_str_clip else [ID2COLOR[0]]
    colors += [ID2COLOR[_id] for _id in unique_label_clip]

    text_features = None
    class_mapping_clip = None
    learning_map_gt = dataloader.dataset.imagepoint_dataset.learning_map_gt
    if args.text_embeddings_path is not None and os.path.exists(args.text_embeddings_path):
        text_features = torch.load(args.text_embeddings_path, map_location='cpu')
        if type(text_features) in [tuple, list]:
            text_features, class_mapping_clip = text_features
            class_mapping_clip = torch.tensor([learning_map_gt[c.item()] for c in class_mapping_clip]).cuda()
        embedding_dim = 512
        if text_features.shape[0] == embedding_dim:
            text_features = text_features.T
        text_features = text_features.float().cuda()
    elif not no_feats:
        raise Exception(f"Given path {args.text_embeddings_path} is either None or does not exist!")

    try:
        return class_mapping_clip, text_features, colors
    except:
        class_mapping_clip = torch.tensor([i for i in range(text_features.shape[0])]).cuda()
        return class_mapping_clip, text_features, colors


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config',
                        default='config/tpv04_occupancy_100_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc_newLoader.py')
    parser.add_argument('--resume-from', type=str,
                        default='out/RN101_100_maskclip_8gpu_6ep_fullRes_2occ2ft_2decOcc_512hidOcc_2decFt_1024hidFt_noClsW_15052023_013402/latest.pth')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--text-embeddings-path', type=str,
                        default='/home/vobecant/PhD/MaskCLIP/pretrain/nuscenes_subcategories_ViT16_clip_text.pth')
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--set-name', default=None, type=str)
    parser.add_argument('--num-classes', default=None, type=int)
    parser.add_argument('--scale', default=None, type=int)
    parser.add_argument('--skip', default=None, type=int)
    parser.add_argument('--mini', action='store_true')
    args = parser.parse_args()

    if args.mini:
        args.py_config = 'config/tpv04_occupancy_mini_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc.py'

    if args.scale is not None and args.save_dir is not None:
        args.save_dir += f'_{args.scale}xScale'

    print(f'Save to {args.save_dir}')

    return args


def res2txt(xyz, lbl, colors, out_path, verbose=True):
    if verbose: print(lbl.shape)
    lbl = lbl.squeeze()
    if isinstance(lbl, torch.Tensor):
        lbl = lbl.cpu()
    lbl = lbl.tolist()
    if xyz.shape[0] == 3:
        xyz = xyz.T
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu()
    xyz = xyz.tolist()

    with open(out_path, 'w') as f:
        for _xyz, _lbl in zip(xyz, lbl):
            if isinstance(_lbl, (tuple, list)):
                color = map(str, _lbl)
            else:
                color = map(str, colors[_lbl][:3])
            color = ' '.join(color)
            if verbose:
                print(_lbl)
                print(colors[_lbl])
                print(color)
            _str = f'{_xyz[0]} {_xyz[1]} {_xyz[2]} {color}\n'
            f.write(_str)

    if verbose:
        print(f'Saved to {out_path}')


def limit_dataloader(val_dataloader, image_names, img2scene_lut, new_infos):
    orig_infos = val_dataloader.dataset.imagepoint_dataset.nusc_infos
    val_dataloader.dataset.imagepoint_dataset.nusc_infos = [
        new_infos[img2scene_lut[img_name.split('.')[0]]] for img_name in image_names
    ]
    return val_dataloader


if __name__ == '__main__':
    args = get_args()
    cfg = Config.fromfile(args.py_config)
    cfg.resume_from = args.resume_from

    if args.scale is not None:
        cfg.model['tpv_aggregator']['scale_h'] = cfg.model['tpv_aggregator']['scale_w'] = cfg.model['tpv_aggregator'][
            'scale_z'] = args.scale

    new_infos, lut = get_img2scene_lut()
    model = load_network(cfg)
    _, val_dataloader = get_dataloader(cfg)

    grid_size = np.array([100, 100, 8])
    grid_size_scaled = grid_size * args.scale
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    vox2real = Vox2Real(grid_size_scaled, point_cloud_range)
    vox2real_gt = Vox2Real(grid_size, point_cloud_range)

    if args.set_name is not None:
        set_name = args.set_name.lower()
        tmp_path = f'/home/vobecant/PhD/MaskCLIP/pretrain/{set_name}_ViT16_clip_text.pth'
        if os.path.exists(tmp_path):
            args.text_embeddings_path = tmp_path
        image_names = SET2IMAGES[set_name]
        val_dataloader = limit_dataloader(val_dataloader, image_names, lut, new_infos)

    if args.skip is not None:
        val_dataloader.dataset.imagepoint_dataset.nusc_infos = val_dataloader.dataset.imagepoint_dataset.nusc_infos[
                                                               ::args.skip]
        print(f'Limited the dataset to {len(val_dataloader.dataset.imagepoint_dataset.nusc_infos)} samples.')

    idx = None
    class_mapping_clip, text_features, colors = get_clip_utils(args, val_dataloader)
    predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
    predict_fts_vox_dino, predict_fts_pts_dino, val_vox_label_cls_val, tgt_shape = get_features(val_dataloader, idx,
                                                                                                model, args.save_dir,
                                                                                                class_mapping_clip,
                                                                                                text_features, colors,
                                                                                                vox2real=vox2real,
                                                                                                vox2real_gt=vox2real_gt)
