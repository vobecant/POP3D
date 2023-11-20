import argparse
import os.path
import pickle

import numpy as np
import torch
# from matplotlib import pyplot as plt
try:
    from mmcv import Config
except:
    from mmengine.config import Config
from tqdm import tqdm

from utils.load_save_util import revise_ckpt_linear_probe

EXCAVATOR = [
    'n015-2018-09-25-11-10-38+0800__CAM_FRONT_RIGHT__1537845410620339',
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

JEEP = [
    'n008-2018-07-27-12-07-38-0400__CAM_FRONT_RIGHT__1532708204920482.jpg',
    'n008-2018-08-28-16-16-48-0400__CAM_BACK_RIGHT__1535487738678113.jpg',
    'n008-2018-09-18-14-43-59-0400__CAM_FRONT_LEFT__1537296342904799.jpg',
    'n008-2018-09-18-14-43-59-0400__CAM_FRONT_RIGHT__1537296701620482.jpg'
]
STAIRS = [
    'n015-2018-10-08-15-44-23+0800__CAM_FRONT_LEFT__1538984983404844.jpg',
    'n015-2018-08-01-16-32-59+0800__CAM_FRONT_LEFT__1533112843155881.jpg',
    'n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984454504844.jpg',
    'n015-2018-10-02-11-11-43+0800__CAM_FRONT_LEFT__1538450232154844.jpg'
]
POLICE = [
    'n008-2018-08-31-12-15-24-0400__CAM_BACK_RIGHT__1535732236378113.jpg',
    'n008-2018-08-31-12-15-24-0400__CAM_FRONT_RIGHT__1535732234870482.jpg',
    'n015-2018-09-26-11-17-24+0800__CAM_FRONT__1537932241262460.jpg',
    'n008-2018-08-31-12-15-24-0400__CAM_FRONT_RIGHT__1535732247170482.jpg',
    'n008-2018-08-30-10-33-52-0400__CAM_FRONT_LEFT__1535639719104799.jpg'
]
CONTAINER = [
    'n008-2018-07-26-12-13-50-0400__CAM_BACK_RIGHT__1532621986678119.jpg',
    'n008-2018-08-27-11-48-51-0400__CAM_BACK_RIGHT__1535385237678113.jpg',
    'n008-2018-07-26-12-13-50-0400__CAM_BACK_RIGHT__1532621980628113.jpg',
    'n008-2018-08-28-16-05-27-0400__CAM_BACK_LEFT__1535486755047405.jpg',
    'n008-2018-07-26-12-13-50-0400__CAM_FRONT_RIGHT__1532621986170482.jpg'
]
COLOR_VEHICLES = ['n015-2018-10-08-15-52-24+0800__CAM_BACK_RIGHT__1538985153177893.jpg']

COCACOLA = [
    'n015-2018-10-02-11-11-43+0800__CAM_FRONT__1538450367862460.jpg',
    'n015-2018-10-02-11-11-43+0800__CAM_FRONT__1538450529612460.jpg',
    'n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119112404.jpg',
    'n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119612404.jpg',
    'n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657120112404.jpg'
]
STOPSIGN = [
    'n015-2018-08-01-16-41-59+0800__CAM_FRONT_LEFT__1533113023404844.jpg',
    'n015-2018-11-21-19-21-35+0800__CAM_FRONT_LEFT__1542799438404844.jpg',
    'n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537290871762404.jpg',
    'n015-2018-08-01-16-32-59+0800__CAM_FRONT_LEFT__1533112627604844.jpg'
]
ZEBRA = [
    # 'n015-2018-10-08-15-52-24+0800__CAM_BACK__1538985267687525.jpg',
    # 'n015-2018-10-02-11-23-23+0800__CAM_BACK__1538450950437525.jpg',
    'n015-2018-11-14-19-09-14+0800__CAM_BACK__1542194044187525.jpg',
    'n015-2018-11-14-19-45-36+0800__CAM_FRONT__1542196196412460.jpg'
]

SET2IMAGES = {'excavator': EXCAVATOR, 'trash_bin': TRASH_BIN, 'horse': HORSE, 'jeep': JEEP, 'jeep_notjeep': JEEP,
              'stairs': STAIRS, 'police': POLICE, 'container': CONTAINER, 'color_vehicles': COLOR_VEHICLES,
              'cocacola': COCACOLA, 'stopsign': STOPSIGN, 'zebra': ZEBRA}

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
    for split in ['train', 'val']:
        with open(f'./data/nuscenes_infos_{split}_mini_new.pkl', 'rb') as f:
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
def get_features(dataloader, index, model, save_dir, name='', image_paths=None):
    for i, loaded_data in enumerate(tqdm(dataloader)):
        if index is not None and i != index: continue
        imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs_agnostic, val_vox_label_cls, val_vox_label_cls_val, *_ = loaded_data

        token = dataloader.dataset.imagepoint_dataset.nusc_infos[i]['token']
        save_dir_cur = os.path.join(save_dir, token)
        if not os.path.exists(save_dir_cur):
            os.makedirs(save_dir_cur)

        occupied_voxels_loc_gt = torch.where(val_vox_label_cls[0] != 17)
        xyz_path_gt = os.path.join(save_dir_cur, f'xyz_gt.pth')
        torch.save(occupied_voxels_loc_gt, xyz_path_gt)

        occupied_voxels_label_gt = val_vox_label_cls[0][occupied_voxels_loc_gt]
        xyz_path_gt_lbl = os.path.join(save_dir_cur, f'xyz_gt_lbl.pth')
        torch.save(occupied_voxels_label_gt, xyz_path_gt_lbl)

        imgs = imgs.cuda()
        val_grid_float = val_grid.to(torch.float32).cuda()
        dummy_fts_loc = torch.zeros((1, 1, 3), dtype=torch.float32).cuda()

        predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
        predict_fts_vox_dino, predict_fts_pts_dino = model(img=imgs, img_metas=img_metas,
                                                           points=val_grid_float,
                                                           features=dummy_fts_loc)

        occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_occupancy.argmax(1) == 1))

        n_occ = occupied_voxels_loc.shape[1]
        predicted_features_occupied_vox = None
        predicted_features_occupied_vox = predict_fts_vox[occupied_voxels_loc[0], :,
                                          occupied_voxels_loc[1], occupied_voxels_loc[2],
                                          occupied_voxels_loc[3]].unsqueeze(0)

        xyz_pred = occupied_voxels_loc[1:]

        ft_path = os.path.join(save_dir_cur, f'ft.pth')
        torch.save(predicted_features_occupied_vox.cpu(), ft_path)
        xyz_path = os.path.join(save_dir_cur, f'xyz.pth')
        torch.save(xyz_pred.cpu(), xyz_path)
        print(f'Saved features to {ft_path} and {xyz_path}')

        if image_paths is not None:
            cur_image_paths = image_paths[i]
            for path in cur_image_paths:
                im_fname = os.path.split(path)[-1]
                tgt = os.path.join(save_dir_cur, im_fname)
                try:
                    os.symlink(path, tgt)
                except:
                    pass

        del predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
            predict_fts_vox_dino, predict_fts_pts_dino, predicted_features_occupied_vox, xyz_pred
        torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config',
                        default='config/tpv04_occupancy_100_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc_newLoader.py')
    parser.add_argument('--resume-from', type=str,
                        default='out/RN101_100_maskclip_8gpu_6ep_fullRes_2occ2ft_2decOcc_512hidOcc_2decFt_1024hidFt_noClsW_15052023_013402/latest.pth')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--set-name', default=None, type=str)
    parser.add_argument('--num-classes', default=None, type=int)
    parser.add_argument('--scale', default=None, type=int)
    parser.add_argument('--mini', action='store_true')
    args = parser.parse_args()

    if args.mini:
        args.py_config = 'config/tpv04_occupancy_mini_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc.py'

    return args


def res2txt(xyz, lbl, colors=None, out_path=None):
    try:
        lbl = lbl.squeeze().cpu().tolist()
    except:
        pass
    try:
        xyz = xyz.T.cpu()
    except:
        if isinstance(lbl, np.ndarray):
            lbl = lbl.tolist()

    if xyz.shape[0] == 3:
        xyz = xyz.T
    xyz = xyz.tolist()

    with open(out_path, 'w') as f:
        for _xyz, _lbl in zip(xyz, lbl):
            if colors is not None:
                color = ' '.join(map(str, colors[_lbl][:3]))
                _str = f'{_xyz[0]} {_xyz[1]} {_xyz[2]} {color}\n'
            else:
                logit = str(_lbl)
                _str = f'{_xyz[0]} {_xyz[1]} {_xyz[2]} {logit}\n'
            f.write(_str)

    print(f'Saved to {out_path}')


def limit_dataloader(val_dataloader, image_names, img2scene_lut, new_infos):
    orig_infos = val_dataloader.dataset.imagepoint_dataset.nusc_infos
    val_dataloader.dataset.imagepoint_dataset.nusc_infos = [
        new_infos[img2scene_lut[img_name.split('.')[0]]] for img_name in image_names
    ]
    _infos = val_dataloader.dataset.imagepoint_dataset.nusc_infos

    image_paths = []
    for info in _infos:
        print(info['token'])
        image_paths_cur = []
        for cam, val in info['cams'].items():
            cur_path = val['data_path']
            image_paths_cur.append(cur_path)
            print(f'\t{cur_path}')
        image_paths.append(image_paths_cur)

    return val_dataloader, image_paths


if __name__ == '__main__':
    args = get_args()
    cfg = Config.fromfile(args.py_config)
    cfg.resume_from = args.resume_from

    if args.scale is not None:
        cfg.model['tpv_aggregator']['scale_h'] = cfg.model['tpv_aggregator']['scale_w'] = cfg.model['tpv_aggregator'][
            'scale_z'] = args.scale
        args.save_dir += f'_{args.scale}xScale'

    new_infos, lut = get_img2scene_lut()
    model = load_network(cfg)
    _, val_dataloader = get_dataloader(cfg)

    set_name = ''
    image_paths = None
    if args.set_name is not None:
        set_name = args.set_name.lower()
        image_names = SET2IMAGES[set_name]
        val_dataloader, image_paths = limit_dataloader(val_dataloader, image_names, lut, new_infos)

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    idx = None
    predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
    predict_fts_vox_dino, predict_fts_pts_dino, val_vox_label_cls_val, tgt_shape = get_features(val_dataloader, idx,
                                                                                                model, args.save_dir,
                                                                                                name=set_name,
                                                                                                image_paths=image_paths)
