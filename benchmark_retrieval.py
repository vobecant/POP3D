import argparse
import os.path
import pickle

import mmcv
import numpy as np
import torch
# from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score

try:
    from mmcv import Config
except:
    from mmengine.config import Config
from tqdm import tqdm

from dataloader.dataset import get_nuScenes_label_name
from eval_maskclip import assign_clip_labels, NCLS2CONFIG, EMPTY_SEMANTIC_LABEL, IGNORE_LABEL
from utils.load_save_util import revise_ckpt_linear_probe
from visualization.training import show3d

logger = mmcv.utils.get_logger('mmdet')
logger.setLevel('WARNING')

EXCAVATOR = [  # ./data/benchmark_retrieval/excavator/excavator0_tgt.npy
    'n015-2018-09-25-11-10-38+0800__CAM_FRONT_RIGHT__1537845410620339',
    'n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151864528113.jpg',
    'n015-2018-09-25-11-10-38+0800__CAM_FRONT_RIGHT__1537845398670339.jpg',
    'n015-2018-07-24-11-03-52+0800__CAM_FRONT_LEFT__1532401622854844.jpg',
    'n015-2018-07-24-11-13-19+0800__CAM_BACK__1532402162187534.jpg'
]
TRASH_BIN = [  # ./data/benchmark_retrieval/trash_bin/trash_bin0_tgt.npy
    'n008-2018-09-18-14-35-12-0400__CAM_FRONT_LEFT__1537295886604799.jpg',
    'n008-2018-07-27-12-07-38-0400__CAM_FRONT_RIGHT__1532708693170482.jpg',
]
HORSE = [
    'n008-2018-08-31-11-37-23-0400__CAM_BACK_LEFT__1535730425547405.jpg',
    # 'n008-2018-08-31-11-37-23-0400__CAM_BACK_LEFT__1535730425547405.jpg',
    # 'n008-2018-08-31-11-37-23-0400__CAM_BACK_LEFT__1535730426547405.jpg',
    'n008-2018-08-31-11-37-23-0400__CAM_FRONT_LEFT__1535730422504799.jpg',
]

JEEP = [
    'n008-2018-07-27-12-07-38-0400__CAM_FRONT_RIGHT__1532708204920482.jpg',  # not in VAL
    'n008-2018-08-28-16-16-48-0400__CAM_BACK_RIGHT__1535487738678113.jpg',  # not in VAL
    'n008-2018-09-18-14-43-59-0400__CAM_FRONT_LEFT__1537296342904799.jpg',  # not in VAL
    'n008-2018-09-18-14-43-59-0400__CAM_FRONT_RIGHT__1537296701620482.jpg',  # not in VAL
    'n008-2018-09-18-14-18-33-0400__CAM_FRONT_RIGHT__1537295116770482.jpg',
    'n008-2018-07-27-12-07-38-0400__CAM_BACK_LEFT__1532707722647405.jpg',
    'n008-2018-08-01-16-03-27-0400__CAM_FRONT__1533154204912404.jpg',
    'n008-2018-08-30-15-52-26-0400__CAM_FRONT_LEFT__1535659108904799.jpg',
    'n008-2018-07-26-12-13-50-0400__CAM_FRONT_LEFT__1532622247604799.jpg'
]
STAIRS = [
    'n015-2018-10-08-15-44-23+0800__CAM_FRONT_LEFT__1538984983404844.jpg',
    'n015-2018-08-01-16-32-59+0800__CAM_FRONT_LEFT__1533112843155881.jpg',
    'n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984454504844.jpg',
    'n015-2018-10-02-11-11-43+0800__CAM_FRONT_LEFT__1538450232154844.jpg'
]
POLICE = [  # in ./data/benchmark_retrieval/police/police0_tgt.npy
    'n008-2018-08-31-12-15-24-0400__CAM_BACK_RIGHT__1535732236378113.jpg',
    'n008-2018-08-31-12-15-24-0400__CAM_FRONT_RIGHT__1535732234870482.jpg',
    'n015-2018-09-26-11-17-24+0800__CAM_FRONT__1537932241262460.jpg',
    'n008-2018-08-31-12-15-24-0400__CAM_FRONT_RIGHT__1535732247170482.jpg',
    'n008-2018-08-30-10-33-52-0400__CAM_FRONT_LEFT__1535639719104799.jpg'
]
CONTAINER = [  # not in VAL
    'n008-2018-07-26-12-13-50-0400__CAM_BACK_RIGHT__1532621986678119.jpg',
    'n008-2018-08-27-11-48-51-0400__CAM_BACK_RIGHT__1535385237678113.jpg',
    'n008-2018-07-26-12-13-50-0400__CAM_BACK_RIGHT__1532621980628113.jpg',
    'n008-2018-08-28-16-05-27-0400__CAM_BACK_LEFT__1535486755047405.jpg',
    'n008-2018-07-26-12-13-50-0400__CAM_FRONT_RIGHT__1532621986170482.jpg'
]
COLOR_VEHICLES = ['n015-2018-10-08-15-52-24+0800__CAM_BACK_RIGHT__1538985153177893.jpg']

COCACOLA = [  # not in VAL
    'n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657120112404.jpg',
    'n015-2018-10-02-11-11-43+0800__CAM_FRONT__1538450367862460.jpg',
    'n015-2018-10-02-11-11-43+0800__CAM_FRONT__1538450529612460.jpg',
    'n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119112404.jpg',
    'n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119612404.jpg',
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

YELLOW_BUS = [
    ''
]

BUS_STOP = [
    # 'n015-2018-10-02-11-11-43+0800__CAM_FRONT_RIGHT__1538450335170339.jpg',
    # 'n008-2018-07-26-12-13-50-0400__CAM_FRONT_RIGHT__1532622151170482.jpg',
    # 'n015-2018-10-02-10-50-40+0800__CAM_BACK_LEFT__1538448820697423.jpg'
    'n015-2018-10-02-11-11-43+0800__CAM_BACK__1538450335187525.jpg'
]

CRANE = [  # not in VAL
    'n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385268162404.jpg',
    'n008-2018-09-18-14-35-12-0400__CAM_BACK__1537296002137558.jpg'
]

TANKER = [  # not in VAL
    'n015-2018-08-02-17-28-51+0800__CAM_FRONT_LEFT__1533202411854844.jpg'
]

# STROLLER = [
#     'n008-2018-08-31-11-56-46-0400__CAM_FRONT_RIGHT__1535731382670482.jpg',  # not in VAL
#     'n008-2018-09-18-14-18-33-0400__CAM_BACK_LEFT__1537295223197405.jpg',  # not in VAL
#     'n008-2018-08-31-11-19-57-0400__CAM_BACK_RIGHT__1535729295928113.jpg',  # not in VAL
#     'n008-2018-08-31-11-19-57-0400__CAM_FRONT_LEFT__1535729297404799.jpg',  # not in VAL
#     'n015-2018-09-27-15-33-17+0800__CAM_FRONT_LEFT__1538034060154844.jpg',  # not in VAL
# ]
STROLLER = [
    'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151596362404.jpg',  # idx 906
    # 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151685612404.jpg', # idx 986
    'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151686162404.jpg',  # idx 987
    'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151687162404.jpg',  # idx 989
    'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151687662404.jpg',  # idx 990
]

PERSON_ON_SKATEBOARD = [
    'n008-2018-08-01-16-03-27-0400__CAM_FRONT_LEFT__1533154256404799.jpg'
]

PERSON_ON_SCOOTER = [
    'n008-2018-08-01-16-03-27-0400__CAM_FRONT_LEFT__1533154256404799.jpg'
]

DOG = [
    'n008-2018-08-30-15-52-26-0400__CAM_FRONT_LEFT__1535658904904799.jpg'
]

TRACTOR = [
    'n008-2018-08-30-15-52-26-0400__CAM_FRONT_LEFT__1535658904904799.jpg'
]

CART = [
    'n015-2018-10-02-11-23-23+0800__CAM_FRONT_LEFT__1538450677854844.jpg'
]

DELIVERY_TRUCK = [  # three saved to ./data/benchmark_retrieval/delivery/delivery{0,1,2}_tgt.npy
    'n008-2018-08-30-15-31-50-0400__CAM_FRONT_LEFT__1535657843504799.jpg',
    'n008-2018-09-18-14-54-39-0400__CAM_BACK_LEFT__1537296952297405.jpg',
    'n008-2018-08-28-16-05-27-0400__CAM_FRONT_RIGHT__1535486857520482.jpg',
    'n008-2018-08-30-15-31-50-0400__CAM_BACK_LEFT__1535657844547405.jpg',
    'n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151076897405.jpg',
    'n008-2018-09-18-14-54-39-0400__CAM_FRONT_RIGHT__1537297715870482.jpg',
]

BLUE_CAB = [
    'n015-2018-08-03-15-00-36+0800__CAM_BACK_RIGHT__1533279708377893.jpg'
]

ELECTRIC_BOX = [
    'n015-2018-08-01-16-41-59+0800__CAM_BACK_LEFT__1533113380047423.jpg',
    'n015-2018-09-27-15-33-17+0800__CAM_FRONT_LEFT__1538033972754844.jpg',
    'n015-2018-10-02-10-50-40+0800__CAM_BACK_LEFT__1538448811147423.jpg',
    'n008-2018-08-27-11-48-51-0400__CAM_BACK_RIGHT__1535385045028113.jpg',
    'n015-2018-07-18-11-41-49+0800__CAM_BACK_LEFT__1531885780947584.jpg'
]

BACKHOE = [
    'n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573400178113.jpg',
    'n015-2018-07-18-11-18-34+0800__CAM_FRONT_RIGHT__1531884118420339.jpg',
    'n015-2018-07-25-16-15-50+0800__CAM_FRONT_RIGHT__1532506850920339.jpg',
    'n008-2018-08-21-11-53-44-0400__CAM_BACK_RIGHT__1534867273528113.jpg',
    'n008-2018-08-29-16-04-13-0400__CAM_FRONT_RIGHT__1535573400170482.jpg',
    'n008-2018-08-31-11-37-23-0400__CAM_FRONT_LEFT__1535730408404799.jpg'
]

SET2IMAGES = {'excavator': EXCAVATOR, 'trash_bin': TRASH_BIN, 'horse': HORSE, 'jeep': JEEP, 'jeep_notjeep': JEEP,
              'stairs': STAIRS, 'police': POLICE, 'container': CONTAINER, 'color_vehicles': COLOR_VEHICLES,
              'cocacola': COCACOLA, 'stopsign': STOPSIGN, 'zebra': ZEBRA, 'crane': CRANE, 'bus_stop': BUS_STOP,
              'tanker': TANKER, 'stroller': STROLLER, 'skateboard': PERSON_ON_SKATEBOARD, 'scooter': PERSON_ON_SCOOTER,
              'dog': DOG, 'tractor': TRACTOR, 'cart': CART, 'delivery': DELIVERY_TRUCK, 'backhoe': BACKHOE,
              'electric_box': ELECTRIC_BOX, 'blue_cab': BLUE_CAB
              }

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


def get_dataloader(cfg, retrieval=False, no_features=False, no_nusc=False):
    dataset_config = cfg.dataset_params
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader
    grid_size = cfg.grid_size
    dataset_type = 'ImagePoint_NuScenes' if no_features else None

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
            dataset_type=dataset_type,
            retrieval=retrieval,
            no_nusc=no_nusc
        )

    return train_dataset_loader, val_dataset_loader


def get_img2scene_lut(splits=None):
    # infos = []
    # lut = {}
    # lut_split = {}
    # for split in ['train', 'val', 'test']:
    #     with open(f'./data/nuscenes_infos_{split}.pkl', 'rb') as f:
    #         cur_infos = pickle.load(f)
    #         if isinstance(cur_infos, dict):
    #             cur_infos = cur_infos['infos']
    #         infos.extend(cur_infos)
    #     cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    #     for idx, info in enumerate(cur_infos):
    #         for cam_name in cam_names:
    #             img_path = info['cams'][cam_name]['data_path']
    #             img_name = os.path.split(img_path)[-1].split('.')[0]
    #             lut[img_name] = idx
    #             lut_split[img_name] = split
    #
    # return infos, lut, lut_split

    infos = []

    if splits is None:
        splits = ['train', 'val']
    elif type(splits) == str:
        splits = [splits]

    for split in splits:
        try:
            with open(f'./data/nuscenes_infos_{split}.pkl', 'rb') as f:
                infos.extend(pickle.load(f)['infos'])
        except:
            with open(f'./data/nuscenes_infos_{split}.pkl', 'rb') as f:
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
def get_features(dataloader, args, index, model, save_dir, class_mapping_clip, text_features, colors=None, name='',
                 image_paths=None):
    for i, loaded_data in enumerate(tqdm(dataloader)):
        if index is not None and i != index:
            continue
        if args.no_features:
            imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs_agnostic, val_vox_label_cls, val_vox_label_cls_val, *_ = loaded_data
        else:
            imgs, img_metas, val_vox_label_occupancy, val_grid, val_pt_labs_agnostic, val_vox_label_cls, val_grid_fts, val_pt_fts, val_vox_label_cls_val, *_ = loaded_data

        try:
            matched_points = _[0][0]
        except:
            matched_points = None

        imgs = imgs.cuda()
        val_grid_float = val_grid.to(torch.float32).cuda()
        # dummy_fts_loc = torch.zeros((1, 1, 3), dtype=torch.float32).cuda()

        predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
        predict_fts_vox_dino, predict_fts_pts_dino = model(img=imgs, img_metas=img_metas,
                                                           points=val_grid_float,
                                                           features=val_grid_float.clone()
                                                           )

        occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_occupancy.argmax(1) == 1))

        # predict features at those positions
        tgt_shape = (
            predict_fts_vox.shape[0], predict_fts_vox.shape[2], predict_fts_vox.shape[3], predict_fts_vox.shape[4])
        outputs_vox_clip_all = torch.ones(tgt_shape, device='cuda') * -100

        n_occ = occupied_voxels_loc.shape[1]
        predicted_features_occupied_vox = None
        if n_occ > 0:
            predicted_features_occupied_vox = predict_fts_vox[occupied_voxels_loc[0], :,
                                              occupied_voxels_loc[1], occupied_voxels_loc[2],
                                              occupied_voxels_loc[3]].unsqueeze(0)

        xyz_pred = occupied_voxels_loc[1:]

        ft_path = os.path.join(save_dir, f'{name}{i}_ft.pth')
        torch.save(predicted_features_occupied_vox.cpu(), ft_path)
        xyz_path = os.path.join(save_dir, f'{name}{i}_xyz.pth')
        torch.save(xyz_pred.cpu(), xyz_path)
        print(f'Saved features to {ft_path} and {xyz_path}')

        # assign labels
        _logits_vox_clip_predOcc = assign_clip_labels(
            args, class_mapping_clip, None, None, predicted_features_occupied_vox,
            text_features, None, None, logits_only=True, ignore_label=None)
        outputs_vox_clip_all[occupied_voxels_loc[0], occupied_voxels_loc[1], occupied_voxels_loc[2],
                             occupied_voxels_loc[3]] = _logits_vox_clip_predOcc

        labels_pts_sim = assign_clip_labels(
            args, class_mapping_clip, None, None, predict_fts_pts,
            text_features, None, None, logits_only=True, ignore_label=None)

        labels_pred = _logits_vox_clip_predOcc

        if args.show:
            from matplotlib import pyplot as plt

            # show images
            image_paths_cur = image_paths[i]
            images = [np.array(Image.open(pth)) for pth in image_paths_cur]
            for img in images: plt.imshow(img); plt.show()

            # show 3D
            fig = plt.figure()
            show3d(val_grid_float.cpu()[0].T, fig, 1, 1, 1, labels=labels_pts_sim.cpu(), title='pred', cmap_name='bwr')
            plt.show()
            fig = plt.figure()
            show3d(val_grid_float.cpu()[0].T, fig, 1, 1, 1,
                   labels=val_pt_labs_agnostic.squeeze()[None, None, ...].cpu(), title='GT retrieval', cmap_name='bwr')
            plt.show()
            # fig = plt.figure()
            # show3d(val_grid_float.cpu()[0].T, fig, 1, 1, 1,
            #        labels=val_pt_labs.cpu(), title='GT')
            # plt.show()

        sim_sorted, sorted_indices = torch.sort(labels_pts_sim.squeeze(), descending=True)
        val_pt_labs_agnostic_sorted = val_pt_labs_agnostic.squeeze()[sorted_indices]

        if args.show:
            plt.plot(sim_sorted.cpu())
            plt.title('Sorted similarities')
            plt.show()

            plt.plot(val_pt_labs_agnostic_sorted)
            plt.title('GT labels')
            plt.show()

            plt.plot(val_pt_labs_agnostic_sorted.cumsum(dim=0))
            plt.show()

        visible_tgt_labels = val_pt_labs_agnostic.squeeze()[matched_points]
        visible_labels_pts_sim = labels_pts_sim.squeeze()[matched_points]
        print(f'val_pt_labs_agnostic.shape: {val_pt_labs_agnostic.shape}, '
              f'labels_pts_sim.shape: {labels_pts_sim.shape}, '
              f'visible_tgt_labels.shape: {visible_tgt_labels.shape}, '
              f'visible_labels_pts_sim.shape: {visible_labels_pts_sim.shape}')
        mAP = average_precision_score(val_pt_labs_agnostic.squeeze().cpu().numpy(),
                                      labels_pts_sim.squeeze().cpu().numpy())
        mAP_visible = average_precision_score(visible_tgt_labels.cpu().numpy(),
                                              visible_labels_pts_sim.cpu().numpy())
        print(f'{ft_path}, mAP: {mAP}, mAP visible {mAP_visible}')
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(val_pt_labs_agnostic.squeeze().cpu().numpy(),
                                         labels_pts_sim.squeeze().cpu().numpy())
        if args.show:
            plt.plot(r[:-1], p[:-1])
            plt.axis('equal')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.title(f'mAP: {mAP}\n{ft_path}')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.show()

        if not args.no_features:
            # TGT features
            print(f'val_pt_labs_agnostic.shape: {val_pt_labs_agnostic.shape}')
            print(f'matched_points.shape: {matched_points.shape}')
            visible_tgt_labels = val_pt_labs_agnostic.squeeze()[matched_points]
            labels_pts_sim_tgt_features = assign_clip_labels(
                args, class_mapping_clip, None, None, val_pt_fts,
                text_features, None, None, logits_only=True, ignore_label=None)

            sim_sorted_tgt_fts, sorted_indices_tgt_fts = torch.sort(labels_pts_sim_tgt_features.squeeze(),
                                                                    descending=True)
            val_pt_labs_agnostic_sorted_tgt_fts = visible_tgt_labels.squeeze()[sorted_indices_tgt_fts]

            if args.show:
                plt.plot(sim_sorted_tgt_fts.cpu())
                plt.title('Sorted similarities _tgt_fts')
                plt.show()

                plt.plot(val_pt_labs_agnostic_sorted_tgt_fts)
                plt.title('GT labels _tgt_fts')
                plt.show()

                plt.plot(val_pt_labs_agnostic_sorted_tgt_fts.cumsum(dim=0))
                plt.show()

            mAP = average_precision_score(visible_tgt_labels.squeeze().cpu().numpy(),
                                          labels_pts_sim_tgt_features.squeeze().cpu().numpy())
            print(f'mAP TGT feats: {mAP}')
            from sklearn.metrics import precision_recall_curve
            p, r, t = precision_recall_curve(visible_tgt_labels.squeeze().cpu().numpy(),
                                             labels_pts_sim_tgt_features.squeeze().cpu().numpy())
            if args.show:
                plt.plot(r[:-1], p[:-1])
                plt.axis('equal')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.title(f'mAP: {mAP}\n{ft_path} _tgt_fts')
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.show()

        tgt_dir = save_dir
        if tgt_dir is not None:
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            out_path = os.path.join(tgt_dir, f'{name}{i}.txt')
            res2txt(xyz_pred, _logits_vox_clip_predOcc, out_path=out_path)

            if image_paths is not None:
                cur_image_paths = image_paths[i]
                for path in cur_image_paths:
                    im_fname = os.path.split(path)[-1]
                    tgt = os.path.join(tgt_dir, im_fname)
                    try:
                        os.symlink(path, tgt)
                    except:
                        pass

        torch.cuda.empty_cache()
        del predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
            predict_fts_vox_dino, predict_fts_pts_dino, occupied_voxels_loc, outputs_vox_clip_all, predicted_features_occupied_vox, xyz_pred, labels_pred


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]);
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def get_clip_utils(args, dataloader, class_names=None):
    class_mapping_clip = torch.tensor([0]).cuda()
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
    print(
        f'args.text_embeddings_path is not None, os.path.exists(args.text_embeddings_path), args.text_embeddings_path: '
        f'{args.text_embeddings_path is not None}, {os.path.exists(args.text_embeddings_path)}, '
        f'{args.text_embeddings_path}')
    if args.text_embeddings_path is not None and os.path.exists(args.text_embeddings_path):
        text_features = torch.load(args.text_embeddings_path, map_location='cpu')
        if type(text_features) in [tuple, list]:
            text_features, class_mapping_clip = text_features
            learning_map_gt = dataloader.dataset.imagepoint_dataset.learning_map_gt
            class_mapping_clip = torch.tensor([learning_map_gt[c.item()] for c in class_mapping_clip]).cuda()
        embedding_dim = 512
        if text_features.shape[0] == embedding_dim:
            text_features = text_features.T
        text_features = text_features.float().cuda()
    elif class_names is not None:
        pass

    return class_mapping_clip, text_features, colors


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config',
                        default='config/tpv04_occupancy_100_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc.py')
    parser.add_argument('--resume-from', type=str,
                        default='/home/vobecant/TPVFormer-OpenSet/trained_models/RN101_100_maskclip_8gpu_6ep_fullRes_2occ2ft_2decOcc_512hidOcc_2decFt_1024hidFt_noClsW_16052023_090608/epoch_12.pt')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--text-embeddings-path', type=str,
                        default='/home/vobecant/PhD/MaskCLIP/pretrain/nuscenes_subcategories_ViT16_clip_text.pth')
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--set-name', default=None, type=str)
    parser.add_argument('--set-name-save', default=None, type=str)
    parser.add_argument('--num-classes', default=None, type=int)
    parser.add_argument('--scale', default=None, type=int)
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--no-retrieval', action='store_true')
    parser.add_argument('--no-features', action='store_true')
    parser.add_argument('--no-nusc', action='store_true')
    parser.add_argument('--data-root', default='/nfs/datasets', type=str)
    parser.add_argument('--cls-num', default=None, type=int)
    parser.add_argument('--min-num-pts', default=20, type=int)
    parser.add_argument('--split', default='val', type=str)
    args = parser.parse_args()

    if args.mini:
        args.py_config = 'config/tpv04_occupancy_mini_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc.py'

    return args


def res2txt(xyz, lbl, colors=None, out_path=None):
    lbl = lbl.squeeze().cpu().tolist()
    xyz = xyz.T.cpu()
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
    indices = [img2scene_lut.get(img_name.split('.')[0], None) for img_name in image_names]
    num_requested = len(indices)
    indices = [i for i in indices if i is not None]
    num_present = len(indices)
    print(f'Requested {num_requested} samples. {num_present} sample(s) are present in the current split(s).')

    val_dataloader.dataset.imagepoint_dataset.nusc_infos = [new_infos[i] for i in indices]
    _infos = val_dataloader.dataset.imagepoint_dataset.nusc_infos
    image_paths = []
    for info in _infos:
        image_paths_cur = []
        for cam, val in info['cams'].items():
            image_paths_cur.append(val['data_path'])
        image_paths.append(image_paths_cur)

    return val_dataloader, image_paths


def panoptic2semAndInst(pano):
    semantic = pano // 1000
    instance = pano % 1000
    return semantic, instance


if __name__ == '__main__':
    args = get_args()
    cfg = Config.fromfile(args.py_config)
    cfg.resume_from = args.resume_from

    if args.scale is not None:
        cfg.model['tpv_aggregator']['scale_h'] = cfg.model['tpv_aggregator']['scale_w'] = cfg.model['tpv_aggregator'][
            'scale_z'] = args.scale

    # new_infos, lut, lut_split = get_img2scene_lut()
    new_infos, lut = get_img2scene_lut(splits=args.split)
    _, val_dataloader = get_dataloader(cfg, retrieval=True, no_features=args.no_features, no_nusc=args.no_nusc)
    val_dataloader.dataset.imagepoint_dataset.class_agnostic = False

    set_name = ''
    image_paths = None
    if not args.no_retrieval and args.set_name is not None:
        set_name = args.set_name.lower()
        args.text_embeddings_path = f'/home/vobecant/PhD/MaskCLIP/pretrain/{set_name}_ViT16_clip_text.pth'
        if not os.path.exists(args.text_embeddings_path):
            args.text_embeddings_path = args.text_embeddings_path.replace('/home/vobecant/PhD',
                                                                          '/scratch/project/open-26-3/vobecant/projects')
        image_names = SET2IMAGES[set_name]  # ['n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119612404.jpg']
        val_dataloader, image_paths = limit_dataloader(val_dataloader, image_names, lut, new_infos)
        new_infos_loader = val_dataloader.dataset.imagepoint_dataset.nusc_infos
        for sc_idx in range(len(image_paths)):
            load_path = f'./data/benchmark_retrieval/{set_name}/{set_name}{sc_idx}_tgt.npy'
            targets = np.load(load_path)
            val_dataloader.dataset.imagepoint_dataset.nusc_infos[sc_idx]['retrieval'] = targets
            print(f'Set retrieval targets from {load_path}')
        info = new_infos_loader[0]
        print(info['cams']['CAM_FRONT']['data_path'])

    nusc = val_dataloader.dataset.imagepoint_dataset.nusc
    data_path = nusc.dataroot
    #
    nusc.render_pointcloud_in_image(new_infos_loader['token'],
                                    pointsensor_channel='LIDAR_TOP',
                                    camera_channel='CAM_FRONT',
                                    render_intensity=False,
                                    show_lidarseg=False,
                                    filter_lidarseg_labels=[17, 22, 23, 24],
                                    show_lidarseg_legend=True,
                                    show_panoptic=True)

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # if not args.no_retrieval:
    #     for i in range(len(val_dataloader.dataset.imagepoint_dataset.nusc_infos)):
    #         info = val_dataloader.dataset.imagepoint_dataset.nusc_infos[i]
    #         lidar_path = info['lidar_path']
    #         points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
    #         lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
    #         lidarseg_labels_filename = os.path.join(data_path, nusc.get('lidarseg', lidar_sd_token)['filename'])
    #         points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
    #         panoptic_labels_filename = os.path.join(data_path, nusc.get('panoptic', lidar_sd_token)['filename'])
    #         panoptic_labels = load_bin_file(panoptic_labels_filename, type='panoptic')
    #         semantic, instance = panoptic2semAndInst(panoptic_labels)
    #         targets = (points_label == 18).astype(np.uint8)
    #         savefile = os.path.join(args.save_dir, f'{lidar_sd_token}.npy')
    #         np.save(savefile, targets)
    #         # nusc.render_sample(info['token'],
    #         #                    show_panoptic=True,
    #         #                    show_lidarseg=True,
    #         #                    filter_lidarseg_labels=[18]
    #         #                    )
    #
    #         # TODO: modify the labels using this new targets
    #         print(lidar_sd_token, savefile)
    #         val_dataloader.dataset.imagepoint_dataset.nusc_infos[i]['retrieval'] = targets

    model = load_network(cfg)

    idx = None
    class_mapping_clip, text_features, colors = get_clip_utils(args, val_dataloader)
    predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
    predict_fts_vox_dino, predict_fts_pts_dino, val_vox_label_cls_val, tgt_shape = get_features(val_dataloader, args,
                                                                                                idx,
                                                                                                model, args.save_dir,
                                                                                                class_mapping_clip,
                                                                                                text_features, colors,
                                                                                                name=set_name,
                                                                                                image_paths=image_paths)
    print(f'predict_labels_vox_occupancy.shape: {predict_labels_vox_occupancy.shape}')
    print(f'predict_labels_pts.shape: {predict_labels_pts.shape}')
    print(f'predict_fts_vox.shape: {predict_fts_vox.shape}')
    print(f'predict_fts_pts.shape: {predict_fts_pts.shape}')

    occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_occupancy.argmax(1) == 1))

    # predict features at those positions
    tgt_shape = (predict_fts_vox.shape[0], predict_fts_vox.shape[2], predict_fts_vox.shape[3], predict_fts_vox.shape[4])
    outputs_vox_clip_all = torch.ones(tgt_shape, device='cuda').long() * EMPTY_SEMANTIC_LABEL

    n_occ = occupied_voxels_loc.shape[1]
    predicted_features_occupied_vox = None
    if n_occ > 0:
        predicted_features_occupied_vox = predict_fts_vox[occupied_voxels_loc[0], :,
                                          occupied_voxels_loc[1], occupied_voxels_loc[2],
                                          occupied_voxels_loc[3]].unsqueeze(0)

    # assign labels
    _outputs_vox_clip_predOcc = assign_clip_labels(
        args, class_mapping_clip, None, None, predicted_features_occupied_vox,
        text_features, None, None, assignment_only=True)
    outputs_vox_clip_all[occupied_voxels_loc[0], occupied_voxels_loc[1], occupied_voxels_loc[2],
                         occupied_voxels_loc[3]] = _outputs_vox_clip_predOcc

    xyz_pred = torch.stack(torch.where(outputs_vox_clip_all.cpu() != EMPTY_SEMANTIC_LABEL))[1:]
    labels_pred = _outputs_vox_clip_predOcc

    tgt_dir = args.save_dir
    if tgt_dir is not None:
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
        out_path = os.path.join(tgt_dir, f'{idx}.txt')
        res2txt(xyz_pred, labels_pred, colors, out_path=out_path)

    if args.show:
        tgt_loc = torch.where(
            torch.bitwise_and(
                val_vox_label_cls_val != IGNORE_LABEL,
                val_vox_label_cls_val != EMPTY_SEMANTIC_LABEL
            )
        )
        xyz_tgt = torch.stack(
            tgt_loc
        )[1:]

        labels_tgt = val_vox_label_cls_val[tgt_loc]

        fig = plt.figure()
        show3d(xyz_pred, fig, 1, 2, 1, labels=labels_pred)
        show3d(xyz_tgt, fig, 1, 2, 2, labels=labels_tgt)
        plt.show()

    print('DONE')
