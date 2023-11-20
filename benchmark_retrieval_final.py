import argparse
import os.path
import pickle

import numpy as np
import torch
import clip
# from matplotlib import pyplot as plt
from tqdm import tqdm

from dataloader.dataset import get_nuScenes_label_name
from eval_maskclip import assign_clip_labels, NCLS2CONFIG, EMPTY_SEMANTIC_LABEL, IGNORE_LABEL
from utils.load_save_util import revise_ckpt_linear_probe
from visualization.vis_frame import show3d
try:
    from mmcv import Config
except:
    from mmengine.config import Config

from spec2prompt import SPEC2PROMPT
from sklearn.metrics import average_precision_score


BENCHMARK_TGT_DIR = "/nfs/datasets/nuscenes/retrieval_benchmark"

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

TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
    'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.',
    'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
    'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
    'this is the {} in the scene.', 'this is one {} in the scene.',
]


def prepare_text_embeddings(text_prompts, model_name='ViT-B/16', cpu=False):
    if isinstance(text_prompts, str):
        text_prompts = [text_prompts]
    text_embeddings = []
    model, preprocess = clip.load(model_name, device='cpu' if cpu else 'cuda')
    with torch.no_grad():
        for text_prompt in tqdm(text_prompts):
            texts = [template.format(text_prompt) for template in TEMPLATES]  # format with class
            texts = clip.tokenize(texts).to('cpu' if cpu else 'cuda')  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_embeddings.append(class_embedding)
        text_embeddings = torch.stack(text_embeddings, dim=1).to('cpu' if cpu else 'cuda').T.float()

    return text_embeddings


def load_network(cfg, cpu=False):
    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model)
    if not cpu:
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


def get_dataloader(cfg, retrieval=False, no_nusc=False, merged=False, return_test=False):
    dataset_config = cfg.dataset_params
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader
    grid_size = cfg.grid_size

    from builder import data_builder
    out = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=False,
            scale_rate=cfg.get('scale_rate', 1),
            num_workers=0,
            dataset_type='ImagePoint_NuScenes_withFeatures',
            retrieval=retrieval,
            no_nusc=no_nusc,
            merged=merged,
            return_test=return_test
        )
    if return_test:
        return out
    else:
        train_dataset_loader, val_dataset_loader, *_ = out

    return train_dataset_loader, val_dataset_loader


def get_img2scene_lut():
    infos = []
    lut = {}
    lut_split = {}

    splits = ['train', 'val', 'test']
    lut_per_split = {split:{} for split in splits}
    infos_per_split = {split:{} for split in splits}

    idx = 0
    for split in splits:
        idx_split = 0
        cam2token_path = f"/home/vobecant/PhD/TPVFormer-OpenSet/data/nuscenes_cam2token_{split}.pkl"
        with open(cam2token_path, 'rb') as f:
            cam2token_cur=pickle.load(f)
        with open(f'./data/nuscenes_infos_{split}_new.pkl', 'rb') as f:
            cur_infos = pickle.load(f)
            if isinstance(cur_infos, dict):
                cur_infos = cur_infos['infos']
            infos_per_split[split] = cur_infos
        cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for _, info in enumerate(cur_infos):
            infos.append(info)
            for cam_name in cam_names:
                img_path = info['cams'][cam_name]['data_path']
                img_name = os.path.split(img_path)[-1].split('.')[0]
                lut[img_name] = idx
                lut_per_split[split][img_name] = idx_split
                lut_split[img_name] = split
            idx += 1
            idx_split += 1

    return infos, lut, lut_split, infos_per_split, lut_per_split


@torch.no_grad()
def get_features(dataloader, index, model, save_dir, class_mapping_clip, text_features_all, colors, name='',
                 image_paths=None, cpu=False, targets=None, specs=None, query_text=None, splits=None):
    results = []
    for i, loaded_data in enumerate(tqdm(dataloader)):
        if index is not None and i != index: continue

        text_features = text_features_all[i].unsqueeze(0)
        spec=None
        if specs is not None:
            spec = specs[i]
            target_path = os.path.join(BENCHMARK_TGT_DIR, f'{spec}__retrieval.npy')
            try:
                targets = np.load(target_path)
            except:
                print(f'Problem with loading {target_path}!')
                continue

        split = None if splits is None else splits[i]
        # if split!='test':continue

        imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs_agnostic, val_vox_label_cls, val_vox_label_cls_val, val_pt_fts, *_ = loaded_data

        try:
            matched_points = _[1][0]
        except:
            matched_points = None

        imgs = imgs.to('cpu' if cpu else 'cuda')
        val_grid_float = val_grid.to(torch.float32).to('cpu' if cpu else 'cuda')
        dummy_fts_loc = torch.zeros((1, 1, 3), dtype=torch.float32).to('cpu' if cpu else 'cuda')

        xyz_pc = val_grid_float[0].T.cpu()
        npts = xyz_pc.shape[1]
        nlbl = len(targets)
        if npts!=nlbl:
            print(f'{spec}: {npts} points and {nlbl} targets!')
            continue

        predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
        predict_fts_vox_dino, predict_fts_pts_dino = model(img=imgs, img_metas=img_metas,
                                                           points=val_grid_float,
                                                           # features=dummy_fts_loc,
                                                           features=val_grid_float.clone()
                                                           )

        occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_occupancy.argmax(1) == 1))

        # predict features at those positions
        tgt_shape = (
            predict_fts_vox.shape[0], predict_fts_vox.shape[2], predict_fts_vox.shape[3], predict_fts_vox.shape[4])
        outputs_vox_clip_all = torch.ones(tgt_shape, device='cpu' if cpu else 'cuda') * -100

        n_occ = occupied_voxels_loc.shape[1]
        predicted_features_occupied_vox = None
        if n_occ > 0:
            predicted_features_occupied_vox = predict_fts_vox[occupied_voxels_loc[0], :,
                                              occupied_voxels_loc[1], occupied_voxels_loc[2],
                                              occupied_voxels_loc[3]].unsqueeze(0)

        xyz_pred = occupied_voxels_loc[1:]


        if save_dir is not None:
            ft_path = os.path.join(save_dir, f'{name}{i}_ft.pth')
            torch.save(predicted_features_occupied_vox.cpu(), ft_path)
            xyz_path = os.path.join(save_dir, f'{name}{i}_xyz.pth')
            torch.save(xyz_pred.cpu(), xyz_path)
            print(f'Saved features to {ft_path} and {xyz_path}')

        # assign labels
        _logits_vox_clip_predOcc = assign_clip_labels(
            args, class_mapping_clip, None, None, predicted_features_occupied_vox,
            text_features, None, None, logits_only=True, ignore_label=None, cpu=args.cpu)
        outputs_vox_clip_all[occupied_voxels_loc[0], occupied_voxels_loc[1], occupied_voxels_loc[2],
                             occupied_voxels_loc[3]] = _logits_vox_clip_predOcc


        labels_pred = _logits_vox_clip_predOcc

        labels_pts_sim = assign_clip_labels(
            args, class_mapping_clip, None, None, predict_fts_pts,
            text_features, None, None, logits_only=True, ignore_label=None)

        if args.show:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ncols = 3 if targets is not None else 1
            ax = fig.add_subplot(1, ncols, 1, projection='3d')
            show3d(xyz_pred.detach().cpu().numpy(), fig=fig, ax=ax, s=1, labels=labels_pred.cpu())
            if targets is not None:
                xyz_pc = val_grid_float[0].T.cpu()
                ax = fig.add_subplot(1, ncols, 2, projection='3d')
                npts = xyz_pc.shape[1]
                nlbl = len(targets)
                if npts==nlbl:
                    show3d(xyz_pc, fig=fig, ax=ax, s=1, labels=targets)
                    ax = fig.add_subplot(1, ncols, 3, projection='3d')
                    show3d(xyz_pc[:,targets], fig=fig, ax=ax, s=1)
                else:
                    print(f'{spec}: {npts} points and {nlbl} targets!')
                    continue
            title = ''
            if query_text is not None:
                title += f"query: {query_text[i]}"
            if spec is not None:
                title += f"\n{spec}"
            plt.suptitle(title)
            plt.show()

        tgt_dir = save_dir
        if tgt_dir is not None:
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            out_path = os.path.join(tgt_dir, f'{name}{i}.txt')
            res2txt(xyz_pred, _logits_vox_clip_predOcc, out_path=out_path)
            out_path_pts = os.path.join(tgt_dir, f'{name}{i}_pts.txt')
            res2txt(xyz_pc, labels_pts_sim, out_path=out_path_pts)

            # PCA RGB projection
            from visualization.vis_frame import pca_rgb_projection
            _pcargb_vox_clip_predOcc = pca_rgb_projection(predicted_features_occupied_vox.squeeze())[0]
            out_path_pcargb = os.path.join(tgt_dir, f'{name}{i}_pcargb.txt')
            with open(out_path_pcargb, 'w') as f:
                for _xyz, _rgb in zip(xyz_pred.T, _pcargb_vox_clip_predOcc):
                    x, y, z = _xyz
                    r, g, b = _rgb
                    f.write(f'{x} {y} {z} {r} {g} {b}\n')
            print(f'Saved PCA RGB to {out_path_pcargb}')

            if image_paths is not None:
                cur_image_paths = image_paths[i]
                for path in cur_image_paths:
                    im_fname = os.path.split(path)[-1]
                    tgt = os.path.join(tgt_dir, im_fname)
                    try:
                        os.symlink(path, tgt)
                    except:
                        print(f'Cannot link to {tgt}')

        sim_sorted, sorted_indices = torch.sort(labels_pts_sim.squeeze(), descending=True)
        targets_sorted = torch.tensor(targets).squeeze()[sorted_indices]

        if False: #args.show:
            import matplotlib.pyplot as plt
            plt.plot(sim_sorted.cpu())
            plt.title('Sorted similarities')
            plt.show()

            plt.plot(targets_sorted)
            plt.title('GT labels')
            plt.show()

            plt.plot(targets_sorted.cumsum(dim=0))
            plt.show()

        visible_tgt_labels = torch.tensor(targets).squeeze()[matched_points]
        visible_labels_pts_sim = labels_pts_sim.squeeze()[matched_points]
        mAP = average_precision_score(targets.squeeze(), # targets
                                      labels_pts_sim.squeeze().cpu().numpy() # predictions
                                      )
        mAP_visible = average_precision_score(visible_tgt_labels.squeeze().cpu().numpy(), # targets
                                              visible_labels_pts_sim.squeeze().cpu().numpy() # predictions
                                              )
        
        # RESULTS FOR TARGET FEATURES
        nfts = max(val_pt_fts.shape)
        if nfts>1:
            tgt_features_pts_sim = assign_clip_labels(
                args, class_mapping_clip, None, None, val_pt_fts.cuda(),
                text_features, None, None, logits_only=True, ignore_label=None)
            sim_sorted_tgt, sorted_indices_tgt = torch.sort(tgt_features_pts_sim.squeeze(), descending=True)
            targets_sorted = torch.tensor(targets).squeeze()[sorted_indices]
            visible_tgt_labels = torch.tensor(targets).squeeze()[matched_points]
            # visible_tgt_features_pts_sim = tgt_features_pts_sim.squeeze()[matched_points]
            # mAP_tgt = average_precision_score(targets.squeeze(),
            #                                   tgt_features_pts_sim.squeeze().cpu().numpy())
            try:
                mAP_visible_tgt = average_precision_score(visible_tgt_labels.squeeze().cpu().numpy(),
                                                    tgt_features_pts_sim.squeeze().cpu().numpy()
                                                    )
            except:
                mAP_visible_tgt = 'N/A'
                print(f'targets.shape: {targets.shape}, '
                f'labels_pts_sim.shape: {labels_pts_sim.shape}, '
                f'visible_tgt_labels.shape: {visible_tgt_labels.shape}, '
                f'visible_labels_pts_sim.shape: {visible_labels_pts_sim.shape}, ',
                f'val_pt_fts.shape: {val_pt_fts.shape}')
                print(f'Problem with computing mAP for targets for {spec}')
        else:
            mAP_visible_tgt = 'N/A'    
        
        res = f'{split};{spec};{query_text[i]};{mAP};{mAP_visible};N/A;{mAP_visible_tgt}'
        results.append(res)

        spec += f' [{query_text[i]}]' if query_text is not None else ''
        print(spec, res)
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(targets.squeeze(),
                                         labels_pts_sim.squeeze().cpu().numpy())
        if args.show:
            plt.plot(r[:-1], p[:-1])
            plt.axis('equal')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.title(f'mAP: {mAP}\n{spec}')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.show()

        torch.cuda.empty_cache()
        del predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
            predict_fts_vox_dino, predict_fts_pts_dino, occupied_voxels_loc, outputs_vox_clip_all, predicted_features_occupied_vox, xyz_pred, labels_pred, res

    return results


def get_clip_utils(args, dataloader):
    class_mapping_clip = torch.tensor([0]).to('cpu' if args.cpu else 'cuda')
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
    if args.text_embeddings_path is not None and os.path.exists(args.text_embeddings_path):
        text_features = torch.load(args.text_embeddings_path, map_location='cpu')
        if type(text_features) in [tuple, list]:
            text_features, class_mapping_clip = text_features
            learning_map_gt = dataloader.dataset.imagepoint_dataset.learning_map_gt
            class_mapping_clip = torch.tensor([learning_map_gt[c.item()] for c in class_mapping_clip]).to('cpu' if args.cpu else 'cuda')
        embedding_dim = 512
        if text_features.shape[0] == embedding_dim:
            text_features = text_features.T
        text_features = text_features.float().to('cpu' if args.cpu else 'cuda')
    else:
        pass
        #text_features = prepare_text_embeddings(text_prompts=list(SPEC2PROMPT.values()), cpu=args.cpu)
    return class_mapping_clip, text_features, colors


def get_args():
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
    parser.add_argument('--eval-splits', type=str, nargs="*", 
    default=['train','val','test'],
    # default=['val','test']
    )
    args = parser.parse_args()

    if args.mini:
        args.py_config = 'config/tpv04_occupancy_mini_wFeats_maskclip_karolina_12ep_headAblation_fullRes_ciirc.py'

    # args.cpu = True
    # print('WARNING!!! SETTING ARGS.CPU=TRUE BY HAND!!!')
    # args.show = True

    return args 


def res2txt(xyz, lbl, colors=None, out_path=None):
    lbl = lbl.squeeze().cpu().tolist()
    xyz = xyz.T.cpu()
    if xyz.shape[0] == 3:
        xyz = xyz.T
    xyz = xyz/8
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


def limit_dataloader(val_dataloader, image_names, img2scene_lut, new_infos, lut_split=None, split_name=None, lut_per_split=None):
    orig_infos = val_dataloader.dataset.imagepoint_dataset.nusc_infos

    # image_names = [
    #     'n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151266154799',
    #     'n015-2018-10-02-11-23-23+0800__CAM_FRONT_LEFT__1538450677854844',
    #     'n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151526897405',
    #     'n015-2018-10-02-11-23-23+0800__CAM_BACK_LEFT__1538450677897423',
    #     'n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151682170482',
    #     'n015-2018-10-02-11-23-23+0800__CAM_BACK_RIGHT__1538450691627893'
    #     ]


    if split_name is not None:
        keys = [img_name.split('.')[0] for img_name in image_names]
        CAM_MAPPING = {
            "CAM_FRONT":"front",
            "CAM_FRONT_RIGHT":"front_right",#,"CAM_FRONT_LEFT"],
            "CAM_FRONT_LEFT":"front_left",
            "CAM_BACK":"back",
            "CAM_BACK_RIGHT":"back_right",#, "CAM_BACK_LEFT"],
            "CAM_BACK_LEFT":"back_left",# ["CAM_BACK_RIGHT", "CAM_BACK_LEFT"]
        }

        kept_names, kept_indices = [],[]
        for i,key in enumerate(keys):
            if (lut_split is None) or (lut_split is not None and lut_split[key]==split_name):
                kept_names.append(image_names[i])
                kept_indices.append(i)
        print('\n{split_name}')
        print(len(kept_names))
        print(kept_indices)


        with open(f'/home/vobecant/PhD/ciirc/nusc_{split_name}_retrieval.txt','w') as f:
            for idx,name in zip(kept_indices, kept_names):
                # name = kept_names[i]
                cam_name = name.split('__')[1]
                cam_name = CAM_MAPPING[cam_name]
                query = SPEC2PROMPT[idx][1]
                if lut_per_split is not None:
                    idx_dataset = lut_per_split[name]
                else:
                    idx_dataset = img2scene_lut[name]
                line=f'{idx_dataset};{query};{cam_name}'
                print(line)
                f.write(f'{line}\n')


    val_dataloader.dataset.imagepoint_dataset.nusc_infos = [
        new_infos[img2scene_lut[img_name.split('.')[0]]] for img_name in image_names
    ]
    
    _infos = val_dataloader.dataset.imagepoint_dataset.nusc_infos
    image_paths = []
    for info in _infos:
        image_paths_cur = []
        for cam, val in info['cams'].items():
            image_paths_cur.append(val['data_path'])
        image_paths.append(image_paths_cur)

    for name, info in zip(image_names, _infos):
        ok = False
        for cam_name, cam_info in info['cams'].items(): 
            cur_name = os.path.split(cam_info['data_path'])[-1].split('.')[0]
            if name==cur_name: ok = True
        if not ok: print(name)

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

    specs = [t[0] for t in SPEC2PROMPT]
    text_queries = [t[1] for t in SPEC2PROMPT]
    eval_splits=args.eval_splits

    new_infos, lut, lut_split, infos_per_split, lut_per_split = get_img2scene_lut()

    specs_filtered, text_queries_filtered = [],[]
    for spec, query in zip(specs, text_queries):
        split = lut_split[spec]
        if split in eval_splits:
            specs_filtered.append(spec)
            text_queries_filtered.append(query)

    specs = specs_filtered
    text_queries = text_queries_filtered
    
    # train_dataloader, _ = get_dataloader(cfg, retrieval=True, no_nusc=True, merged=False)
    merged_dataloader, val_dataloader = get_dataloader(cfg, retrieval=True, no_nusc=True, merged=True)
    # test_dataloader = get_dataloader(cfg, retrieval=True, no_nusc=True, merged=False, return_test=True)
    # limit_dataloader(test_dataloader, specs, lut, new_infos, lut_split, split_name = 'test', lut_per_split=lut_per_split['test'])
    # val_dataloader, image_paths = limit_dataloader(val_dataloader, specs, lut, new_infos, lut_split, split_name = 'val', lut_per_split=lut_per_split['val'])
    val_dataloader = merged_dataloader


    val_dataloader, image_paths = limit_dataloader(val_dataloader, specs, lut, new_infos)
    new_infos_loader = val_dataloader.dataset.imagepoint_dataset.nusc_infos
    splits = [lut_split[info['cams']['CAM_FRONT']['data_path'].split('/')[-1].split('.')[0]] for info in val_dataloader.dataset.imagepoint_dataset.nusc_infos]


    # limit splits
    infos = [info for info,split in zip(val_dataloader.dataset.imagepoint_dataset.nusc_infos,splits) if split in eval_splits]

    def get_image_names(info):
        names = []
        for cam_name, cam_info in info['cams'].items():
            name = cam_info['data_path'].split('/')[-1].split('.')[0]
            names.append(name)
        return names

    image_names = [
        get_image_names(info) for info in infos
    ]

    # missing_names = [
    #     "n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295261162404",
    #                 "n015-2018-10-02-11-23-23+0800__CAM_FRONT_LEFT__1538450677854844",
    #                 "n008-2018-08-01-16-03-27-0400__CAM_FRONT_LEFT__1533154256404799",
    #                 "n008-2018-08-01-16-03-27-0400__CAM_FRONT_LEFT__1533154275504799",
    #                 "n008-2018-08-31-12-15-24-0400__CAM_BACK_LEFT__1535732735447405",
    #                 "n008-2018-08-31-12-15-24-0400__CAM_FRONT_RIGHT__1535732234870482",
    #                 "n015-2018-10-02-11-23-23+0800__CAM_BACK_LEFT__1538450677897423",
    #                 "n015-2018-10-02-11-23-23+0800__CAM_BACK_RIGHT__1538450691627893"
    #             ]

    # for name in missing_names:
    #     for img_names in image_names:
    #         if name in img_names:
    #             for n in img_names:
    #                 cam_name = n.split('__')[1]
    #                 path = os.path.join('/home/vobecant/PhD/TPVFormer-OpenSet/data/nuscenes/samples',cam_name,f'{n}.jpg')
    #                 print(f'"{path}",')

    all_ok = True
    missing=[]
    for ii,names in enumerate(image_names):
        spec = specs[ii]
        target_path = os.path.join(BENCHMARK_TGT_DIR, f'{spec}__retrieval.npy')
        ok = False
        for name in names:
            if spec in name:
                ok = True
        all_ok = all_ok and ok
        if not ok:
            missing.append(spec)
            print(spec, target_path)
    print('missing:',missing)
    assert all_ok


    nusc = val_dataloader.dataset.imagepoint_dataset.nusc
    data_path = args.nusc_dir

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # for info in new_infos_loader:
    #     lidar_path = info['lidar_path']
    #     points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
    #     lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
    #     lidarseg_labels_filename = os.path.join(data_path, nusc.get('lidarseg', lidar_sd_token)['filename'])
    #     points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
    #     panoptic_labels_filename = os.path.join(data_path, nusc.get('panoptic', lidar_sd_token)['filename'])
    #     panoptic_labels = load_bin_file(panoptic_labels_filename, type='panoptic')
    #     semantic, instance = panoptic2semAndInst(panoptic_labels)
    #     targets = (points_label == 18).astype(np.uint8)
    #     savefile = os.path.join(args.save_dir, f'{lidar_sd_token}.npy')
    #     np.save(savefile, targets)
    #     # nusc.render_sample(info['token'],
    #     #                    show_panoptic=True,
    #     #                    show_lidarseg=True,
    #     #                    filter_lidarseg_labels=[18]
    #     #                    )

    #     print(lidar_sd_token, savefile)

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = load_network(cfg, cpu=args.cpu)

    idx = None
    class_mapping_clip, text_features, colors = get_clip_utils(args, val_dataloader)

    query_text = text_queries
    text_features = prepare_text_embeddings(query_text, cpu=args.cpu)
    # specs = image_names

    results = get_features(val_dataloader, idx, model, args.save_dir, class_mapping_clip, text_features, colors,
                                                                                            # name=set_name,
                           image_paths=image_paths, cpu=args.cpu, specs=specs, query_text=query_text, splits=splits)

    from datetime import datetime
    currentDateAndTime = datetime.now()
    timestamp = currentDateAndTime.strftime("%H%M%S")

    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    out_path = f'./results/results_{timestamp}.txt'
    with open(out_path, 'w') as f:
        for res in results:
            f.write(f'{res}\n')
    print(f'Results written to {out_path}')

    exit(0)

    for spec, set_name in SPEC2PROMPT.items():
        text_features = prepare_text_embeddings(set_name, cpu=args.cpu)
        target_path = os.path.join(BENCHMARK_TGT_DIR, f'{spec}__retrieval.npy')
        targets = np.load(target_path)
        predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
        predict_fts_vox_dino, predict_fts_pts_dino, val_vox_label_cls_val, tgt_shape = get_features(val_dataloader, idx,
                                                                                                    model, args.save_dir,
                                                                                                    class_mapping_clip,
                                                                                                    text_features, colors,
                                                                                                    name=set_name,
                                                                                                    image_paths=image_paths,
                                                                                                    cpu=args.cpu,
                                                                                                    targets=targets)
                                                                                        
        print(f'predict_labels_vox_occupancy.shape: {predict_labels_vox_occupancy.shape}')
        print(f'predict_labels_pts.shape: {predict_labels_pts.shape}')
        print(f'predict_fts_vox.shape: {predict_fts_vox.shape}')
        print(f'predict_fts_pts.shape: {predict_fts_pts.shape}')

        # occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_occupancy.argmax(1) == 1))

        # # predict features at those positions
        # tgt_shape = (predict_fts_vox.shape[0], predict_fts_vox.shape[2], predict_fts_vox.shape[3], predict_fts_vox.shape[4])
        # outputs_vox_clip_all = torch.ones(tgt_shape, device='cpu' if args.cpu else 'cuda').long() * EMPTY_SEMANTIC_LABEL
        #                                     occupied_voxels_loc[3]].unsqueeze(0)

        # # assign labels
        # _outputs_vox_clip_predOcc = assign_clip_labels(
        #     args, class_mapping_clip, None, None, predicted_features_occupied_vox,
        #     text_features, None, None, assignment_only=True)
        # outputs_vox_clip_all[occupied_voxels_loc[0], occupied_voxels_loc[1], occupied_voxels_loc[2],
        #                     occupied_voxels_loc[3]] = _outputs_vox_clip_predOcc

        # xyz_pred = torch.stack(torch.where(outputs_vox_clip_all.cpu() != EMPTY_SEMANTIC_LABEL))[1:]
        # labels_pred = _outputs_vox_clip_predOcc

        # tgt_dir = args.save_dir
        # if tgt_dir is not None:
        #     if not os.path.exists(tgt_dir):
        #         os.makedirs(tgt_dir)
        #     out_path = os.path.join(tgt_dir, f'{idx}.txt')
        #     res2txt(xyz_pred, labels_pred, colors, out_path=out_path)

        # if args.show:
        #     tgt_loc = torch.where(
        #         torch.bitwise_and(
        #             val_vox_label_cls_val != IGNORE_LABEL,
        #             val_vox_label_cls_val != EMPTY_SEMANTIC_LABEL
        #         )
        #     )
        #     xyz_tgt = torch.stack(
        #         tgt_loc
        #     )[1:]

        #     labels_tgt = val_vox_label_cls_val[tgt_loc]

        #     fig = plt.figure()
        #     show3d(xyz_pred, fig, 1, 2, 1, labels=labels_pred)
        #     show3d(xyz_tgt, fig, 1, 2, 2, labels=labels_tgt)
        #     plt.show()

        print('DONE')
