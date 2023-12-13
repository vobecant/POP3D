import argparse
import os.path
import pickle

import numpy as np
import torch
import clip
# from matplotlib import pyplot as plt
from tqdm import tqdm

from dataloader.dataset import get_nuScenes_label_name
from eval_maskclip import assign_clip_labels
from utils.load_save_util import revise_ckpt_linear_probe
from visualization.vis_frame import show3d
try:
    from mmcv import Config
except:
    from mmengine.config import Config

from spec2prompt import SPEC2PROMPT
from sklearn.metrics import average_precision_score

from prettytable import PrettyTable
from collections import defaultdict


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
    test_dataloader_config = cfg.test_data_loader
    grid_size = cfg.grid_size

    from builder import data_builder
    out = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            test_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=False,
            scale_rate=cfg.get('scale_rate', 1),
            num_workers=0,
            dataset_type='ImagePoint_NuScenes_withFeatures',
            retrieval=retrieval,
            no_nusc=no_nusc,
            merged=merged,
            return_test=return_test,
            no_aug=True
        )
    if return_test:
        return out
    else:
        train_dataset_loader, val_dataset_loader, *_ = out
    return train_dataset_loader, val_dataset_loader


def generate_cam2token(split_name):

    with open(f'./data/nuscenes_infos_{split_name}.pkl','rb') as f:
        infos = pickle.load(f)

    token2cams = {}
    for info in infos:
        token = info['token']
        cam_paths = []
        for cam_name, cam_info in info['cams'].items():
            cam_specif = cam_info['data_path'].split(os.path.sep)[-1].split('.')[0]
            cam_paths.append(cam_specif)
        token2cams[token] = cam_paths
    
    cam2token = {}
    for tok, cams in token2cams.items():
        for cam in cams:
            cam2token = tok

    # save
    cam2token_path = f"./nuscenes_cam2token_{split_name}.pkl"
    with open(cam2token_path,'wb') as f:
        pickle.dump(cam2token, f)

    # return
    return cam2token



def get_img2scene_lut():
    infos = []
    lut = {}
    token2scene_lut = {}
    token2split = {}
    lut_split = {}

    splits = ['train', 'val', 'test']
    lut_per_split = {split:{} for split in splits}
    infos_per_split = {split:{} for split in splits}
    spec2token_per_split = {split:{} for split in splits}

    idx = 0
    for split in splits:
        idx_split = 0
        cam2token_path = f"./data/nuscenes_cam2token_{split}.pkl"
        if os.path.exists(cam2token_path):
            with open(cam2token_path, 'rb') as f:
                cam2token_cur=pickle.load(f)
        else:
            cam2token_cur = generate_cam2token(split)

        with open(f'./data/nuscenes_infos_{split}.pkl', 'rb') as f:
            cur_infos = pickle.load(f)
            if isinstance(cur_infos, dict):
                cur_infos = cur_infos['infos']
            infos_per_split[split] = cur_infos
        cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for _, info in enumerate(cur_infos):
            infos.append(info)
            token = info['token']
            token2scene_lut[token] = idx
            token2split[token] = split
            for cam_name in cam_names:
                img_path = info['cams'][cam_name]['data_path']
                img_name = os.path.split(img_path)[-1].split('.')[0]
                lut[img_name] = idx
                lut_per_split[split][img_name] = idx_split
                lut_split[img_name] = split
                spec2token_per_split[split][img_name] = token
            idx += 1
            idx_split += 1

    return infos, lut, token2scene_lut, lut_split, infos_per_split, lut_per_split, spec2token_per_split, token2split



@torch.no_grad()
def get_features(dataloader, benchmark_tgt_dir, index, model, save_dir, class_mapping_clip, text_features_all, name='',
                 image_paths=None, cpu=False, targets=None, specs=None, query_text=None, splits=None, tokens_list=None):
    
    results = []
    for i, loaded_data in enumerate(tqdm(dataloader)):
        if index is not None and i != index: continue

        text_features = text_features_all[i].unsqueeze(0)
        spec=None
        if specs is not None:
            spec = specs[i]
            target_path = os.path.join(benchmark_tgt_dir, 'annotations', f'{spec}__retrieval.npy')
            try:
                targets = np.load(target_path)
            except:
                print(f'Problem with loading {target_path}!')
                continue

        split = None if splits is None else splits[i]

        imgs, img_metas, _, val_grid, _, _, _, val_pt_fts, *_ = loaded_data

        try:
            matched_points = _[1][0]
        except:
            matched_points = None

        imgs = imgs.to('cpu' if cpu else 'cuda')
        val_grid_float = val_grid.to(torch.float32).to('cpu' if cpu else 'cuda')

        xyz_pc = val_grid_float[0].T.cpu()
        npts = xyz_pc.shape[1]
        nlbl = len(targets)
        if npts!=nlbl:
            print(f'{spec}: {npts} points and {nlbl} targets!')
            continue

        predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
        predict_fts_vox_dino, predict_fts_pts_dino = model(img=imgs, img_metas=img_metas,
                                                           points=val_grid_float,
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
            visible_tgt_labels = torch.tensor(targets).squeeze()[matched_points]
            mAP_visible_tgt = average_precision_score(visible_tgt_labels.squeeze().cpu().numpy(),
                                                    tgt_features_pts_sim.squeeze().cpu().numpy()
                                                    )

        else:
            mAP_visible_tgt = 'N/A'    
        
        res = f'{split};{spec};{query_text[i]};{mAP};{mAP_visible};N/A;{mAP_visible_tgt}'
        results.append(res)

        spec += f' [{query_text[i]}]' if query_text is not None else ''
        print(spec, res)

        torch.cuda.empty_cache()
        del predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts, \
            predict_fts_vox_dino, predict_fts_pts_dino, occupied_voxels_loc, outputs_vox_clip_all, predicted_features_occupied_vox, xyz_pred, labels_pred, res

    return results


def get_clip_utils(args, dataloader):
    class_mapping_clip = torch.tensor([0]).to('cpu' if args.cpu else 'cuda')
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
    return class_mapping_clip, text_features


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config',
                        default='config/pop3d_maskclip_12ep.py')
    parser.add_argument('--resume-from', type=str,
                        default='./pretrained/pop3d_weights.pth')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--text-embeddings-path', type=str, default=None)
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--scale', default=None, type=int)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--normalized-cosine', action='store_true')
    parser.add_argument('--nusc-dir', type=str, default='./data/nuscenes')
    parser.add_argument('--eval-splits', type=str, nargs="*",  default=['train','val','test'])
    parser.add_argument('--benchmark-tgt-dir',default='./data/retrieval_benchmark', type=str, help='Path to the folder with benchmark files.')
    args = parser.parse_args()
    print(args)

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


def limit_dataloader(dataloader, token2scene_lut, new_infos, splits, lut_split=None, lut_per_split=None):

    tokens = []
    text_queries = []

    for split in splits:
        split_file = f'./data/retrieval_benchmark/retrieval_anns_{split}.csv'

        for line in open(split_file,'r').readlines():
            line = line.strip()
            token, split, ann_path, pts_path, query = line.split(';')
            tokens.append(token)
            text_queries.append(query)


    dataloader.dataset.imagepoint_dataset.nusc_infos = [
        new_infos[token2scene_lut[token]] for token in tokens
    ]
    
    _infos = dataloader.dataset.imagepoint_dataset.nusc_infos
    image_paths = []
    for info in _infos:
        image_paths_cur = []
        for cam, val in info['cams'].items():
            image_paths_cur.append(val['data_path'])
        image_paths.append(image_paths_cur)

    return dataloader, image_paths, tokens, text_queries


if __name__ == '__main__':
    args = get_args()
    cfg = Config.fromfile(args.py_config)
    cfg.resume_from = args.resume_from

    if args.scale is not None:
        cfg.model['tpv_aggregator']['scale_h'] = cfg.model['tpv_aggregator']['scale_w'] = cfg.model['tpv_aggregator'][
            'scale_z'] = args.scale

    eval_splits=args.eval_splits

    new_infos, lut, token2scene_lut, lut_split, infos_per_split, lut_per_split, spec2token_per_split, token2split = get_img2scene_lut()
    
    merged_dataloader, val_dataloader = get_dataloader(cfg, retrieval=True, no_nusc=True, merged=True)
    val_dataloader = merged_dataloader


    val_dataloader, image_paths, tokens, text_queries = limit_dataloader(val_dataloader, token2scene_lut, new_infos, eval_splits)
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

    nusc = val_dataloader.dataset.imagepoint_dataset.nusc
    data_path = args.nusc_dir

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = load_network(cfg, cpu=args.cpu)

    idx = None
    class_mapping_clip, text_features = get_clip_utils(args, val_dataloader)

    query_text = text_queries
    text_features = prepare_text_embeddings(query_text, cpu=args.cpu)

    for i in range(len(infos)):
        info = infos[i]
        txt = text_features[i]
        img = image_paths[i]
        print(img)
        tok_list = tokens[i]
        tok_info = info['token']
    
    results = get_features(val_dataloader,args.benchmark_tgt_dir, idx, model, args.save_dir, class_mapping_clip, text_features,
                           image_paths=image_paths, cpu=args.cpu, specs=tokens, query_text=query_text, splits=splits)

    from datetime import datetime
    currentDateAndTime = datetime.now()
    timestamp = currentDateAndTime.strftime("%d%m%Y_%H%M%S")

    if not os.path.exists('./results'):
        os.makedirs('./results')

    to_write = []
    splits = args.eval_splits + ['valtest']
    for split in splits:
        pop3d, pop3d_visible, maskclip_visible = [], [], []
        for res in results:
            _, token, query, mAP_pop3d, mAP_visible_pop3d, _, mAP_visible_maskclip = res.split(';')
            if not token2split[token] in split:
                continue
            pop3d.append(float(mAP_pop3d))
            pop3d_visible.append(float(mAP_visible_pop3d))
            maskclip_visible.append(float(mAP_visible_maskclip))

        count = len(pop3d)
        pop3d_mAP = np.mean(pop3d) * 100
        pop3d_visible_mAP = np.mean(pop3d_visible) * 100
        maskclip_visible_mAP = np.mean(maskclip_visible) * 100

        pop3d_mAP = "{:.1f}".format(pop3d_mAP)
        pop3d_visible_mAP = "{:.1f}".format(pop3d_visible_mAP)
        maskclip_visible_mAP = "{:.1f}".format(maskclip_visible_mAP)

        print()
        x = PrettyTable()
        x.field_names=['method', "mAP", "mAP visible"]
        x.title = f'{split} ({count} samples)'
        x.add_rows([
            ["POP3D", pop3d_mAP, pop3d_visible_mAP],
            ["MaskCLIP", "N/A", maskclip_visible_mAP]
                    ])
        print(x)
        print()
        to_write.append(str(x))

    
    out_path = f'./results/results_{timestamp}.txt'
    with open(out_path, 'w') as f:
        for res in results:
            f.write(f'{res}\n')
    print(f'Results written to {out_path}')

    out_path = f'./results/results_tables_{timestamp}.txt'
    with open(out_path, 'w') as f:
        for res in to_write:
            f.write(f'{res}\n')
    print(f'Results written to {out_path}')