import argparse
import os
import os.path as osp
import pickle
import socket
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt

try:
    from mmcv import Config
except:
    from mmengine.config import Config
try:
    from mmseg.utils import get_root_logger
except:
    from mmengine.logging import MMLogger
from torch.nn import MSELoss

from builder import loss_builder
from dataloader.dataset import get_nuScenes_label_name
from train import IGNORE_LABEL_SEMANTIC
from utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_linear_probe
from utils.metric_util import MeanIoU
from utils.prompt_extractor import PromptExtractor
from visualization.training import show3d

VIS = True  # False
EMPTY_SEMANTIC_LABEL = 17
IGNORE_LABEL = 255
OCCUPIED_AGNOSTIC_LABEL = 1

warnings.filterwarnings("ignore")

NCLS2CONFIG = {
    7: 'config/label_mapping/nuscenes-7cls.yaml',
    29: 'config/label_mapping/nuscenes-29cls.yaml',
}

ROAD_SEM_IDX = 11


def pass_print(*args, **kwargs):
    pass


def assign_labels_clip(predicted_features, text_features, class_offset, maskclip=False, class_mapping_clip=None,
                       ignore_label=None, normalized_cosine=False):
    if maskclip:
        if ignore_label is not None and class_mapping_clip is not None:
            nonignored_indices = class_mapping_clip != ignore_label
            text_features = text_features[nonignored_indices]
            class_mapping_clip_nonignore = class_mapping_clip[nonignored_indices]
        else:
            class_mapping_clip_nonignore = class_mapping_clip
        if normalized_cosine:
            predicted_features_norm = F.normalize(predicted_features, dim=-1)
            text_features_norm = F.normalize(text_features, dim=-1)[None]

            logits = torch.einsum('bnc,bkc->bkn', predicted_features_norm, text_features_norm)
        else:
            logits = F.conv1d(predicted_features.permute(0, 2, 1), text_features[:, :, None])
        class_preds = logits.argmax(1)
        if class_mapping_clip is None:
            class_preds += class_offset
        else:
            class_preds = class_mapping_clip_nonignore[class_preds]
            logits = max_logits_per_class(logits, class_mapping_clip_nonignore)
    else:
        # L2-normalize predicted features
        predicted_features /= predicted_features.norm(dim=-1, keepdim=True)
        # compute cosine similarities
        logits = predicted_features @ text_features.T
        class_preds = (logits).argmax(-1) + class_offset
    return class_preds, logits


def max_logits_per_class(logits, class_mapping):
    unique, counts = class_mapping.unique(return_counts=True)
    if unique.min() != 0:
        unique -= unique.min()
    max_labels = counts.max()
    n_samples = logits.shape[-1]
    n_subclasses = logits.shape[1]
    mapping = []
    for unq, cnt in zip(unique, counts):
        for c in range(cnt):
            mapping.append([unq, c])
    mapping = torch.tensor(mapping)
    logits_dense = -torch.ones((n_samples, unique.max() + 1, max_labels), device=logits.device) * torch.inf

    for subclass_idx in range(n_subclasses):
        subclass_logits = logits[0, subclass_idx]
        subclass_mapping_cls, subclass_mapping_idx = mapping[subclass_idx]
        logits_dense[:, subclass_mapping_cls, subclass_mapping_idx] = subclass_logits

    max_logits = logits_dense.max(-1)[0].T.unsqueeze(0)
    return max_logits


def next_free_port(port, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return str(port)
        except OSError:
            print(f'Port {port} is occupied.')
            port += 1
    raise IOError('no free ports')


def semantic2agnostic(val_pt_labs, mapping):
    val_pt_labs_agnostic = torch.zeros_like(val_pt_labs)
    val_pt_labs_agnostic[val_pt_labs < EMPTY_SEMANTIC_LABEL] = 1
    return val_pt_labs_agnostic


@torch.no_grad()
def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    print(f'socket.gethostname(): {socket.gethostname()}')

    fill_label_gt = None if args.num_classes is None else args.num_classes + 1
    if fill_label_gt is not None:
        EMPTY_SEMANTIC_LABEL = fill_label_gt
    else:
        EMPTY_SEMANTIC_LABEL = 17

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # modify the config with passed arguments
    if args.dec_layers_occupancy is not None:
        cfg.model['tpv_aggregator']['dec_layers_occupancy'] = args.dec_layers_occupancy
    if args.hidden_dims is not None:
        cfg.model['tpv_aggregator']['hidden_dims'] = args.hidden_dims
    if args.dec_layers_features is not None:
        cfg.model['tpv_aggregator']['dec_layers_features'] = args.dec_layers_features
    if args.hidden_dims_ft is not None:
        cfg.model['tpv_aggregator']['hidden_dims_ft'] = args.hidden_dims_ft

    free_label = cfg.dataset_params.fill_label
    agnostic = False

    if args.class_weights_path is None:
        try:
            args.class_weights_path = cfg.class_weights_path
        except:
            pass

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader
    test_dataloader_config = cfg.test_data_loader

    max_num_epochs = cfg.max_epochs
    grid_size = cfg.grid_size

    # init DDP
    if local_rank == -1:
        distributed = False
        global_rank = 0
    else:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        # port = next_free_port(int(port))
        print(f'Initial port: {port}')
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        global_rank = rank * gpus + local_rank
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=global_rank
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

    if global_rank == 0 and not args.no_wandb:
        config = vars(args)
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="TPVFormer-Open",
            # set the wandb run name
            name=args.name,
            # track hyperparameters and run metadata
            config=config
        )

    # configure logger
    if not distributed or dist.get_rank() == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    try:
        logger = get_root_logger(log_file=log_file, log_level='INFO')
    except:
        logger = MMLogger.get_current_instance()
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder

    my_model = model_builder.build(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print(f'done ddp model (local_rank {local_rank}), torch.cuda.current_device(): {torch.cuda.current_device()}')

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label] if len(unique_label) > 2 else ['empty',
                                                                                                     'occupied']

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            test_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1),
            # eval_mode=True,
            num_workers=0 if (args.show or args.debug) else 1,
            class_agnostic=False,
            fill_label_gt=fill_label_gt
        )

    # setup configuration and loss for feature learning
    # if the feature learning is ON
    try:
        feature_learning = cfg.feature_learning
    except:
        feature_learning = False

    if feature_learning:
        clip_features = dataset_config['features_type'] == 'clip'
    else:
        clip_features = False

    if clip_features:
        class_mapping_clip = None
        if args.num_classes is None:
            unique_label_clip = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            SemKITTI_label_name_clip = get_nuScenes_label_name("./config/label_mapping/nuscenes.yaml")
            unique_label_str_clip = [SemKITTI_label_name_clip[x] for x in unique_label_clip]
        else:
            unique_label_clip = [i for i in range(1, args.num_classes + 1)]
            SemKITTI_label_name_clip = get_nuScenes_label_name(NCLS2CONFIG[args.num_classes])
            unique_label_str_clip = [SemKITTI_label_name_clip[x] for x in unique_label_clip]
        if args.text_embeddings_path is not None and os.path.exists(args.text_embeddings_path):
            text_features = torch.load(args.text_embeddings_path, map_location='cpu')
            if type(text_features) in [tuple, list]:
                text_features, class_mapping_clip = text_features
                learning_map_gt = train_dataset_loader.dataset.imagepoint_dataset.learning_map_gt
                class_mapping_clip = torch.tensor([learning_map_gt[c] for c in class_mapping_clip]).cuda()
            if 'odise' in args.text_embeddings_path.lower():
                embedding_dim = 256
            else:
                embedding_dim = 512
            if text_features.shape[0] == embedding_dim:
                text_features = text_features.T
            text_features = text_features.float().cuda()

    voxel_feature_loss = None
    if feature_learning:
        try:
            voxel_feature_loss_name = cfg.voxel_feature_loss.lower()
        except:
            voxel_feature_loss_name = None
        if voxel_feature_loss_name is not None:
            if voxel_feature_loss_name in ['l2', 'mse']:
                voxel_feature_loss = MSELoss(reduction='mean')

    class_weights = None
    if args.class_weights_path is not None and os.path.exists(args.class_weights_path):
        with open(args.class_weights_path, 'rb') as f:
            class_weights = pickle.load(f)

    if class_weights is not None:
        class_weights = class_weights.float().cuda()

    # get optimizer, loss, scheduler
    loss_func, lovasz_softmax = loss_builder.build(ignore_label=ignore_label, weight=class_weights)
    loss_func_noW, _ = loss_builder.build(ignore_label=-100)

    CalMeanIou_vox_occupied_agnostic_unique = MeanIoU([0, 1], IGNORE_LABEL, ['empty', 'occupied'], 'vox_occ_agn_unq')
    CalMeanIou_pts = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts')
    if clip_features:
        CalMeanIou_pts_clip = MeanIoU(unique_label_clip, ignore_label, unique_label_str_clip, 'pts_clip')
        CalMeanIou_pts_clip_ignore0 = MeanIoU(unique_label_clip, 0, unique_label_str_clip, 'pts_clip')
        CalMeanIou_pts_clip_wRoad = MeanIoU(unique_label_clip, 0, unique_label_str_clip, 'pts_clip_wRoad')
        CalMeanIou_pts_clip_visible = MeanIoU(unique_label_clip, IGNORE_LABEL_SEMANTIC, unique_label_str_clip,
                                              'pts_clip_visible')
        CalMeanIou_pts_clip_visible_gt = MeanIoU(unique_label_clip, ignore_label, unique_label_str_clip,
                                                 'pts_clip_visible_gt')
        CalMeanIou_vox_clip_all = MeanIoU(unique_label_clip + [EMPTY_SEMANTIC_LABEL], IGNORE_LABEL,
                                          unique_label_str_clip + ['empty'],
                                          'vox_clip_all', extra_classes=0)
        CalMeanIou_vox_clip_occupied = MeanIoU(unique_label_clip, ignore_label,
                                               unique_label_str_clip, 'vox_clip_occupied', extra_classes=0,
                                               extra_classes_pred=1)
        CalMeanIou_pts_clip_gt = MeanIoU(unique_label_clip, ignore_label, unique_label_str_clip, 'vox_clip_occupied')

    CalMeanIou_vox_all_agnostic = MeanIoU([0, 1], ignore_label=IGNORE_LABEL, label_str=['empty', 'occupied'],
                                          name='vox_agn_all')
    CalMeanIou_pts_agnostic = MeanIoU([0, 1], ignore_label=IGNORE_LABEL, label_str=['empty', 'occupied'],
                                      name='pts_agn')

    # resume and load
    epoch = 0
    best_val_miou_pts, best_val_miou_vox = 0, 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    print('work dir: ', args.work_dir)

    assert args.debug or (cfg.resume_from and osp.isfile(cfg.resume_from))

    if cfg.resume_from and osp.isfile(cfg.resume_from):
        print('resume from: ', cfg.resume_from, f'(local_rank {local_rank})')
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        if not distributed or args.no_dist:
            revise_fnc = revise_ckpt_linear_probe
        else:
            revise_fnc = revise_ckpt
        print(my_model.load_state_dict(revise_fnc(ckpt['state_dict'], ddp=distributed and not args.no_dist),
                                       strict=True),f'(local_rank {local_rank})')
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch} (local_rank {local_rank})')
    elif cfg.resume_from and not osp.isfile(cfg.resume_from):
        print(f'Given path {cfg.resume_from} but it does not exist!')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if args.no_dist:
            print(f'Loading state dict from {cfg.load_from}')
            print(my_model.img_backbone.load_state_dict(state_dict, strict=False))

        else:
            state_dict = revise_ckpt(state_dict, add_image_bbn_name='small' in args.py_config)
            print(f'Loading state dict from {cfg.load_from}')
            try:
                print(my_model.load_state_dict(state_dict, strict=False))
            except:
                state_dict = revise_ckpt_2(state_dict)
                print(my_model.load_state_dict(state_dict, strict=False))

    # printing frequency
    print_freq = cfg.print_freq
    try:
        print_freq_wandb_train = cfg.print_freq_wandb_train
    except:
        print_freq_wandb_train = cfg.print_freq
    try:
        print_freq_wandb_val = cfg.print_freq_wandb_val
    except:
        print_freq_wandb_val = cfg.print_freq

    print(f'Start training! (local_rank {local_rank})')
    val_vis_iter = 0

    # eval
    my_model.eval()
    val_loss_list = []
    val_loss_dict = {}
    CalMeanIou_pts.reset()
    CalMeanIou_vox_occupied_agnostic_unique.reset()
    CalMeanIou_pts_agnostic.reset()
    CalMeanIou_vox_all_agnostic.reset()
    if clip_features:
        CalMeanIou_pts_clip.reset()
        CalMeanIou_pts_clip_ignore0.reset()
        CalMeanIou_pts_clip_visible.reset()
        CalMeanIou_pts_clip_visible_gt.reset()
        CalMeanIou_pts_clip_gt.reset()
        CalMeanIou_vox_clip_all.reset()
        CalMeanIou_vox_clip_occupied.reset()
        CalMeanIou_pts_clip_wRoad.reset()

    data_time_s = time.time()
    time_val_ep_start = time.time()

    sem2agn_mapping = {i: 1 for i in range(EMPTY_SEMANTIC_LABEL)}
    sem2agn_mapping[EMPTY_SEMANTIC_LABEL] = 0

    if args.plot_dir is not None:
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

    used_dataloader = train_dataset_loader if args.val_on_train else val_dataset_loader

    print(f'len(used_dataloader): {len(used_dataloader)}')
    with torch.no_grad():
        for i_iter_val, loaded_data in enumerate(used_dataloader):
            time_it_s = time.time()
            if args.max_iter is not None and i_iter_val >= args.max_iter:
                break
            data_time_e = time.time()
            time_s = time.time()
            imgs, img_metas, val_vox_label_semantic, val_grid, val_pt_labs, val_vox_label_cls, val_grid_fts, val_pt_fts, val_vox_label_cls_val, matching_points, *_ = loaded_data

            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            val_vox_label_cls_val = val_vox_label_cls_val.cuda()

            val_pt_labs = val_pt_labs.cuda()
            val_pt_labs_agnostic = semantic2agnostic(val_pt_labs, sem2agn_mapping)

            val_vox_label_agnostic = semantic2agnostic(val_vox_label_semantic, sem2agn_mapping).cuda()

            val_grid_fts = val_grid_fts.cuda()
            if voxel_feature_loss is not None:
                val_pt_fts = val_pt_fts.cuda()

            predict_labels_vox_agn, predict_labels_pts_agn, predict_fts_vox, predicted_features_pts, *_ = my_model \
                (img=imgs, img_metas=img_metas, points=val_grid_float, features=val_grid_float.clone())
            # get the occupied voxels; for these voxels, get the feature predictions
            occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_agn.argmax(1) == 1))

            n_occ = occupied_voxels_loc.shape[1]

            # predict features at those positions
            outputs_vox_clip_all = torch.ones_like(val_vox_label_cls_val,
                                                   device=imgs.device) * EMPTY_SEMANTIC_LABEL

            if n_occ > 0:
                predicted_features_occupied_vox = predict_fts_vox[occupied_voxels_loc[0], :,
                                                  occupied_voxels_loc[1], occupied_voxels_loc[2],
                                                  occupied_voxels_loc[3]].unsqueeze(0)

                # get the occupied voxels; for these voxels, get the feature predictions
                occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_agn.argmax(1) == 1))

                # assign labels
                _outputs_vox_clip_predOcc = assign_clip_labels(
                    args, class_mapping_clip, None, None, predicted_features_occupied_vox,
                    text_features, None, None, assignment_only=True)
                outputs_vox_clip_all[occupied_voxels_loc[0], occupied_voxels_loc[1], occupied_voxels_loc[2],
                                     occupied_voxels_loc[3]] = _outputs_vox_clip_predOcc

            # evaluate voxels
            if args.show or args.plot_dir is not None:
                xyz_pred = torch.stack(torch.where(outputs_vox_clip_all.cpu() != EMPTY_SEMANTIC_LABEL))[1:]
                xyz_tgt = torch.stack(
                    torch.where(
                        torch.bitwise_and(
                            val_vox_label_cls_val != IGNORE_LABEL,
                            val_vox_label_cls_val != EMPTY_SEMANTIC_LABEL
                        )
                    )
                )[1:]
                fig = plt.figure()
                show3d(xyz_pred, fig, 1, 2, 1)
                show3d(xyz_tgt, fig, 1, 2, 2)
                if args.plot_dir is not None:
                    save_path = os.path.join(args.plot_dir, f'{i_iter_val}pred.jpg')
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f'Saved to {save_path}')
                plt.show()

            # a) all voxels
            CalMeanIou_vox_clip_all._after_step(outputs_vox_clip_all, val_vox_label_cls_val)

            # b) GT-occupied voxels only
            _occupied_idx = torch.bitwise_and(val_vox_label_cls[0] != EMPTY_SEMANTIC_LABEL,
                                              val_vox_label_cls[0] != IGNORE_LABEL_SEMANTIC)
            targets = val_vox_label_semantic[0, _occupied_idx]
            predictions = outputs_vox_clip_all[0, _occupied_idx]
            CalMeanIou_vox_clip_occupied._after_step(predictions, targets.cuda())

            # assign labels at occupied points
            _outputs_pts_clip_gtOcc = assign_clip_labels(
                args, class_mapping_clip, None, None, predicted_features_pts,
                text_features, None, None, assignment_only=True)
            CalMeanIou_pts_clip._after_step(_outputs_pts_clip_gtOcc.squeeze(), val_pt_labs.squeeze())
            CalMeanIou_pts_clip_ignore0._after_step(_outputs_pts_clip_gtOcc.squeeze(),
                                                    val_pt_labs.squeeze())
            CalMeanIou_pts_clip_visible._after_step(_outputs_pts_clip_gtOcc[:, matching_points[0]].squeeze(),
                                                    val_pt_labs[:, matching_points[0]].squeeze())
            # at invisible points, predict drivable surface
            _outputs_pts_clip_gtOcc_wRoad = torch.ones_like(_outputs_pts_clip_gtOcc) * ROAD_SEM_IDX
            _outputs_pts_clip_gtOcc_wRoad[:, matching_points[0]] = _outputs_pts_clip_gtOcc[:, matching_points[0]]
            if i_iter_val == 0:
                print(f'_outputs_pts_clip_gtOcc_wRoad.unique(): {_outputs_pts_clip_gtOcc_wRoad.unique()}, '
                      f'val_pt_labs.unique(): {val_pt_labs.unique()}')
            CalMeanIou_pts_clip_wRoad._after_step(_outputs_pts_clip_gtOcc_wRoad.squeeze(),
                                                  val_pt_labs.squeeze())

            ######################
            # PROJECTED FEATURES #
            ######################
            _outputs_pts_clip_projFt = assign_clip_labels(
                args, class_mapping_clip, None, None, predicted_features_pts,
                text_features, None, None, assignment_only=True)

            CalMeanIou_pts_clip_gt._after_step(_outputs_pts_clip_projFt.squeeze(), val_pt_labs.squeeze())
            CalMeanIou_pts_clip_visible_gt._after_step(_outputs_pts_clip_projFt[:, matching_points[0]].squeeze(),
                                                       val_pt_labs[:, matching_points[0]].squeeze())

            predict_labels_pts_agn = predict_labels_pts_agn.squeeze(-1).squeeze(-1)
            predict_labels_pts_agn = torch.argmax(predict_labels_pts_agn, dim=1)  # bs, n
            predict_labels_pts_agn = predict_labels_pts_agn.detach()
            val_pt_labs = val_pt_labs.squeeze(-1)

            predict_labels_vox_agn = torch.argmax(predict_labels_vox_agn, dim=1)
            predict_labels_vox_agn = predict_labels_vox_agn.detach()

            gt_labels_vox_agnostic = val_vox_label_agnostic
            predict_labels_vox_agnostic = predict_labels_vox_agn
            gt_labels_pts_agnostic = val_pt_labs_agnostic
            predict_labels_pts_agnostic = predict_labels_pts_agn

            for count in range(len(val_grid_int)):
                # points
                CalMeanIou_pts_agnostic._after_step(predict_labels_pts_agnostic[count],
                                                    gt_labels_pts_agnostic[count, :, 0])

                # voxels
                # a) all voxels
                val_vox_label_occupancy_val = val_vox_label_cls_val.clone()
                # occluded areas + semantic ignore labels
                val_vox_label_occupancy_val[val_vox_label_cls_val != IGNORE_LABEL] = 1
                # add back the semantic ignore areas
                val_vox_label_occupancy_val[val_vox_label_cls == IGNORE_LABEL_SEMANTIC] = 1
                val_vox_label_occupancy_val[val_vox_label_cls_val == EMPTY_SEMANTIC_LABEL] = 0
                CalMeanIou_vox_all_agnostic._after_step(predict_labels_vox_agnostic[count].flatten(),
                                                        val_vox_label_occupancy_val[count].flatten())
                # b) occupied only
                _occupied_idx = val_vox_label_agnostic[count] == OCCUPIED_AGNOSTIC_LABEL
                targets = val_vox_label_agnostic[count, _occupied_idx]
                predictions = predict_labels_vox_agnostic[count, _occupied_idx]
                CalMeanIou_vox_occupied_agnostic_unique._after_step(predictions, targets)

            if i_iter_val % print_freq == 0 and (args.no_dist or dist.get_rank() == 0):
                logger.info('[EVAL] Epoch %d Iter %5d' % (epoch, i_iter_val))

            time_e = time.time()
            pass_iter_time = time_e - time_s
            data_iter_time = data_time_e - data_time_s
            if global_rank == 0 and not args.no_wandb:
                wandb.log({"time/val_iter": pass_iter_time, "time/val_data_iter": data_iter_time}, commit=False)
            data_time_s = time.time()
            val_vis_iter += 1

    log2wandb = global_rank == 0 and not args.no_wandb
    val_miou_pts = CalMeanIou_pts._after_epoch(log_wandb=log2wandb, tag="miou_pts/val",
                                               # step=global_iter
                                               step=None
                                               )
    val_miou_vox_occupied_agnostic_unique = CalMeanIou_vox_occupied_agnostic_unique._after_epoch(
        log_wandb=log2wandb, tag="miou_vox_occupied_agnostic_unique/val", step=None
    )
    val_miou_pts_agn = CalMeanIou_pts_agnostic._after_epoch(log_wandb=log2wandb, tag="miou_pts_agn/val", step=None)
    val_miou_vox_all = CalMeanIou_vox_all_agnostic._after_epoch(log_wandb=log2wandb, tag="miou_vox_agn/val", step=None)

    if clip_features:
        val_miou_pts_clip = CalMeanIou_pts_clip._after_epoch(log_wandb=log2wandb, tag="miou_pts_clip/val", step=None)
        val_miou_pts_clip_ignore0 = CalMeanIou_pts_clip_ignore0._after_epoch(log_wandb=log2wandb,
                                                                             tag="miou_pts_clip_ignore0/val", step=None)
        val_miou_pts_clip_visible = CalMeanIou_pts_clip_visible._after_epoch(log_wandb=log2wandb,
                                                                             tag="miou_pts_clip_visible/val", step=None)
        val_miou_pts_clip_gt = CalMeanIou_pts_clip_gt._after_epoch(log_wandb=log2wandb, tag="miou_pts_clip_gt/val",
                                                                   step=None)
        val_miou_vox_clip_all = CalMeanIou_vox_clip_all._after_epoch(log_wandb=log2wandb, tag="miou_vox_clip_all/val",
                                                                     step=None)
        val_miou_vox_clip_occupied = CalMeanIou_vox_clip_occupied._after_epoch(
            log_wandb=log2wandb, tag="miou_vox_clip_occupied/val", step=None)
        val_miou_pts_clip_wRoad = CalMeanIou_pts_clip_wRoad._after_epoch(
            log_wandb=log2wandb, tag="miou_pts_clip_wRoad/val", step=None)

        print(f'val_miou_vox_agn (evaluate voxel agnostic segmentation everywhere): {val_miou_vox_all}')

    time_val_ep_elapsed = time.time() - time_val_ep_start
    # log metrics to wandb
    if global_rank == 0 and not args.no_wandb:
        wandb.log({"miou_pts/val": val_miou_pts, "miou_vox_all/val": val_miou_vox_all}, commit=False)
        wandb.log({"miou_pts_agn/val": val_miou_pts_agn, "miou_vox_agn/val": val_miou_vox_all}, commit=False)
        wandb.log({"loss/val_mean_ep": np.mean(val_loss_list)}, commit=False)
        if clip_features:
            wandb.log({"miou_pts_clip/val": val_miou_pts_clip})
            wandb.log({"miou_pts_clip_visible/val": val_miou_pts_clip_visible})
            wandb.log({"miou_pts_clip_gt/val": val_miou_pts_clip_gt})
            wandb.log({"miou_vox_clip/val": val_miou_vox_clip_all})
        for key, vals in val_loss_dict.items():
            wandb.log({f"loss/val_{key}": torch.mean(torch.tensor(vals))}, commit=False)
        wandb.log({"time/val_epoch": time_val_ep_elapsed})
        

def assign_clip_labels(args, class_mapping_clip, loss_func_noW, lovasz_softmax, outputs_pts_fts, text_features,
                       train_grid_fts, train_vox_label_cls, ignore_label=0, compute_loss=True, assignment_only=False,
                       no_nonignore=False, logits_only=False, cpu=False):
    # points
    outputs_pts_clip, logits_clip = assign_labels_clip(
        outputs_pts_fts.float().to('cpu' if cpu else 'cuda'), text_features, 1, maskclip=True,
        class_mapping_clip=class_mapping_clip, ignore_label=ignore_label, normalized_cosine=args.normalized_cosine)
    if logits_only:
        return logits_clip
    if assignment_only:
        return outputs_pts_clip

    train_grid_fts_int = train_grid_fts.long()
    train_pt_labs_fts_list = []
    for bi in range(train_grid_fts_int.shape[0]):
        train_grid_fts_int_cur = train_grid_fts_int[bi]
        train_pt_labs_fts_cur = train_vox_label_cls[
            bi, train_grid_fts_int_cur[:, 0], train_grid_fts_int_cur[:, 1], train_grid_fts_int_cur[:, 2]
        ]
        train_pt_labs_fts_list.append(train_pt_labs_fts_cur)
    train_pt_labs_fts = torch.cat(train_pt_labs_fts_list).unsqueeze(0).to('cpu' if cpu else 'cuda')
    if no_nonignore:
        non_ignore = torch.ones_like(train_pt_labs_fts, dtype=bool)
    else:
        non_ignore = train_pt_labs_fts != ignore_label
    targets = (train_pt_labs_fts[non_ignore] - 1).unsqueeze(0)
    outputs_pts_clip = outputs_pts_clip[non_ignore].unsqueeze(0).to('cpu' if cpu else 'cuda')

    loss_ce_clip, loss_lovasz_clip = None, None
    if compute_loss:
        logits_clip_non_ignore = logits_clip.permute(0, 2, 1)[non_ignore].unsqueeze(0).permute(0, 2, 1)
        loss_ce_clip = loss_func_noW(logits_clip_non_ignore.detach(), targets.detach())
        loss_lovasz_clip = lovasz_softmax(
            torch.nn.functional.softmax(logits_clip_non_ignore.unsqueeze(-1).unsqueeze(-1), dim=1).detach(),
            targets.unsqueeze(-1).detach())
    else:
        train_grid_fts_int = train_grid_fts_int[non_ignore].unsqueeze(0)
    return loss_ce_clip, loss_lovasz_clip, non_ignore, outputs_pts_clip, targets, train_grid_fts_int


def get_agnostic_labels(args, predict_labels_pts, predict_labels_vox, val_vox_labs, val_pt_labs):
    if not args.agnostic:
        # 1) voxel grid
        #   A) GT
        gt_labels_vox_agnostic = torch.zeros_like(val_vox_labs)
        gt_labels_vox_agnostic_occ = torch.where(val_vox_labs < EMPTY_SEMANTIC_LABEL)
        gt_labels_vox_agnostic[gt_labels_vox_agnostic_occ] = 1
        #   B) predictions
        predict_labels_vox_agnostic = torch.zeros_like(predict_labels_vox)
        predict_labels_vox_agnostic_occ = torch.where(predict_labels_vox < EMPTY_SEMANTIC_LABEL)
        predict_labels_vox_agnostic[predict_labels_vox_agnostic_occ] = 1

        # 2) points
        #   A) GT
        gt_labels_pts_agnostic = torch.zeros_like(val_pt_labs)
        gt_labels_pts_agnostic_occ = torch.where(val_pt_labs < EMPTY_SEMANTIC_LABEL)
        gt_labels_pts_agnostic[gt_labels_pts_agnostic_occ] = 1
        #   B) predictions
        predict_labels_pts_agnostic = torch.zeros_like(predict_labels_pts)
        predict_labels_pts_agnostic_occ = torch.where(predict_labels_pts < EMPTY_SEMANTIC_LABEL)
        predict_labels_pts_agnostic[predict_labels_pts_agnostic_occ] = 1
    else:
        # in this setup, we do have class-agnostic predictions and class-agnostic targets
        # this means that we do not need to do anything about the labels

        assert len(val_vox_labs.unique()) <= 2 and len(val_pt_labs.unique()) == 1 and len(
            predict_labels_vox.unique()) <= 2 and len(predict_labels_pts) <= 2

        # 1) voxels
        gt_labels_vox_agnostic = val_vox_labs.detach().cpu()
        predict_labels_vox_agnostic = predict_labels_vox.detach().cpu()
        # 2) points
        gt_labels_pts_agnostic = val_pt_labs.detach().cpu()
        predict_labels_pts_agnostic = predict_labels_pts.detach().cpu()

    return gt_labels_vox_agnostic, predict_labels_vox_agnostic, gt_labels_pts_agnostic, predict_labels_pts_agnostic


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--eval-at-start', action='store_true')
    parser.add_argument('--no-dist', action='store_true', help='Do not use distributed setup.')
    parser.add_argument('--compute-weights', action='store_true')
    parser.add_argument('--compute-upperbound-clip', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--agnostic', action='store_true')
    parser.add_argument('--class-weights-path',
                        # default='./class_weights.pkl',
                        default=None,
                        type=str)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--ft-loss-weight', type=float, default=1., help='Weight of the feature loss.')
    parser.add_argument('--maskclip', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--text-embeddings-path', default=None, type=str)
    parser.add_argument('--max-iter', default=None, type=int)
    parser.add_argument('--projected-features', action='store_true')
    parser.add_argument('--plot-dir', default=None, type=str)
    # number of layers in the decoders
    parser.add_argument('--dec-layers-occupancy', default=None, type=int,
                        help='Number of layers in the occupancy decoder.')
    parser.add_argument('--dec-layers-features', default=None, type=int,
                        help='Number of layers in the feature decoder.')
    # number of channels in the hidden decoder layers
    parser.add_argument('--hidden-dims', default=512, type=int,
                        help='Number of channels in the hidden layers of the occupancy decoder.')
    parser.add_argument('--hidden-dims-ft', default=1024, type=int,
                        help='Number of channels in the hidden layers of the feature decoder.')
    parser.add_argument('--num-classes', default=None, type=int, help='Used for evaluation.')
    parser.add_argument('--normalized-cosine', action='store_true')
    parser.add_argument('--val-on-train', action='store_true')

    args = parser.parse_args()
    args.agnostic = args.agnostic or 'agnostic' in args.py_config
    args.maskclip = args.maskclip or 'maskclip' in args.py_config

    if args.name is None:
        args.name = args.work_dir.split(os.path.sep)[-1]

    if args.ft_loss_weight != 1.:
        args.name += f'_{args.ft_loss_weight}ftW'
        args.work_dir += f'_{args.ft_loss_weight}ftW'

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    args.name += f'_{timestamp}'
    args.work_dir += f'_{timestamp}'

    if args.debug:
        args.name += '_debug'
        args.work_dir += '_debug'

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.show or args.no_dist:
        main(-1, args)
    else:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
