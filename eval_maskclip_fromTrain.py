import argparse
import os
import os.path as osp
import pickle
import socket
import time
import warnings
from datetime import datetime

import mmcv
import numpy as np
import open_clip
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
try:
    from mmcv import Config
except:
    from mmengine.config import Config
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger
from timm.scheduler import CosineLRScheduler
from torch.nn import MSELoss
from torchvision import transforms
from tqdm import tqdm

from builder import loss_builder
from dataloader.dataset import get_nuScenes_label_name
from utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_linear_probe
from utils.metric_util import MeanIoU
from utils.prompt_extractor import PromptExtractor
from visualization.training import log_comparison_wandb, log_comparison_clip_wandb, show3d, CLASS_COLORS, show3d_wandb

VIS = True  # False
IGNORE_LABEL = 255
IGNORE_LABEL_SEMANTIC = 0
EMPTY_SEMANTIC_LABEL = 17

warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def assign_labels_clip(predicted_features, text_features, class_offset, maskclip=False, class_mapping_clip=None,
                       ignore_label=None):
    if maskclip:
        if ignore_label is not None and class_mapping_clip is not None:
            nonignored_indices = class_mapping_clip != ignore_label
            text_features = text_features[nonignored_indices]
            class_mapping_clip_nonignore = class_mapping_clip[nonignored_indices]
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


def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    print(f'socket.gethostname(): {socket.gethostname()}')

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
    try:
        agnostic = cfg.feature_learning
    except:
        agnostic = False

    if args.class_weights_path is None:
        try:
            args.class_weights_path = cfg.class_weights_path
        except:
            pass

    if not args.no_class_weights:
        assert 'agn' not in args.py_config or (args.class_weights_path is not None and
                                               os.path.exists(args.class_weights_path)), \
            "If we perform class-agnostic training, we should use class weights!"

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

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

    if distributed and dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    # configure logger
    if not distributed or dist.get_rank() == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
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
    print('done ddp model')

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
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1)
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
        unique_label_clip = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        SemKITTI_label_name_clip = get_nuScenes_label_name("./config/label_mapping/nuscenes.yaml")
        unique_label_str_clip = [SemKITTI_label_name_clip[x] for x in unique_label_clip]
        if args.text_embeddings_path is not None and os.path.exists(args.text_embeddings_path):
            text_features = torch.load(args.text_embeddings_path, map_location='cpu')
            if type(text_features) in [tuple, list]:
                text_features, class_mapping_clip = text_features
                learning_map_gt = train_dataset_loader.dataset.imagepoint_dataset.learning_map_gt
                class_mapping_clip = torch.tensor([learning_map_gt[c.item()] for c in class_mapping_clip]).cuda()
                nonignored = class_mapping_clip != IGNORE_LABEL_SEMANTIC
                text_features_nonignore = text_features[:, nonignored]
                class_mapping_clip_nonignore = class_mapping_clip[nonignored]
            embedding_dim = 512
            if text_features.shape[0] == embedding_dim:
                text_features = text_features.T
                text_features_nonignore = text_features_nonignore.T
            text_features = text_features.float().cuda()
            text_features_nonignore = text_features_nonignore.float().cuda()

        else:
            assert False
            prompt_extractor = PromptExtractor()
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14',
                                                                     pretrained='./ckpts/ovseg_clip_l_9a1909.pth')
            # clip_model.cuda()
            text_features = prompt_extractor(unique_label_str_clip, clip_model).cuda()

    voxel_feature_loss = None
    if feature_learning:
        try:
            voxel_feature_loss_name = cfg.voxel_feature_loss.lower()
        except:
            voxel_feature_loss_name = None
        if voxel_feature_loss_name is not None:
            if voxel_feature_loss_name in ['l2', 'mse']:
                voxel_feature_loss = MSELoss(reduction='mean')
                # voxel_feature_loss = L2Loss()

    class_weights = None
    if args.no_class_weights:
        pass
    elif args.class_weights_path is not None and os.path.exists(args.class_weights_path):
        with open(args.class_weights_path, 'rb') as f:
            class_weights = pickle.load(f)
    elif args.compute_weights:
        n_classes = cfg.get("nbr_class")
        trainId_to_count = {idx: 0 for idx in range(n_classes)}
        for i_iter, loaded_data in enumerate(tqdm(train_dataset_loader)):
            if not feature_learning:
                imgs, img_metas, train_vox_label_occupancy, train_grid, train_pt_labs_agnostic, train_pt_labs_cls = loaded_data
                train_grid_fts = None
            else:
                imgs, img_metas, train_vox_label_occupancy, train_grid, train_pt_labs_agnostic, train_pt_labs_cls, train_grid_fts, train_pt_fts, *_ = loaded_data

            unq, counts = torch.unique(train_vox_label_occupancy, return_counts=True)
            for u, cn in zip(unq, counts):
                trainId_to_count[u.item()] += cn

        # compute the class weights according to the ENet paper:
        class_weights = []
        total_count = sum(trainId_to_count.values())
        for trainId in range(n_classes):
            count = trainId_to_count[trainId]
            trainId_prob = float(count) / float(total_count)
            trainId_weight = 1 / np.log(1.02 + trainId_prob)
            class_weights.append(trainId_weight)
        class_weights = class_weights
        with open(args.class_weights_path, 'wb') as f:
            pickle.dump(class_weights, f)

    if class_weights is not None:
        class_weights = class_weights.float().cuda()

    # get optimizer, loss, scheduler
    optimizer = build_optimizer(my_model, cfg.optimizer)
    loss_func, lovasz_softmax = loss_builder.build(ignore_label=ignore_label, weight=class_weights)
    loss_func_noW, _ = loss_builder.build(ignore_label=-100)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader) * max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )

    # * METRICS *
    # A)  Points
    # A1) Occupied
    CalMeanIou_pts_agnostic = MeanIoU([0, 1], ignore_label=IGNORE_LABEL, label_str=['empty', 'occupied'],
                                      name='pts_agn')

    # B)  Voxels
    # B1) Unique occupied
    CalMeanIou_vox_occupied_agnostic = MeanIoU([0, 1], ignore_label=IGNORE_LABEL, label_str=['empty', 'occupied'],
                                               name='vox_unique_occupied_agnostic')
    # B2) All
    CalMeanIou_vox_all_agnostic = MeanIoU([0, 1], ignore_label=IGNORE_LABEL, label_str=['empty', 'occupied'],
                                          name='vox_agn_all')

    if clip_features:
        CalMeanIou_pts_clip = MeanIoU(unique_label_clip, IGNORE_LABEL_SEMANTIC, unique_label_str_clip, 'pts_clip')
        CalMeanIou_pts_clip_gt = MeanIoU(unique_label_clip, IGNORE_LABEL_SEMANTIC, unique_label_str_clip, 'pts_clip_gt',
                                         extra_classes=0)
        # All
        CalMeanIou_vox_clip_all = MeanIoU(unique_label_clip + [17], IGNORE_LABEL, unique_label_str_clip + ['empty'],
                                          'vox_clip_all', extra_classes=0)
        CalMeanIou_vox_clip_all_gt = MeanIoU(unique_label_clip + [17], IGNORE_LABEL, unique_label_str_clip + ['empty'],
                                             'vox_clip_all', extra_classes=0)
        # occupied
        CalMeanIou_vox_clip_occupied = MeanIoU(unique_label_clip, IGNORE_LABEL,
                                               unique_label_str_clip, 'vox_clip_occupied', extra_classes=0,
                                               extra_classes_pred=1)
        CalMeanIou_vox_clip_occupied_gt = MeanIoU(unique_label_clip, IGNORE_LABEL,
                                                  unique_label_str_clip, 'vox_clip_occupied', extra_classes=0,
                                                  extra_classes_pred=1)

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

    if cfg.resume_from and osp.isfile(cfg.resume_from):
        print('resume from: ', cfg.resume_from)
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        if not distributed or args.no_dist:
            revise_fnc = revise_ckpt_linear_probe
        else:
            revise_fnc = revise_ckpt
        print(my_model.load_state_dict(revise_fnc(ckpt['state_dict'], ddp=distributed and not args.no_dist),
                                       strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
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
    niter_train = len(train_dataset_loader)
    niter_val = len(val_dataset_loader)
    print_freq = cfg.print_freq
    print_freq_wandb_train = niter_train // 10
    print_freq_wandb_val = niter_val // 10
    try:
        print_freq_wandb_train = cfg.print_freq_wandb_train
    except:
        pass
    try:
        print_freq_wandb_val = cfg.print_freq_wandb_val
    except:
        pass

    print('Start training!')
    val_vis_iter = 0

    log2wandb = global_rank == 0 and not args.no_wandb

    # eval
    my_model.eval()
    val_loss_list = []
    val_loss_dict = {}
    CalMeanIou_pts_agnostic.reset()
    CalMeanIou_vox_all_agnostic.reset()
    CalMeanIou_vox_occupied_agnostic.reset()
    if clip_features:
        CalMeanIou_pts_clip.reset()
        CalMeanIou_pts_clip_gt.reset()
        CalMeanIou_vox_clip_all.reset()
        CalMeanIou_vox_clip_occupied.reset()

    data_time_s = time.time()
    time_val_ep_start = time.time()

    with torch.no_grad():
        for i_iter_val, loaded_data in enumerate(val_dataset_loader):
            data_time_e = time.time()
            time_s = time.time()
            if not feature_learning:
                imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs_agnostic, val_vox_label_cls = loaded_data
                val_pt_fts = None
                val_grid_fts = None
            else:
                imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs_agnostic, val_vox_label_cls, val_grid_fts, val_pt_fts, val_vox_label_cls_val, *_ = loaded_data

            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            vox_label = val_vox_label_agnostic.cuda()
            val_pt_labs_agnostic = val_pt_labs_agnostic.cuda()
            val_grid_fts = val_grid_fts.cuda()
            if voxel_feature_loss is not None:
                val_pt_fts = val_pt_fts.cuda()

            predict_labels_vox_occupancy, predict_labels_pts, predict_fts_vox, predict_fts_pts = my_model(img=imgs,
                                                                                                          img_metas=img_metas,
                                                                                                          points=val_grid_float,
                                                                                                          features=val_grid_fts.clone())

            # CLIP-related code
            if clip_features:
                # have class_offset=1 as we want to ignore label 0
                with torch.no_grad():

                    ##############
                    # * POINTS * #
                    ##############

                    # assign labels to predictions
                    loss_ce_clip, loss_lovasz_clip, non_ignore, outputs_pts_clip, targets, val_grid_fts_int = assign_clip_labels(
                        args, class_mapping_clip, loss_func_noW, lovasz_softmax, predict_fts_pts, text_features,
                        val_grid_fts, val_vox_label_cls, compute_loss=args.debug)
                    if loss_ce_clip is not None and global_rank == 0 and not args.no_wandb:
                        wandb.log({"loss/val_ce_clip": loss_ce_clip}, commit=False)
                        wandb.log({"loss/val_lovasz_clip": loss_lovasz_clip}, commit=False)
                    if args.show:
                        fig = plt.figure()
                        show3d(val_grid_fts_int[non_ignore].cpu(), fig, 1, 2, 1, s=0.5,
                               labels=outputs_pts_clip.squeeze().cpu(), title='assignments of predicted features',
                               colors=CLASS_COLORS)  # _TINY)
                        show3d(val_grid_fts_int[non_ignore].cpu(), fig, 1, 2, 2, s=0.5,
                               title='GT labels',
                               labels=targets.squeeze().cpu() + 1,
                               colors=CLASS_COLORS)  # _TINY)
                        plt.show()
                        print('Delete this plotting!!!')
                    CalMeanIou_pts_clip._after_step(outputs_pts_clip, targets + 1)

                    # assign labels to GT OpenCLIP+ features
                    loss_ce_clip_gt, loss_lovasz_clip_gt, non_ignore, outputs_pts_clip_gt, targets, val_grid_fts_int = assign_clip_labels(
                        args, class_mapping_clip, loss_func_noW, lovasz_softmax, val_pt_fts, text_features,
                        val_grid_fts, val_vox_label_cls, compute_loss=args.debug,
                        ignore_label=IGNORE_LABEL_SEMANTIC)
                    if loss_ce_clip_gt is not None and global_rank == 0 and not args.no_wandb:
                        wandb.log({"loss/val_ce_clip_gt": loss_ce_clip_gt}, commit=False)
                        wandb.log({"loss/val_lovasz_clip_gt": loss_lovasz_clip_gt}, commit=False)
                    if args.show:
                        fig = plt.figure()
                        show3d(val_grid_fts_int[non_ignore].cpu(), fig, 1, 2, 1, s=0.5,
                               labels=outputs_pts_clip_gt.squeeze().cpu(),
                               title='assignments of projected GT features',
                               colors=CLASS_COLORS)  # _TINY)
                        show3d(val_grid_fts_int[non_ignore].cpu(), fig, 1, 2, 2, s=0.5,
                               title='GT labels',
                               labels=targets.squeeze().cpu() + 1,
                               colors=CLASS_COLORS)  # _TINY)
                        plt.show()
                        print('Delete this plotting!!!')
                    CalMeanIou_pts_clip_gt._after_step(outputs_pts_clip_gt, targets + 1)

                    with torch.no_grad():
                        ##############
                        # * VOXELS * #
                        ##############
                        # get the occupied voxels; for these voxels, get the feature predictions
                        occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox_occupancy.argmax(1) == 1))
                        # predict features at those positions
                        n_occ = occupied_voxels_loc.shape[1]
                        outputs_vox_clip = torch.ones_like(val_vox_label_cls_val,
                                                           device=predict_fts_vox.device) * EMPTY_SEMANTIC_LABEL
                        if n_occ > 0:
                            predicted_features_occupied_vox = predict_fts_vox[occupied_voxels_loc[0], :,
                                                              occupied_voxels_loc[1], occupied_voxels_loc[2],
                                                              occupied_voxels_loc[3]].unsqueeze(0)
                            # assign labels
                            _outputs_vox_clip = assign_clip_labels(
                                args, class_mapping_clip, None, None, predicted_features_occupied_vox,
                                text_features, None, None, assignment_only=True)
                            outputs_vox_clip[occupied_voxels_loc[0], occupied_voxels_loc[1], occupied_voxels_loc[2],
                                             occupied_voxels_loc[3]] = _outputs_vox_clip
                            if i_iter_val % print_freq_wandb_val == 0 and log2wandb:
                                name = f'val/predict_occ+clip'
                                show3d_wandb(occupied_voxels_loc[1:].T.squeeze().long().cpu(), name, None,
                                             colors=CLASS_COLORS, labels=_outputs_vox_clip.cpu().squeeze())
                        # evaluate voxels
                        # a) all voxels
                        CalMeanIou_vox_clip_all._after_step(outputs_vox_clip.cpu(), val_vox_label_cls_val)
                        # b) GT-occupied voxels only
                        _occupied_idx = torch.bitwise_and(val_vox_label_cls[0] != EMPTY_SEMANTIC_LABEL,
                                                          val_vox_label_cls[0] != IGNORE_LABEL_SEMANTIC)
                        targets = val_vox_label_cls[0, _occupied_idx]
                        predictions = outputs_vox_clip[0, _occupied_idx]
                        CalMeanIou_vox_clip_occupied._after_step(predictions, targets.cuda())

            loss = 0
            if cfg.lovasz_input.lower() != 'none':
                if cfg.lovasz_input == 'voxel':
                    lovasz_input = predict_labels_vox_occupancy
                    lovasz_label = vox_label
                else:
                    lovasz_input = predict_labels_pts
                    lovasz_label = val_pt_labs_agnostic
                loss_lovasz = lovasz_softmax(
                    torch.nn.functional.softmax(lovasz_input, dim=1).detach(),
                    lovasz_label, ignore=ignore_label
                )
                loss += loss_lovasz
                if 'lovasz' not in val_loss_dict:
                    val_loss_dict['lovasz'] = [loss_lovasz.item()]

            if cfg.ce_input.lower() != 'none':
                if cfg.ce_input == 'voxel':
                    ce_input = predict_labels_vox_occupancy
                    ce_label = vox_label
                else:
                    ce_input = predict_labels_pts.squeeze(-1).squeeze(-1)
                    ce_label = val_pt_labs_agnostic.squeeze(-1)
                loss_ce = loss_func(ce_input.detach(), ce_label)
                if 'ce' not in val_loss_dict:
                    val_loss_dict['ce'] = [loss_ce.item()]
                loss += loss_ce

            if voxel_feature_loss is not None:
                ft_pred = predict_fts_pts
                ft_target = val_pt_fts
                loss_ft = voxel_feature_loss(ft_pred.detach(), ft_target)
                if 'feature' not in val_loss_dict:
                    val_loss_dict['feature'] = [loss_ft.item()]
                loss += loss_ft

            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1)  # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs_agnostic = val_pt_labs_agnostic.squeeze(-1).cpu()

            predict_labels_vox_occupancy = torch.argmax(predict_labels_vox_occupancy, dim=1)
            predict_labels_vox_occupancy = predict_labels_vox_occupancy.detach().cpu()

            gt_labels_vox_agnostic, predict_labels_vox_occupancy, gt_labels_pts_agnostic, predict_labels_pts_agnostic = get_agnostic_labels(
                args, predict_labels_pts, predict_labels_vox_occupancy, val_vox_label_agnostic,
                val_pt_labs_agnostic)

            for count in range(len(val_grid_int)):
                # points
                CalMeanIou_pts_agnostic._after_step(predict_labels_pts_agnostic[count],
                                                    val_pt_labs_agnostic[count])

                # voxels
                CalMeanIou_vox_occupied_agnostic._after_step(
                    predict_labels_vox_occupancy[count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                                 val_grid_int[count][:, 2]].flatten(),
                    val_pt_labs_agnostic[count]
                )

                val_vox_label_occupancy_val = val_vox_label_cls_val.clone()
                # occluded areas + semantic ignore labels
                val_vox_label_occupancy_val[val_vox_label_cls_val != IGNORE_LABEL] = 1
                # add back the semantic ignore areas
                val_vox_label_occupancy_val[val_vox_label_cls == IGNORE_LABEL_SEMANTIC] = 1
                val_vox_label_occupancy_val[val_vox_label_cls_val == EMPTY_SEMANTIC_LABEL] = 0
                CalMeanIou_vox_all_agnostic._after_step(predict_labels_vox_occupancy[count],
                                                        val_vox_label_agnostic[count])

            val_loss_list.append(loss.detach().cpu().numpy())
            if i_iter_val % print_freq == 0 and (args.no_dist or dist.get_rank() == 0):
                logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)' % (
                    epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))

            if i_iter_val % print_freq_wandb_val == 0 and (args.no_dist or dist.get_rank() == 0):
                if VIS and not args.no_wandb:
                    if clip_features:
                        # CLIP predictions (already computed)
                        predict_labels_vox_clip0 = torch.ones_like(val_vox_label_cls[0]) * 17
                        val_grid_fts_int_nonignore = val_grid_fts_int[non_ignore]
                        predict_labels_vox_clip0[val_grid_fts_int_nonignore[..., 0],
                                                 val_grid_fts_int_nonignore[..., 1],
                                                 val_grid_fts_int_nonignore[..., 2]] = outputs_pts_clip.cpu()
                        # CLIP targets (the best that we can get from CLIP features)
                        val_vox_label_cls_clip0 = torch.ones_like(val_vox_label_cls[0]) * 17
                        gt_clip_pred, _ = assign_labels_clip(
                            val_pt_fts[non_ignore].unsqueeze(0).float().cuda(), text_features, 1,
                            maskclip=args.maskclip, class_mapping_clip=class_mapping_clip, ignore_label=0)
                        val_vox_label_cls_clip0[val_grid_fts_int_nonignore[..., 0],
                                                val_grid_fts_int_nonignore[..., 1],
                                                val_grid_fts_int_nonignore[..., 2]] = gt_clip_pred.cpu()

                        log_comparison_clip_wandb(val_vox_label_cls_clip0, predict_labels_vox_clip0, 17, 255,
                                                  'val', global_step=None, debug=args.debug)

                    gt_fts = val_pt_fts[0] if feature_learning else None
                    pred_fts = predict_fts_pts[0] if feature_learning else None
                    loc_fts = val_grid_fts[0] if feature_learning else val_grid_int[0]
                    log_comparison_wandb(imgs[0], val_vox_label_agnostic[0], predict_labels_vox_occupancy[0],
                                         free_label,
                                         ignore_label,
                                         agnostic, 'val',
                                         # val_vis_iter,
                                         None,
                                         gt_fts, pred_fts, loc_fts, gt_cls_labels=val_vox_label_cls[0],
                                         debug=args.debug, commit=True)

            time_e = time.time()
            pass_iter_time = time_e - time_s
            data_iter_time = data_time_e - data_time_s
            if global_rank == 0 and not args.no_wandb:
                wandb.log({"time/val_iter": pass_iter_time, "time/val_data_iter": data_iter_time}, commit=False)
            data_time_s = time.time()
            val_vis_iter += 1

    # POINTS
    val_miou_pts_agn = CalMeanIou_pts_agnostic._after_epoch(log_wandb=log2wandb, tag='miou_pts_agnostic/val/',
                                                            step=global_iter)
    if clip_features:
        val_miou_pts_clip = CalMeanIou_pts_clip._after_epoch(log_wandb=log2wandb, tag='miou_pts_clip/val/',
                                                             step=global_iter)
        val_miou_pts_clip_gt = CalMeanIou_pts_clip_gt._after_epoch(log_wandb=log2wandb, step=global_iter,
                                                                   tag='miou_pts_clip_gt/val/')
    # VOXELS
    val_miou_vox_agn_occ = CalMeanIou_vox_occupied_agnostic._after_epoch(log_wandb=log2wandb, step=global_iter,
                                                                         tag='miou_vox_agnostic_occupied/val/')
    val_miou_vox_agn_all = CalMeanIou_vox_all_agnostic._after_epoch(log_wandb=log2wandb, step=global_iter,
                                                                    tag='miou_vox_agnostic_all/val/')
    if clip_features:
        val_miou_vox_clip_occ = CalMeanIou_vox_clip_occupied._after_epoch(log_wandb=log2wandb, step=global_iter,
                                                                          tag='miou_vox_clip_occupied/val/')
        val_miou_vox_clip_all = CalMeanIou_vox_clip_all._after_epoch(log_wandb=log2wandb, step=global_iter,
                                                                     tag='miou_vox_clip_all/val/')

    time_val_ep_elapsed = time.time() - time_val_ep_start
    # log metrics to wandb
    if log2wandb:
        # POINTS
        wandb.log({"miou_pts_agnostic/val/mean": val_miou_pts_agn}, commit=False)
        # VOXELS
        wandb.log({"miou_vox_agnostic_occupied/val/mean": val_miou_vox_agn_occ,
                   "miou_vox_agnostic_all/val/mean": val_miou_vox_agn_all},
                  commit=False)
        if clip_features:
            # POINTS
            wandb.log({"miou_pts_clip/val/mean": val_miou_pts_clip,
                       "miou_pts_clip_gt/val/mean": val_miou_pts_clip_gt}, commit=False)
            # VOXELS
            wandb.log({"miou_vox_clip_occupied/val/mean": val_miou_vox_clip_occ,
                       "miou_vox_clip_all/val/mean": val_miou_vox_clip_all}, commit=False)
        wandb.log({"time/val_epoch": time_val_ep_elapsed})

    if clip_features:
        if best_val_miou_pts < val_miou_pts_clip:
            best_val_miou_pts = val_miou_pts_clip
        if best_val_miou_vox < val_miou_vox_clip_all:
            best_val_miou_vox = val_miou_vox_clip_all
        logger.info(
            'Current val miou pts is %.3f while the best val miou pts is %.3f' % (
                val_miou_pts_clip, best_val_miou_pts))
        logger.info(
            'Current val miou vox is %.3f while the best val miou vox is %.3f' % (
                val_miou_vox_clip_all, best_val_miou_vox))
    else:
        if best_val_miou_pts < val_miou_pts_agn:
            best_val_miou_pts = val_miou_pts_agn
        if best_val_miou_vox < val_miou_vox_agn_all:
            best_val_miou_vox = val_miou_vox_agn_all
        logger.info(
            'Current val miou pts is %.3f while the best val miou pts is %.3f' % (
                val_miou_pts_agn, best_val_miou_pts))
        logger.info(
            'Current val miou vox is %.3f while the best val miou vox is %.3f' % (
                val_miou_vox_agn_all, best_val_miou_vox))
    logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))
    dist.barrier()

    print('Finish.')
    return


def compute_upper_bound_clip(args, class_mapping_clip, feature_learning, text_features, train_dataset_loader_noaug,
                             train_pt_fts, unique_label, unique_label_str, val_dataset_loader):
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = os.path.join('./upperbound_out', timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Saving outputs to {output_dir}')
    # inverse transformation for images
    img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    mean_neg = [-m for m in img_norm_cfg['mean']]
    std_inv = [1 / s for s in img_norm_cfg['std']]
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=std_inv),
                                   transforms.Normalize(mean=mean_neg,
                                                        std=[1., 1., 1.]),
                                   ])
    CalMeanIou_pts_upperbound_val = MeanIoU(unique_label, 0, unique_label_str, 'ptsWProj_clip_upperbound_val')
    CalMeanIou_pts_upperbound_val.reset()
    CalMeanIou_pts_upperbound_trainval = MeanIoU(unique_label, 0, unique_label_str,
                                                 'ptsWProj_clip_upperbound_trainval')
    CalMeanIou_pts_upperbound_trainval.reset()
    for i_iter, loaded_data in enumerate(tqdm(val_dataset_loader)):
        if not feature_learning:
            imgs, img_metas, train_vox_label_agnostic, train_grid, train_pt_labs, train_vox_label_cls = loaded_data
            train_grid_fts = None
        else:
            imgs, img_metas, train_vox_label_agnostic, train_grid, train_pt_labs, train_vox_label_cls, train_grid_fts, train_pt_fts, train_vox_label_cls_val, *_ = loaded_data

        train_grid_fts_int = train_grid_fts.long()[0]
        targets = train_vox_label_cls[
            0, train_grid_fts_int[:, 0], train_grid_fts_int[:, 1], train_grid_fts_int[:, 2]
        ]
        predictions = assign_labels_clip(
            train_pt_fts.float().cuda(), text_features, 1, maskclip=args.maskclip,
            class_mapping_clip=class_mapping_clip, ignore_label=0)[0].cpu()
        acc = targets == predictions
        CalMeanIou_pts_upperbound_val._after_step(predictions.squeeze(), targets.squeeze())
        CalMeanIou_pts_upperbound_trainval._after_step(predictions.squeeze(), targets.squeeze())

        if args.show or i_iter % 10 == 0:
            # save predictions
            fig = plt.figure()
            images_t = invTrans(imgs[0])
            images_t = images_t.permute(0, 2, 3, 1).detach().cpu().numpy()[..., ::-1]
            images_2rows_color = np.concatenate((np.concatenate(images_t[[2, 0, 1]], axis=1),
                                                 np.concatenate(images_t[[4, 3, 5]], axis=1)), axis=0)
            images_2rows_color = images_2rows_color.clip(0, 255).astype(np.uint8)

            gs = GridSpec(2, 2, figure=fig)
            ax_img = fig.add_subplot(gs[0, :])
            ax_img.imshow(images_2rows_color)
            ax_img.get_xaxis().set_visible(False)
            ax_img.get_yaxis().set_visible(False)

            ax_pred = fig.add_subplot(gs[1, 0], projection='3d')
            ax_gt = fig.add_subplot(gs[1, 1], projection='3d')

            show3d(train_grid_fts_int, fig, 2, 2, 3, ax=ax_pred, s=1., labels=predictions.squeeze(),
                   colors=CLASS_COLORS)  # _TINY)
            show3d(train_grid_fts_int, fig, 2, 2, 4, ax=ax_gt, s=1., labels=targets.squeeze(),
                   colors=CLASS_COLORS)  # _TINY)
            # plt.tight_layout()
            if args.show:
                plt.show()
            else:
                plt.savefig(os.path.join(output_dir, f'val{i_iter}.jpg'))  # , bbox_inches='tight')
                plt.close(fig)
    print('\n********************************************\n')
    miou_val, miou_val_per_class = CalMeanIou_pts_upperbound_val._after_epoch(return_per_class=True)
    print(f'OVSeg VAL mIoU: {miou_val}\n')
    miou_val_per_class_str_names = ';'.join(CalMeanIou_pts_upperbound_val.label_str)
    miou_val_per_class_str_ious = ';'.join(['{:.1f}'.format(iou) for iou in miou_val_per_class])
    print(f'OVSeg VAL mIoU per class:\n{miou_val_per_class_str_names}\n{miou_val_per_class_str_ious}\n')
    print('\n********************************************\n')
    # exit()
    CalMeanIou_pts_upperbound_train = MeanIoU(unique_label, 0, unique_label_str, 'ptsWProj_clip_upperbound_train')
    CalMeanIou_pts_upperbound_train.reset()
    for i_iter, loaded_data in enumerate(tqdm(train_dataset_loader_noaug)):
        if not feature_learning:
            imgs, img_metas, train_vox_label_agnostic, train_grid, train_pt_labs, train_vox_label_cls = loaded_data
            train_grid_fts = None
        else:
            imgs, img_metas, train_vox_label_agnostic, train_grid, train_pt_labs, train_vox_label_cls, train_grid_fts, train_pt_fts, *_ = loaded_data

        train_grid_fts_int = train_grid_fts.long()[0]
        targets = train_vox_label_cls[
            0, train_grid_fts_int[:, 0], train_grid_fts_int[:, 1], train_grid_fts_int[:, 2]
        ]
        predictions = assign_labels_clip(
            train_pt_fts.float().cuda(), text_features, 1, maskclip=args.maskclip,
            class_mapping_clip=class_mapping_clip, ignore_label=0)[0].cpu()
        acc = targets == predictions
        cur_miou = CalMeanIou_pts_upperbound_train._after_step(predictions.squeeze(), targets.squeeze(),
                                                               return_cur_miou=True)
        CalMeanIou_pts_upperbound_trainval._after_step(predictions.squeeze(), targets.squeeze())

        if args.show or i_iter % 20 == 0:
            # save predictions
            fig = plt.figure()
            images_t = invTrans(imgs[0])
            images_t = images_t.permute(0, 2, 3, 1).detach().cpu().numpy()[..., ::-1]
            images_2rows_color = np.concatenate((np.concatenate(images_t[[2, 0, 1]], axis=1),
                                                 np.concatenate(images_t[[4, 3, 5]], axis=1)), axis=0)
            images_2rows_color = images_2rows_color.clip(0, 255).astype(np.uint8)

            gs = GridSpec(2, 2, figure=fig)
            ax_img = fig.add_subplot(gs[0, :])
            ax_img.imshow(images_2rows_color)
            ax_img.get_xaxis().set_visible(False)
            ax_img.get_yaxis().set_visible(False)

            ax_pred = fig.add_subplot(gs[1, 0], projection='3d')
            ax_gt = fig.add_subplot(gs[1, 1], projection='3d')

            show3d(train_grid_fts_int, fig, 2, 2, 3, ax=ax_pred, s=1., labels=predictions.squeeze(),
                   colors=CLASS_COLORS)  # _TINY)
            show3d(train_grid_fts_int, fig, 2, 2, 4, ax=ax_gt, s=1., labels=targets.squeeze(),
                   colors=CLASS_COLORS)  # _TINY)
            # plt.tight_layout()
            if args.show:
                plt.suptitle('Scene mIoU of seen categories: {:.1f}'.format(cur_miou))
                plt.show()
            else:
                plt.savefig(os.path.join(output_dir, f'train{i_iter}.jpg'))  # , bbox_inches='tight')
                plt.close(fig)
    print('\n********************************************\n')
    miou_train = CalMeanIou_pts_upperbound_train._after_epoch()
    print(f'OVSeg TRAIN mIoU: {miou_train}\n')
    print('\n********************************************\n')
    print('\n********************************************\n')
    miou_trainval = CalMeanIou_pts_upperbound_trainval._after_epoch()
    print(f'OVSeg TRAINVAL mIoU: {miou_trainval}\n')
    print('\n********************************************\n')
    torch.save(CalMeanIou_pts_upperbound_train.confusion_matrix.cpu(), os.path.join(output_dir, 'train.pth'))
    torch.save(CalMeanIou_pts_upperbound_trainval.confusion_matrix.cpu(), os.path.join(output_dir, 'trainval.pth'))
    torch.save(CalMeanIou_pts_upperbound_val.confusion_matrix.cpu(), os.path.join(output_dir, 'val.pth'))
    print(f'Saved confusion matrices to {output_dir}')
    print('\nSummary:')
    print(f'OVSeg VAL mIoU: {miou_val}')
    print(f'OVSeg TRAIN mIoU: {miou_train}')
    print(f'OVSeg TRAINVAL mIoU: {miou_trainval}')
    exit()
    return train_grid_fts_int, train_pt_fts


def assign_clip_labels(args, class_mapping_clip, loss_func_noW, lovasz_softmax, outputs_pts_fts, text_features,
                       train_grid_fts, train_vox_label_cls, compute_loss=True, assignment_only=False, ignore_label=0):
    # points
    outputs_pts_clip, logits_clip = assign_labels_clip(
        outputs_pts_fts.float().cuda(), text_features, 1, maskclip=args.maskclip,
        class_mapping_clip=class_mapping_clip, ignore_label=ignore_label)
    if assignment_only:
        return outputs_pts_clip
    train_grid_fts_int = None
    if train_vox_label_cls is not None:
        train_grid_fts_int = train_grid_fts.long()
        train_pt_labs_fts_list = []
        for bi in range(train_grid_fts_int.shape[0]):
            train_grid_fts_int_cur = train_grid_fts_int[bi]
            train_pt_labs_fts_cur = train_vox_label_cls[
                bi, train_grid_fts_int_cur[:, 0], train_grid_fts_int_cur[:, 1], train_grid_fts_int_cur[:, 2]
            ]
            train_pt_labs_fts_list.append(train_pt_labs_fts_cur)
        train_pt_labs_fts = torch.cat(train_pt_labs_fts_list).unsqueeze(0).cuda()
        non_ignore = train_pt_labs_fts != 0
        targets = (train_pt_labs_fts[non_ignore] - 1).unsqueeze(0)
        outputs_pts_clip = outputs_pts_clip[non_ignore].unsqueeze(0).cuda()
    else:
        targets = None
        non_ignore = None
        outputs_pts_clip = outputs_pts_clip.cuda()

    loss_ce_clip, loss_lovasz_clip = None, None
    if compute_loss:
        logits_clip_non_ignore = logits_clip.permute(0, 2, 1)[non_ignore].unsqueeze(0).permute(0, 2, 1)
        # set_ft = set(list(map(tuple, train_grid_fts.cpu()[0].tolist())))
        # set_all = set(list(map(tuple, train_grid.cpu()[0].tolist())))
        # diff = set_ft - set_all
        loss_ce_clip = loss_func_noW(logits_clip_non_ignore.detach(), targets.detach())
        loss_lovasz_clip = lovasz_softmax(
            torch.nn.functional.softmax(logits_clip_non_ignore.unsqueeze(-1).unsqueeze(-1), dim=1).detach(),
            targets.unsqueeze(-1).detach())
    return loss_ce_clip, loss_lovasz_clip, non_ignore, outputs_pts_clip, targets, train_grid_fts_int


def get_agnostic_labels(args, predict_labels_pts, predict_labels_vox, val_vox_labs, val_pt_labs):
    if not args.agnostic:
        # 1) voxel grid
        #   A) GT
        gt_labels_vox_agnostic = torch.zeros_like(val_vox_labs)
        gt_labels_vox_agnostic_occ = torch.where(val_vox_labs < 17)
        gt_labels_vox_agnostic[gt_labels_vox_agnostic_occ] = 1
        #   B) predictions
        predict_labels_vox_agnostic = torch.zeros_like(predict_labels_vox)
        predict_labels_vox_agnostic_occ = torch.where(predict_labels_vox < 17)
        predict_labels_vox_agnostic[predict_labels_vox_agnostic_occ] = 1

        # 2) points
        #   A) GT
        gt_labels_pts_agnostic = torch.zeros_like(val_pt_labs)
        gt_labels_pts_agnostic_occ = torch.where(val_pt_labs < 17)
        gt_labels_pts_agnostic[gt_labels_pts_agnostic_occ] = 1
        #   B) predictions
        predict_labels_pts_agnostic = torch.zeros_like(predict_labels_pts)
        predict_labels_pts_agnostic_occ = torch.where(predict_labels_pts < 17)
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
    parser.add_argument('--no-class-weights', action='store_true', help="Do not use class re-weighting.")
    parser.add_argument('--full-eval-freq', default=10, type=int,
                        help='How often during the training should we evaluate predicted features at the predicted occupied voxels.')
    # number of layers in the decoders
    parser.add_argument('--dec-layers-occupancy', default=None, type=int,
                        help='Number of layers in the occupancy decoder.')
    parser.add_argument('--dec-layers-features', default=None, type=int,
                        help='Number of layers in the feature decoder.')
    # number of channels in the hidden decoder layers
    parser.add_argument('--hidden-dims', default=None, type=int,
                        help='Number of channels in the hidden layers of the occupancy decoder.')
    parser.add_argument('--hidden-dims-ft', default=None, type=int,
                        help='Number of channels in the hidden layers of the feature decoder.')

    args = parser.parse_args()
    args.agnostic = args.agnostic or 'agnostic' in args.py_config
    args.maskclip = args.maskclip or 'maskclip' in args.py_config

    if args.name is None:
        args.name = args.work_dir.split(os.path.sep)[-1]

    if args.ft_loss_weight != 1.:
        args.name += f'_{args.ft_loss_weight}ftW'
        args.work_dir += f'_{args.ft_loss_weight}ftW'

    if args.dec_layers_occupancy is not None:
        args.name += f'_{args.dec_layers_occupancy}decOcc'
        args.work_dir += f'_{args.dec_layers_occupancy}decOcc'
    if args.hidden_dims is not None:
        args.name += f'_{args.hidden_dims}hidOcc'
        args.work_dir += f'_{args.hidden_dims}hidOcc'

    if args.dec_layers_features is not None:
        args.name += f'_{args.dec_layers_features}decFt'
        args.work_dir += f'_{args.dec_layers_features}decFt'
    if args.hidden_dims_ft is not None:
        args.name += f'_{args.hidden_dims_ft}hidFt'
        args.work_dir += f'_{args.hidden_dims_ft}hidFt'

    if args.no_class_weights:
        args.name += '_noClsW'
        args.work_dir += '_noClsW'

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
        torch.set_num_threads(1)
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
