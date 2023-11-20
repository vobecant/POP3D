import argparse
import os
import os.path as osp
import socket
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from mmcv import Config
from mmseg.utils import get_root_logger

from dataloader.dataset import get_nuScenes_label_name
from train import IGNORE_LABEL_SEMANTIC
from utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_linear_probe
from utils.metric_util import MeanIoU
from visualization.training import log_comparison_wandb

VIS = True  # False
EMPTY_SEMANTIC_LABEL = 17
IGNORE_LABEL = 255
OCCUPIED_AGNOSTIC_LABEL = 1

warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


class L2Loss(torch.nn.Module):
    def __init__(self, sum_dim=2):
        super(L2Loss, self).__init__()
        self.sum_dim = sum_dim

    def forward(self, pred, target):
        loss = ((target - pred) ** 2).sum(dim=2).mean()
        return loss


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


@torch.no_grad()
def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    print(f'socket.gethostname(): {socket.gethostname()}')

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

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
            scale_rate=cfg.get('scale_rate', 1),
            # eval_mode=True,
            num_workers=0,
            class_agnostic=False
        )

    feature_learning = clip_features = False

    CalMeanIou_vox_occupied_agnostic_unique = MeanIoU([0, 1], ignore_label=IGNORE_LABEL,
                                                      label_str=['empty', 'occupied'], name='vox_occ_agn_unq')
    CalMeanIou_pts = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts')
    CalMeanIou_vox_all_agnostic = MeanIoU([0, 1], ignore_label=IGNORE_LABEL, label_str=['empty', 'occupied'],
                                          name='vox_agn_all')
    CalMeanIou_vox_all_semantic = MeanIoU(unique_label, ignore_label=IGNORE_LABEL, label_str=unique_label_str,
                                          name='vox_sem_all')
    CalMeanIou_vox_occupied_semantic = MeanIoU(unique_label, ignore_label=IGNORE_LABEL, label_str=unique_label_str,
                                          name='vox_sem_occ', extra_classes=0, extra_classes_pred=1)
    CalMeanIou_pts_agnostic = MeanIoU([0, 1], ignore_label=IGNORE_LABEL, label_str=['empty', 'occupied'],
                                      name='pts_agn')

    # resume and load
    epoch = 0

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
    print_freq = cfg.print_freq
    try:
        print_freq_wandb_train = cfg.print_freq_wandb_train
    except:
        print_freq_wandb_train = cfg.print_freq
    try:
        print_freq_wandb_val = cfg.print_freq_wandb_val
    except:
        print_freq_wandb_val = cfg.print_freq

    print('Start training!')
    val_vis_iter = 0

    # eval
    my_model.eval()
    val_loss_list = []
    val_loss_dict = {}
    CalMeanIou_pts.reset()
    CalMeanIou_vox_occupied_agnostic_unique.reset()
    CalMeanIou_pts_agnostic.reset()
    CalMeanIou_vox_all_agnostic.reset()
    CalMeanIou_vox_all_semantic.reset()
    CalMeanIou_vox_occupied_semantic.reset()

    data_time_s = time.time()
    time_val_ep_start = time.time()

    with torch.no_grad():
        for i_iter_val, loaded_data in enumerate(val_dataset_loader):
            if args.max_iter is not None and i_iter_val >= args.max_iter:
                break
            data_time_e = time.time()
            time_s = time.time()
            if not feature_learning:
                imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs, val_vox_label_cls, val_vox_label_cls_val = loaded_data
                val_pt_fts = None
                val_grid_fts = None
            else:
                imgs, img_metas, val_vox_label_agnostic, val_grid, val_pt_labs, val_vox_label_cls, val_grid_fts, val_pt_fts, val_vox_label_cls_val, *_ = loaded_data

            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            vox_label = val_vox_label_agnostic.cuda()
            val_pt_labs = val_pt_labs.cuda()
            if feature_learning:
                val_grid_fts = val_grid_fts.cuda()

            predict_labels_vox, predict_labels_pts, *_ = my_model(img=imgs, img_metas=img_metas,
                                                                  points=val_grid_float.clone())
            occupied_voxels_loc = torch.stack(torch.where(predict_labels_vox.argmax(1) == 1))

            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1)  # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs = val_pt_labs.squeeze(-1).cpu()

            predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
            predict_labels_vox = predict_labels_vox.detach().cpu()

            gt_labels_vox_agnostic = val_vox_label_agnostic
            predict_labels_vox_agnostic = torch.zeros_like(predict_labels_vox, dtype=val_vox_label_cls.dtype)
            predict_labels_vox_agnostic[predict_labels_vox < 17] = 1
            gt_labels_pts_agnostic = torch.ones_like(val_pt_labs, dtype=val_pt_labs.dtype)
            predict_labels_pts_agnostic = torch.zeros_like(predict_labels_pts, dtype=val_pt_labs.dtype)
            predict_labels_pts_agnostic[predict_labels_pts < 17] = 1

            for count in range(len(val_grid_int)):
                # points
                CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count])
                CalMeanIou_pts_agnostic._after_step(predict_labels_pts_agnostic[count],
                                                    gt_labels_pts_agnostic[count])

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

                CalMeanIou_vox_all_semantic._after_step(predict_labels_vox, val_vox_label_cls_val)

                # b) occupied only
                _occupied_idx = val_vox_label_agnostic[count] == OCCUPIED_AGNOSTIC_LABEL
                targets = val_vox_label_agnostic[count, _occupied_idx]
                predictions = predict_labels_vox_agnostic[count, _occupied_idx]
                CalMeanIou_vox_occupied_agnostic_unique._after_step(predictions, targets)
                # b) GT-occupied voxels only
                _occupied_idx = torch.bitwise_and(val_vox_label_cls[0] != EMPTY_SEMANTIC_LABEL,
                                                  val_vox_label_cls[0] != IGNORE_LABEL_SEMANTIC)
                targets = val_vox_label_cls[0, _occupied_idx]
                predictions = predict_labels_vox[0, _occupied_idx].cuda()
                CalMeanIou_vox_occupied_semantic._after_step(predictions, targets.cuda())

            # val_loss_list.append(loss.detach().cpu().numpy())
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
    val_miou_vox_all_semantic = CalMeanIou_vox_all_semantic._after_epoch(log_wandb=log2wandb,
                                                                         tag="miou_vox_sem_all/val", step=None)
    val_miou_vox_occ_semantic = CalMeanIou_vox_occupied_semantic._after_epoch(log_wandb=log2wandb,
                                                                              tag="miou_vox_sem_occ/val", step=None)

    print(f'val_miou_pts: {val_miou_pts}')
    print(f'val_miou_vox_occupied (evaluate voxel agnostic segmentation only at the GT-occupied voxels): '
          f'{val_miou_vox_occupied_agnostic_unique}')
    print(f'val_miou_pts_agn: {val_miou_pts_agn}')
    print(f'val_miou_vox_agn (evaluate voxel agnostic segmentation everywhere): {val_miou_vox_all}')
    print(f'val_miou_vox_semantic (evaluate voxel agnostic segmentation *everywhere*): {val_miou_vox_all_semantic}')
    print(f'val_miou_vox_semantic (evaluate voxel agnostic segmentation *at GT*): {val_miou_vox_occ_semantic}')

    time_val_ep_elapsed = time.time() - time_val_ep_start
    # log metrics to wandb
    if global_rank == 0 and not args.no_wandb:
        wandb.log({"miou_pts/val": val_miou_pts, "miou_vox_all/val": val_miou_vox_all}, commit=False)
        wandb.log({"miou_pts_agn/val": val_miou_pts_agn, "miou_vox_agn/val": val_miou_vox_all}, commit=False)
        wandb.log({"loss/val_mean_ep": np.mean(val_loss_list)}, commit=False)
        for key, vals in val_loss_dict.items():
            wandb.log({f"loss/val_{key}": torch.mean(torch.tensor(vals))}, commit=False)
        wandb.log({"time/val_epoch": time_val_ep_elapsed})  # , step=global_iter)
    # val_vis_iter += 10000

    # if best_val_miou_pts < val_miou_pts:
    #     best_val_miou_pts = val_miou_pts
    # if best_val_miou_vox < val_miou_vox:
    #     best_val_miou_vox = val_miou_vox

    # logger.info(
    #     'Current val miou pts is %.3f while the best val miou pts is %.3f' % (val_miou_pts, best_val_miou_pts))
    # logger.info(
    #     'Current val miou vox is %.3f while the best val miou vox is %.3f' % (val_miou_vox, best_val_miou_vox))
    # logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))


def assign_clip_labels(args, class_mapping_clip, loss_func_noW, lovasz_softmax, outputs_pts_fts, text_features,
                       train_grid_fts, train_vox_label_cls, ignore_label=0, compute_loss=True, assignment_only=False):
    # points
    outputs_pts_clip, logits_clip = assign_labels_clip(
        outputs_pts_fts.float().cuda(), text_features, 1, maskclip=args.maskclip,
        class_mapping_clip=class_mapping_clip, ignore_label=0)
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
    train_pt_labs_fts = torch.cat(train_pt_labs_fts_list).unsqueeze(0).cuda()
    non_ignore = train_pt_labs_fts != ignore_label
    targets = (train_pt_labs_fts[non_ignore] - 1).unsqueeze(0)
    outputs_pts_clip = outputs_pts_clip[non_ignore].unsqueeze(0).cuda()

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
    else:
        train_grid_fts_int = train_grid_fts_int[non_ignore].unsqueeze(0)
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
    parser.add_argument('--max-iter', default=None, type=int)
    parser.add_argument('--projected-features', action='store_true')

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
        torch.set_num_threads(1)
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
