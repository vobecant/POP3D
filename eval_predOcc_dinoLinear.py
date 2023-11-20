import argparse
import copy
import os
import os.path as osp
import time
import warnings
from datetime import datetime

import numba as nb
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmseg.utils import get_root_logger

from builder import loss_builder
from dataloader.dataset import get_nuScenes_label_name
from linear_probe import build_linear_probe
from utils.load_save_util import revise_ckpt, revise_ckpt_linear_probe
from utils.metric_util import MeanIoU

warnings.filterwarnings("ignore")

FREE_LABEL_SEMANTIC = 17


def pass_print(*args, **kwargs):
    pass


@nb.jit('int64[:,:,:](int64[:,:,:],int64[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)

    # check label_mapping, fill_label, ignore_label, pc_dataset_type
    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    # check num_workers, imageset
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    grid_size = cfg.grid_size

    # init DDP
    distributed = True
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "20506")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    print(f"tcp://{ip}:{port}")
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}",
        world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    world_size = dist.get_world_size()
    cfg.gpu_ids = range(world_size)
    torch.cuda.set_device(local_rank)

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    logger = get_root_logger(log_file=None, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build linear probe
    num_epochs = cfg['max_epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_linear_probe(cfg['model_params']).to(device)
    for param in model.parameters():
        param.requires_grad = False
    n_parameters_lin = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of trainable params of the linear probe: {n_parameters_lin}')
    if distributed and n_parameters_lin > 0:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        model = ddp_model_module(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = model.cuda()

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder

    model_occ = model_builder.build(cfg.model)
    n_parameters = sum(p.numel() for p in model_occ.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        model_occ = ddp_model_module(
            model_occ.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model_occ = model_occ.cuda()
    print('done ddp model')

    model_occ.eval()
    model.eval()

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

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
            unique_features=args.unique_features,
            num_workers=args.num_workers,
            semantic_points=True
        )

    # get optimizer, loss, scheduler
    loss_func, lovasz_softmax = \
        loss_builder.build(
            ignore_label=ignore_label)

    unique_label_str_agn = ['empty', 'occupied']
    ignore_label_semantic = 0
    unique_label_cls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    SemKITTI_label_name_cls = get_nuScenes_label_name("./config/label_mapping/nuscenes.yaml")
    unique_label_str_cls = [SemKITTI_label_name_cls[x] for x in unique_label_cls]

    CalMeanIou_vox = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'vox')
    CalMeanIou_vox_unq = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'vox_unq')
    CalMeanIou_vox_agn = MeanIoU([0, 1], 255, unique_label_str_agn, 'vox_agn')
    CalMeanIou_vox_agn_unq = MeanIoU([0, 1], 255, unique_label_str_agn, 'vox_agn_unq')
    CalMeanIou_pts = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'pts')
    CalMeanIou_pts_agn = MeanIoU([0, 1], 255, unique_label_str_agn, 'pts_agn')

    # resume and load
    assert osp.isfile(args.ckpt_path)
    cfg.resume_from = args.ckpt_path
    print('ckpt path:', cfg.resume_from)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_linear_probe(cfg['model_params']).to(device)
    if args.weights_path_linear is not None and os.path.isfile(args.weights_path_linear):
        print('resume from: ', args.ckpt_path)
        map_location = 'cpu'
        ckpt = torch.load(args.weights_path_linear, map_location=map_location)
        print('Loading segmentation model:')
        print(model.load_state_dict(
            revise_ckpt_linear_probe(ckpt['state_dict'], ddp=distributed and n_parameters_lin > 0), strict=True))
    else:
        raise Exception(f'You must give a valid path to the linear model! You gave "{args.weights_path_linear}"!')

    map_location = 'cpu'
    ckpt = torch.load(args.ckpt_path, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(model_occ.load_state_dict(revise_ckpt(ckpt), strict=True))
    print(f'successfully loaded ckpt to *OCCUPANCY* model')

    print_freq = cfg.print_freq

    # eval
    model_occ.eval()
    val_loss_list = []
    CalMeanIou_pts.reset()
    CalMeanIou_vox.reset()
    CalMeanIou_vox_unq.reset()
    CalMeanIou_pts_agn.reset()
    CalMeanIou_vox_agn.reset()
    CalMeanIou_vox_agn_unq.reset()

    eval_s = time.time()
    with torch.no_grad():
        for i_iter_val, loaded_data in enumerate(val_dataset_loader):

            imgs, img_metas, val_vox_label, val_grid, val_pt_labs, val_vox_label_cls, val_grid_fts_gt, val_fts, matching_points, _, val_pt_labs_cls = loaded_data
            val_grid_fts_gt_int = val_grid_fts_gt.long()

            if args.max_iter is not None and i_iter_val == args.max_iter:
                break

            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            vox_label = val_vox_label.type(torch.LongTensor).cuda()
            val_pt_labs = val_pt_labs.cuda()

            if i_iter_val == 0:
                print(f'vox_label.unique(): {vox_label.unique()}')

            predict_occ_vox, predict_occ_pts, *_ = model_occ(img=imgs, img_metas=img_metas, points=val_grid_float)

            dino_feats_list = []
            dino_targets_list = []
            xyz_pred_occ = []
            xyz_pred_idx = []
            xyz_gt_occ = []
            gt_idx = []
            for bi in range(predict_occ_vox.shape[0]):
                if args.unique_features:
                    occupied_voxels_gt, unq_fts = val_grid_fts_gt_int[bi].unique(dim=0, return_inverse=True)
                else:
                    occupied_voxels_gt = val_grid_fts_gt_int[bi]
                xyz_gt_occ.append(occupied_voxels_gt)
                # get the predicted occupancy at the positions of GT-occupied voxels
                occupied_voxels_pred = predict_occ_vox.argmax(1)[
                    bi, occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1], occupied_voxels_gt[:, 2]
                ]
                # get locations where it is predicted correctly
                occupied_voxels_pred_occ_bool = occupied_voxels_pred == 1
                xyz_pred_idx.extend(occupied_voxels_pred_occ_bool)
                occupied_voxels_pred_occ = torch.where(occupied_voxels_pred_occ_bool)[0]
                # get the location of the correctly predicted occupied voxels
                xyz_pred_occ.append(occupied_voxels_gt[occupied_voxels_pred_occ])
                # get the GT DINO features at locations of correctly predicted occupied voxels
                dino_feats = val_fts[bi][occupied_voxels_pred_occ]
                dino_feats_list.append(dino_feats)
                gt_idx.append(val_vox_label_cls[bi][occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1],
                                                    occupied_voxels_gt[:, 2]])
                dino_targets_list.append(val_vox_label_cls[bi][occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1],
                                                               occupied_voxels_gt[:, 2]][
                                             occupied_voxels_pred_occ])
            dino_feats = torch.cat(dino_feats_list).T.unsqueeze(0).cuda().float()
            dino_targets = torch.cat(dino_targets_list).unsqueeze(0).cuda()
            xyz_pred_idx = torch.tensor(xyz_pred_idx)

            # get predictions
            # don't get gradients since we do not train anything
            with torch.no_grad():
                prediction = model(dino_feats)

            predicted_idx_occupied = prediction.argmax(1).flatten().detach().cpu()

            full_targets = val_vox_label_cls[:, val_grid_fts_gt_int[..., 0], val_grid_fts_gt_int[..., 1],
                           val_grid_fts_gt_int[..., 2]].flatten()
            predicted_idx = torch.ones_like(full_targets) * FREE_LABEL_SEMANTIC  # initialize with "empty" predictions

            predicted_idx[xyz_pred_idx] = predicted_idx_occupied

            xyz_pred_grid_loc = val_grid_fts_gt_int[0, xyz_pred_idx]

            label_voxel_pair = np.concatenate([xyz_pred_grid_loc, predicted_idx_occupied[..., None]], axis=1)
            label_voxel_pair = label_voxel_pair[
                               np.lexsort((xyz_pred_grid_loc[:, 0], xyz_pred_grid_loc[:, 1], xyz_pred_grid_loc[:, 2])),
                               :
                               ]

            predict_labels_vox = np.ones_like(val_vox_label_cls)[0] * FREE_LABEL_SEMANTIC
            predict_labels_vox = torch.from_numpy(nb_process_label(predict_labels_vox, label_voxel_pair)).unsqueeze(
                0).unsqueeze(-1).cuda()

            val_pt_labs_wProj = val_pt_labs_cls[0, matching_points[0]].squeeze().unsqueeze(0)
            val_pt_labs_wProj_agn = val_pt_labs[0, matching_points[0]].squeeze().unsqueeze(0)
            # initialize with "empty" predictions
            predict_labels_pts = torch.ones_like(val_pt_labs_wProj).cuda() * FREE_LABEL_SEMANTIC
            predict_labels_pts[:,xyz_pred_idx] = predicted_idx_occupied.cuda()

            if cfg.lovasz_input == 'voxel':
                lovasz_input = predict_occ_vox
                lovasz_label = vox_label
            else:
                lovasz_input = predict_labels_pts
                lovasz_label = val_pt_labs

            if cfg.ce_input == 'voxel':
                ce_input = predict_labels_vox
                ce_label = vox_label
            else:
                ce_input = predict_occ_pts.squeeze(-1).squeeze(-1)
                ce_label = val_pt_labs.squeeze(-1)

            # loss = lovasz_softmax(
            #     torch.nn.functional.softmax(lovasz_input, dim=1).detach(),
            #     lovasz_label, ignore=ignore_label
            # ) + loss_func(ce_input.detach(), ce_label)
            loss = -1

            locations_for_vox = val_grid_int if not args.unique_voxels else torch.stack(
                torch.where(val_vox_label_cls != FREE_LABEL_SEMANTIC)).T

            if args.agnostic:
                predict_labels_vox_agn = copy.deepcopy(predict_occ_vox.argmax(1))
                val_pt_labs_agn = torch.zeros_like(val_pt_labs, device=val_pt_labs.device)
                val_pt_labs_agn[val_pt_labs < FREE_LABEL_SEMANTIC] = 1
                predict_labels_pts_agn = copy.deepcopy(predict_occ_pts.argmax(1)).squeeze()[matching_points[0]].unsqueeze(0)
                vox_label_agn = torch.zeros_like(vox_label, device=vox_label.device)
                vox_label_agn[val_vox_label_cls < FREE_LABEL_SEMANTIC] = 1
            else:
                predict_labels_vox_agn = torch.zeros_like(predict_labels_vox, device=predict_labels_vox.device)
                predict_labels_vox_agn[predict_labels_vox != FREE_LABEL_SEMANTIC] = 1

                val_pt_labs_agn = torch.ones_like(val_pt_labs, device=val_pt_labs.device)

                predict_labels_pts_agn = torch.ones_like(predict_labels_pts, device=predict_labels_pts.device)
                predict_labels_pts_agn[predict_labels_pts == FREE_LABEL_SEMANTIC] = 0

                vox_label_agn = torch.zeros_like(vox_label, device=vox_label.device)
                vox_label_agn[val_vox_label_cls != FREE_LABEL_SEMANTIC] = 1

            for count in range(len(val_grid_int)):
                # assert False, "Take only points that do have a projection to the camera!"
                CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs_wProj.cuda()[count])
                CalMeanIou_pts_agn._after_step(predict_labels_pts_agn[count].squeeze(), val_pt_labs_wProj_agn.cuda()[count])

                CalMeanIou_vox._after_step(predict_labels_vox[
                                               count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                               val_grid_int[count][:, 2]].flatten(),
                                           val_pt_labs_cls[count].flatten().cuda())
                CalMeanIou_vox_agn._after_step(predict_labels_vox_agn[
                                                   count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                                   val_grid_int[count][:, 2]].flatten(),
                                               val_pt_labs_agn[count].flatten())

                if args.unique_voxels:
                    cur_indices = locations_for_vox[:, 0] == count
                    CalMeanIou_vox_agn_unq._after_step(predict_labels_vox_agn[
                                                           count, locations_for_vox[cur_indices, 1], locations_for_vox[
                                                               cur_indices, 2], locations_for_vox[
                                                               cur_indices, 3]].flatten().cuda(), vox_label_agn[
                                                           count, locations_for_vox[cur_indices, 1], locations_for_vox[
                                                               cur_indices, 2], locations_for_vox[
                                                               cur_indices, 3]].flatten().cuda())
                    CalMeanIou_vox_unq._after_step(predict_labels_vox[
                                                       count, locations_for_vox[cur_indices, 1], locations_for_vox[
                                                           cur_indices, 2], locations_for_vox[
                                                           cur_indices, 3]].flatten().cuda(),
                                                   val_vox_label_cls[count, locations_for_vox[cur_indices, 1],
                                                                     locations_for_vox[cur_indices, 2],
                                                                     locations_for_vox[cur_indices, 3]].flatten().cuda())
            # val_loss_list.append(loss.detach().cpu().numpy())
            val_loss_list.append(loss)
            if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                logger.info('[EVAL] Iter %5d: Loss: %.3f (%.3f)' % (
                    i_iter_val,
                    # loss.item(),
                    loss,
                    np.mean(val_loss_list)))

    val_miou_pts = CalMeanIou_pts._after_epoch()
    val_miou_pts_agn = CalMeanIou_pts_agn._after_epoch()
    val_miou_vox = CalMeanIou_vox._after_epoch()
    val_miou_vox_agn = CalMeanIou_vox_agn._after_epoch()
    val_miou_vox_unq = CalMeanIou_vox_unq._after_epoch()
    val_miou_vox_agn_unq = CalMeanIou_vox_agn_unq._after_epoch()

    logger.info('Current val miou pts is %.3f\n' % (val_miou_pts))
    logger.info('Current val miou vox is %.3f\n' % (val_miou_vox))
    logger.info('Current val miou pts AGNOSTIC is %.3f\n' % (val_miou_pts_agn))
    logger.info('Current val miou vox AGNOSTIC is %.3f\n' % (val_miou_vox_agn))
    logger.info('Current val miou vox UNIQUE is %.3f\n' % (val_miou_vox_unq))
    logger.info('Current val miou vox UNIQUE AGNOSTIC is %.3f\n' % (val_miou_vox_agn_unq))
    logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))

    print('\n***************************************************\n')


    if dist.get_rank() == 0:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        output_dir = os.path.join('./eval_out', timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(CalMeanIou_pts.confusion_matrix.cpu(), os.path.join(output_dir, 'pts_conf.pth'))
        torch.save(CalMeanIou_pts_agn.confusion_matrix.cpu(), os.path.join(output_dir, 'pts_agn_conf.pth'))
        torch.save(CalMeanIou_vox.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_conf.pth'))
        torch.save(CalMeanIou_vox_agn.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_agn_conf.pth'))
        torch.save(CalMeanIou_vox_unq.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_unq_conf.pth'))
        torch.save(CalMeanIou_vox_agn_unq.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_agn_unq_conf.pth'))
        print(f'Saved confusion matrices to {output_dir}')

        eval_t = time.time() - eval_s
        print('Evaluation completed in {:.2f}s'.format(eval_t))
        n_samples = len(val_dataset_loader.dataset)
        eval_t_persample = eval_t / n_samples
        print('Speed was {:.2f}s per sample ({} samples)'.format(eval_t_persample, n_samples))


if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--ckpt-path', type=str, default='')
    parser.add_argument('--unique-voxels', action='store_true')
    parser.add_argument('--max-iter', type=int, default=None)
    parser.add_argument('--agnostic', action='store_true')
    parser.add_argument('--unique-features', action='store_true')
    parser.add_argument('--weights-path-linear', type=str, default=None)
    parser.add_argument('--num-workers', default=None, type=int)

    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
