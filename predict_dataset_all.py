import argparse
import os
import os.path as osp
import pickle
import time
import warnings
from datetime import datetime

import numba as nb
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmseg.utils import get_root_logger

from dataloader.dataset import get_nuScenes_label_name
from train import assign_clip_labels
from utils.load_save_util import revise_ckpt
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

    save_dir = args.save_dir

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

    # get various configs
    try:
        feature_learning = cfg.feature_learning
    except:
        feature_learning = False

    if feature_learning:
        clip_features = dataset_config['features_type'] == 'clip'
    else:
        clip_features = False

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

    global_rank = dist.get_rank()

    if global_rank != 0:
        import builtins
        builtins.print = pass_print

    logger = get_root_logger(log_file=None, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder

    model = model_builder.build(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        model = ddp_model_module(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = model.cuda()
    print('done ddp model')

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

    unique_label_str_agn = ['empty', 'occupied']
    ignore_label_semantic = 0
    unique_label_cls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    SemKITTI_label_name_cls = get_nuScenes_label_name("./config/label_mapping/nuscenes.yaml")
    unique_label_str_cls = [SemKITTI_label_name_cls[x] for x in unique_label_cls]

    # predict both semantics and occupancy
    CalMeanIou_vox_gtOcc = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'vox_gtOcc')
    CalMeanIou_vox_unq = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'vox_unq')

    CalMeanIou_vox_agn = MeanIoU([0, 1], 255, unique_label_str_agn, 'vox_agn')
    CalMeanIou_vox_agn_unq = MeanIoU([0, 1], 255, unique_label_str_agn, 'vox_agn_unq')
    CalMeanIou_pts_predOcc = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'pts_predOcc')
    CalMeanIou_pts_agn = MeanIoU([0, 1], 255, unique_label_str_agn, 'pts_agn')

    # if we take GT occupancy from LiDAR
    CalMeanIou_vox_gtOcc = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'vox_gtOcc')
    CalMeanIou_vox_predOcc = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'vox_predOcc')
    CalMeanIou_pts_gtOcc = MeanIoU(unique_label_cls, ignore_label_semantic, unique_label_str_cls, 'pts_gtOcc')

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
            embedding_dim = 512
            if text_features.shape[0] == embedding_dim:
                text_features = text_features.T
            text_features = text_features.float().cuda()

        else:
            assert False, "We should use the given prompts!"

    # resume and load
    assert osp.isfile(args.ckpt_path)
    cfg.resume_from = args.ckpt_path
    print('ckpt path:', cfg.resume_from)

    map_location = 'cpu'
    ckpt = torch.load(args.ckpt_path, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(model.load_state_dict(revise_ckpt(ckpt), strict=True))
    print(f'successfully loaded ckpt to *OCCUPANCY* model')

    print_freq = cfg.print_freq

    # eval
    model.eval()
    val_loss_list = []
    CalMeanIou_pts_predOcc.reset()
    CalMeanIou_pts_predOcc.reset()
    CalMeanIou_pts_gtOcc.reset()
    CalMeanIou_vox_gtOcc.reset()
    CalMeanIou_vox_predOcc.reset()
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
            val_grid_float_ft = val_grid_fts_gt.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            vox_label = val_vox_label.type(torch.LongTensor).cuda()
            val_pt_labs = val_pt_labs.cuda()

            if i_iter_val == 0:
                print(f'vox_label.unique(): {vox_label.unique()}')

            # get predictions for the points with projections
            predicted_labels_allVox, predicted_labels_allPts, predicted_features_all_points = model(img=imgs,
                                                                                                    img_metas=img_metas,
                                                                                                    points=val_grid_float.clone(),
                                                                                                    features=val_grid_float.clone())

            # get the occupied voxels in the GT data and their corresponding features
            # 1) Take the unique val_grid_int locations.
            occupied_voxels_loc_gt, idx = torch.unique(val_grid_int,dim=1,return_inverse=True)
            # 2_ Take features at these locations.
            features_gtOccupied_voxels = []
            for _i in range(occupied_voxels_loc_gt.shape[1]):
                _mean_ft = torch.mean(predicted_features_all_points[:,idx==_i],dim=1)
                features_gtOccupied_voxels.append(_mean_ft)
            features_gtOccupied_voxels = torch.cat(features_gtOccupied_voxels).unsqueeze(0)

            # get the occupied voxels; for these voxels, get the feature predictions
            occupied_voxels_loc = torch.stack(torch.where(predicted_labels_allVox[0].argmax(0))).T
            evaluated_voxels = occupied_voxels_loc.unsqueeze(0).float()
            # predict features at those positions
            _, _, predicted_features_occupied_vox = model(img=imgs, img_metas=img_metas,
                                                          points=evaluated_voxels.clone(),
                                                          features=evaluated_voxels.clone())

            # get the GT points that were predicted as occupied
            occupied_points_loc_bool = predicted_labels_allPts[0].argmax(0) == 1
            occupied_points_loc = torch.stack(torch.where(occupied_points_loc_bool)).T
            predicted_features_occupied_points = predicted_features_all_points[:, occupied_points_loc_bool.squeeze()]

            if clip_features:
                predicted_labels_allPts, predicted_labels_visiblePts, predicted_labels_allVox_predOccPts, predicted_labels_allVox_allPts = process_clip(
                    args, class_mapping_clip, evaluated_voxels.long(),
                    predicted_features_occupied_vox, features_gtOccupied_voxels,
                    predicted_features_occupied_points, predicted_features_all_points,
                    text_features, val_grid, val_grid_float,
                    val_grid_float[:, occupied_points_loc_bool.squeeze()],
                    occupied_voxels_loc_gt,
                    val_vox_label_cls,
                    occupied_points_loc_bool.squeeze())
            else:
                # DINO features + classification probe
                used_features = predicted_features_occupied_vox if (feature_learning
                                                                    # or args.linear_probe
                                                                    ) else val_fts
                if feature_learning:
                    assert False, "This case is not covered yet!"
                else:
                    dino_feats = val_fts.permute(0, 2, 1).float().cuda()
                # get predictions
                # don't get gradients since we do not train anything
                with torch.no_grad():
                    # POINTS
                    # a) visible
                    predicted_labels_visiblePts = model_cls_probe(dino_feats).detach().cpu()
                    predicted_labels_visiblePts = predicted_labels_visiblePts.argmax(1).flatten().detach().cpu()
                    predicted_labels_allPts = torch.ones(len(val_pt_labs.squeeze()),
                                                         device=predicted_labels_visiblePts.device,
                                                         dtype=val_pt_labs.dtype) * FREE_LABEL_SEMANTIC
                    predicted_labels_allPts[matching_points[0]] = predicted_labels_visiblePts

                    xyz_pred_grid_loc = val_grid_fts_gt_int[0]
                    label_voxel_pair = np.concatenate([xyz_pred_grid_loc, predicted_labels_visiblePts[..., None]],
                                                      axis=1)
                    label_voxel_pair = label_voxel_pair[
                                       np.lexsort(
                                           (xyz_pred_grid_loc[:, 0], xyz_pred_grid_loc[:, 1],
                                            xyz_pred_grid_loc[:, 2])), :
                                       ]
                    predicted_labels_allVox_predOccPts = np.ones_like(val_vox_label_cls)[0] * FREE_LABEL_SEMANTIC
                    predicted_labels_allVox_predOccPts = torch.from_numpy(
                        nb_process_label(predicted_labels_allVox_predOccPts, label_voxel_pair)).unsqueeze(
                        0).unsqueeze(-1).cuda()

                    predicted_labels_visiblePts = predicted_labels_visiblePts.unsqueeze(0)
                    predicted_labels_allPts = predicted_labels_allPts.unsqueeze(0)

            # we want to evaluate also for class-agnostic occupancy prediction
            # If the predicted labels are already agnostic...
            predict_labels_vox_agn = torch.zeros_like(predicted_labels_allVox_predOccPts)
            predict_labels_vox_agn[predicted_labels_allVox_predOccPts < FREE_LABEL_SEMANTIC] = 1

            val_pt_labs_agn = torch.zeros_like(val_pt_labs, device=val_pt_labs.device)
            val_pt_labs_agn[val_pt_labs < FREE_LABEL_SEMANTIC] = 1
            predict_labels_pts_agn = torch.zeros_like(predicted_labels_visiblePts,
                                                      device=predicted_labels_visiblePts.device)
            # [matching_points[0]].unsqueeze(0)
            predict_labels_pts_agn[predicted_labels_visiblePts < FREE_LABEL_SEMANTIC] = 1
            vox_label_agn = torch.zeros_like(vox_label, device=vox_label.device)
            vox_label_agn[val_vox_label_cls < FREE_LABEL_SEMANTIC] = 1

            if save_dir is not None:
                save_path = os.path.join(save_dir, f'val{i_iter_val}.pkl')
                save_dict = {
                    'predicted_labels_allPts': predicted_labels_allPts.cpu().numpy(),
                    'matching_points_wOcc': matching_points,
                    'predicted_labels_allVox': predicted_labels_allVox_predOccPts.cpu().numpy(),
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(save_dict, f)

            for count in range(len(val_grid_int)):
                # assert False, "Take only points that do have a projection to the camera!"
                # CalMeanIou_pts._after_step(predict_labels_pts_wProj[count], val_pt_labs_wProj.cuda()[count])
                CalMeanIou_pts_predOcc._after_step(predicted_labels_visiblePts[count].squeeze().cuda(),
                                           val_pt_labs_cls.cuda()[count].squeeze())
                CalMeanIou_pts_gtOcc._after_step(predicted_labels_allPts[count].squeeze().cuda(),
                                               val_pt_labs_cls.cuda()[count].squeeze())
                CalMeanIou_pts_agn._after_step(predict_labels_pts_agn[count].squeeze().cuda(),
                                               val_pt_labs_agn.cuda()[count].squeeze())

                CalMeanIou_vox_gtOcc._after_step(predicted_labels_allVox_allPts[
                                               count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                               val_grid_int[count][:, 2]].flatten().cuda(),
                                           val_pt_labs_cls[count].flatten().cuda())
                CalMeanIou_vox_predOcc._after_step(predicted_labels_allVox_predOccPts[
                                               count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                               val_grid_int[count][:, 2]].flatten().cuda(),
                                           val_pt_labs_cls[count].flatten().cuda())
                CalMeanIou_vox_agn._after_step(predict_labels_vox_agn[
                                                   count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                                   val_grid_int[count][:, 2]].flatten().cuda(),
                                               val_pt_labs_agn[count].flatten())

                if args.unique_voxels:
                    locations_for_vox = torch.stack(torch.where(val_vox_label_cls != FREE_LABEL_SEMANTIC)).T
                    cur_indices = locations_for_vox[:, 0] == count
                    CalMeanIou_vox_agn_unq._after_step(predict_labels_vox_agn[
                                                           count, locations_for_vox[cur_indices, 1], locations_for_vox[
                                                               cur_indices, 2], locations_for_vox[
                                                               cur_indices, 3]].flatten().cuda(), vox_label_agn[
                                                           count, locations_for_vox[cur_indices, 1], locations_for_vox[
                                                               cur_indices, 2], locations_for_vox[
                                                               cur_indices, 3]].flatten().cuda())
                    CalMeanIou_vox_unq._after_step(predicted_labels_allVox_predOccPts[
                                                       count, locations_for_vox[cur_indices, 1], locations_for_vox[
                                                           cur_indices, 2], locations_for_vox[
                                                           cur_indices, 3]].flatten().cuda(),
                                                   val_vox_label_cls[count, locations_for_vox[cur_indices, 1],
                                                                     locations_for_vox[cur_indices, 2],
                                                                     locations_for_vox[
                                                                         cur_indices, 3]].flatten().cuda())
            if i_iter_val % print_freq == 0 and global_rank == 0:
                logger.info('[EVAL] Iter %5d:' % (i_iter_val))

    val_miou_pts = CalMeanIou_pts_predOcc._after_epoch()
    val_miou_pts_agn = CalMeanIou_pts_agn._after_epoch()
    val_miou_pts_all = CalMeanIou_pts_gtOcc._after_epoch()

    val_miou_vox_gtOcc = CalMeanIou_vox_gtOcc._after_epoch()
    val_miou_vox_predOcc = CalMeanIou_vox_predOcc._after_epoch()
    val_miou_vox_agn = CalMeanIou_vox_agn._after_epoch()
    val_miou_vox_unq = CalMeanIou_vox_unq._after_epoch()
    val_miou_vox_agn_unq = CalMeanIou_vox_agn_unq._after_epoch()

    logger.info('*** POINTS ***\n')
    logger.info('Current val miou pts with projections is %.3f\n' % (val_miou_pts))
    logger.info('Current val miou pts all (w/o projections are considered empty) is %.3f\n' % (val_miou_pts_all))
    logger.info('Current val miou pts AGNOSTIC is %.3f\n\n' % (val_miou_pts_agn))

    logger.info('*** VOXELS ***\n')
    logger.info('Current val miou vox (classes asigned to GT-occupied points) is %.3f\n' % (val_miou_vox_gtOcc))
    logger.info('Current val miou vox (classes asigned to predicted-occupied points only) is %.3f\n' % (val_miou_vox_predOcc))
    logger.info('Current val miou vox AGNOSTIC is %.3f\n' % (val_miou_vox_agn))
    logger.info('Current val miou vox UNIQUE is %.3f\n' % (val_miou_vox_unq))
    logger.info('Current val miou vox UNIQUE AGNOSTIC is %.3f\n' % (val_miou_vox_agn_unq))
    logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))

    print('\n***************************************************\n')

    if global_rank == 0:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        output_dir = os.path.join('./eval_out', timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(CalMeanIou_pts_predOcc.confusion_matrix.cpu(), os.path.join(output_dir, 'pts_conf.pth'))
        torch.save(CalMeanIou_pts_agn.confusion_matrix.cpu(), os.path.join(output_dir, 'pts_agn_conf.pth'))
        torch.save(CalMeanIou_vox_gtOcc.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_conf.pth'))
        torch.save(CalMeanIou_vox_agn.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_agn_conf.pth'))
        torch.save(CalMeanIou_vox_unq.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_unq_conf.pth'))
        torch.save(CalMeanIou_vox_agn_unq.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_agn_unq_conf.pth'))
        print(f'Saved confusion matrices to {output_dir}')

        eval_t = time.time() - eval_s
        print('Evaluation completed in {:.2f}s'.format(eval_t))
        n_samples = len(val_dataset_loader.dataset)
        eval_t_persample = eval_t / n_samples
        print('Speed was {:.2f}s per sample ({} samples)'.format(eval_t_persample, n_samples))


def process_clip(args, class_mapping_clip, evaluated_voxels,
                 features_evaluated_voxels, features_gtOccupied_voxels,
                 predicted_features_occupied_points, predicted_features_all_points,
                 text_features, val_grid,
                 val_grid_int_unq, val_vox_label_cls,
                 matching_points):
    # for points, we assign a label to every point from LiDAR
    predicted_labels_visiblePts, target_labels_visiblePts = assign_clip_labels(
        args, class_mapping_clip, None, None, predicted_features_occupied_points, text_features,
        None, None, compute_loss=False)[3:5]
    predicted_labels_visiblePts_all = torch.ones(val_grid.shape[1]).long().cuda() * FREE_LABEL_SEMANTIC
    predicted_labels_visiblePts_all[matching_points] = predicted_labels_visiblePts
    predicted_labels_visiblePts = predicted_labels_visiblePts_all

    predicted_labels_allPts, target_labels_allPts = assign_clip_labels(
        args, class_mapping_clip, None, None, predicted_features_all_points, text_features,
        val_grid, None, compute_loss=False)[3:5]

    # for voxels, we do assign labels to only such locations, that we predicted as "occupied"
    # therefore, we initialize the grid with "empty" labels
    _predicted_labels_allVox_predOcc = assign_clip_labels(
        args, class_mapping_clip, None, None, features_evaluated_voxels, text_features,
        None, None, compute_loss=False)[3]
    predicted_labels_allVox_predOcc = torch.ones_like(val_vox_label_cls)[0] * FREE_LABEL_SEMANTIC
    predicted_labels_allVox_predOcc[
        evaluated_voxels[0, :, 0], evaluated_voxels[0, :, 1], evaluated_voxels[0, :, 2]
    ] = _predicted_labels_allVox_predOcc.cpu().squeeze()
    # If we take the GT from LiDAR
    _predicted_labels_allVox_allPts = assign_clip_labels(
        args, class_mapping_clip, None, None, features_gtOccupied_voxels, text_features,
        None, None, compute_loss=False)[3]
    predicted_labels_allVox_allPts = torch.ones_like(val_vox_label_cls)[0] * FREE_LABEL_SEMANTIC
    predicted_labels_allVox_allPts[val_grid_int_unq[0, :, 0], val_grid_int_unq[0, :, 1],
                                   val_grid_int_unq[0, :, 2]] = _predicted_labels_allVox_allPts.cpu()

    predicted_labels_visiblePts = predicted_labels_visiblePts.unsqueeze(0)
    predicted_labels_allVox_allPts = predicted_labels_allVox_allPts.unsqueeze(0)
    predicted_labels_allVox_predOcc = predicted_labels_allVox_predOcc.unsqueeze(0)

    return predicted_labels_allPts, predicted_labels_visiblePts, predicted_labels_allVox_predOcc, predicted_labels_allVox_allPts


def get_dino_preds_and_targets(args, predict_occ_vox, features_pts, val_grid_fts_gt_int, val_vox_label_cls,
                               matching_points):
    dino_feats_list = []
    dino_targets_list = []
    xyz_pred_occ = []
    xyz_pred_idx = []
    xyz_gt_occ = []  # (X,Y,Z) locations in the voxel grid
    gt_idx = []
    matching_points_wOcc = []
    for bi in range(predict_occ_vox.shape[0]):
        # 1) get locations of GT occupied voxels
        if args.unique_features:
            # we take only a single feature per voxel
            occupied_voxels_gt, unq_fts = val_grid_fts_gt_int[bi].unique(dim=0, return_inverse=True)
        else:
            # we take features belonging to all the points
            occupied_voxels_gt = val_grid_fts_gt_int[bi]
        xyz_gt_occ.append(occupied_voxels_gt)

        # 2) get the predicted occupancy at the positions of GT-occupied voxels
        if args.projected_features:
            # take all the GT-occupied voxels, i.e., consider everything predicted correctly
            occupied_voxels_pred_occ = occupied_voxels_gt
        else:
            occupied_voxels_pred = predict_occ_vox.argmax(1)[
                bi, occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1], occupied_voxels_gt[:, 2]
            ]
            # get locations where it is predicted correctly
            occupied_voxels_pred_occ_bool = occupied_voxels_pred == 1
            xyz_pred_idx.extend(occupied_voxels_pred_occ_bool)
            occupied_voxels_pred_occ = torch.where(occupied_voxels_pred_occ_bool)[0]

        # get the location of the correctly predicted occupied voxels
        xyz_pred_occ.append(occupied_voxels_gt[occupied_voxels_pred_occ])
        matching_points_wOcc.append(matching_points[bi][occupied_voxels_pred_occ.cpu()])
        # get the GT DINO features at locations of correctly predicted occupied voxels
        dino_feats = features_pts[bi][occupied_voxels_pred_occ]
        dino_feats_list.append(dino_feats)
        gt_idx.append(val_vox_label_cls[bi][occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1],
                                            occupied_voxels_gt[:, 2]])
        dino_targets_list.append(val_vox_label_cls[bi][occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1],
                                                       occupied_voxels_gt[:, 2]][
                                     occupied_voxels_pred_occ])
    dino_feats = torch.cat(dino_feats_list).T.unsqueeze(0).cuda().float()
    # dino_targets = torch.cat(dino_targets_list).unsqueeze(0).cuda()
    xyz_pred_idx = torch.tensor(xyz_pred_idx)
    matching_points_wOcc = np.concatenate(matching_points_wOcc)
    return dino_feats, xyz_pred_idx, matching_points_wOcc


if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--ckpt-path', type=str, default='')
    parser.add_argument('--unique-voxels', action='store_true')
    parser.add_argument('--max-iter', type=int, default=None)
    parser.add_argument('--agnostic', action='store_true')
    parser.add_argument('--unique-features', action='store_true')
    parser.add_argument('--linear-probe', action='store_true')
    parser.add_argument('--projected-features', action='store_true')
    parser.add_argument('--weights-path-linear', type=str, default=None)
    parser.add_argument('--num-workers', default=None, type=int)
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--text-embeddings-path', default=None, type=str)
    parser.add_argument('--maskclip', action='store_true', default=True)

    args = parser.parse_args()
    if args.weights_path_linear is not None and args.weights_path_linear.lower() == 'none':
        args.weights_path_linear = None

    if args.save_dir is not None:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        args.save_dir += f'__{timestamp}'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        print(f'Save predictions to {args.save_dir}')

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
