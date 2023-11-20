import argparse
import os
import os.path as osp
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmseg.utils import get_root_logger

from builder import loss_builder
from dataloader.dataset import get_nuScenes_label_name
from utils.load_save_util import revise_ckpt
from utils.metric_util import MeanIoU

warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


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
            scale_rate=cfg.get('scale_rate', 1)
        )

    # get optimizer, loss, scheduler
    loss_func, lovasz_softmax = \
        loss_builder.build(
            ignore_label=ignore_label)

    unique_label_str_agn = ['empty', 'occupied']
    CalMeanIou_vox_all = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox_allPts')
    CalMeanIou_vox_unq = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox_unq')
    CalMeanIou_vox_unq_visible = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox_unq_visible')
    CalMeanIou_vox_agn_all = MeanIoU([0, 1], 255, unique_label_str_agn, 'vox_agn_allPts')
    CalMeanIou_vox_agn_unq = MeanIoU([0, 1], 255, unique_label_str_agn, 'vox_agn_unq')
    CalMeanIou_pts = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts')
    CalMeanIou_pts_visible = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts_visible')
    CalMeanIou_pts_agn = MeanIoU([0, 1], 255, unique_label_str_agn, 'pts_agn')

    # resume and load
    assert osp.isfile(args.ckpt_path)
    cfg.resume_from = args.ckpt_path
    print('ckpt path:', cfg.resume_from)

    map_location = 'cpu'
    ckpt = torch.load(cfg.resume_from, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(my_model.load_state_dict(revise_ckpt(ckpt), strict=False))
    print(f'successfully loaded ckpt')

    print_freq = cfg.print_freq

    # eval
    my_model.eval()
    val_loss_list = []
    CalMeanIou_pts.reset()
    CalMeanIou_vox_all.reset()
    CalMeanIou_vox_unq.reset()
    CalMeanIou_pts_agn.reset()
    CalMeanIou_vox_agn_all.reset()
    CalMeanIou_vox_agn_unq.reset()
    CalMeanIou_vox_unq_visible.reset()
    CalMeanIou_pts_visible.reset()

    with torch.no_grad():
        for i_iter_val, loaded_data in enumerate(val_dataset_loader):

            imgs, img_metas, val_vox_label, val_grid, val_pt_labs, *_ = loaded_data

            if args.max_iter is not None and i_iter_val == args.max_iter:
                break

            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            vox_label = val_vox_label.type(torch.LongTensor).cuda()
            val_pt_labs = val_pt_labs.cuda()

            if i_iter_val == 0:
                print(f'vox_label.unique(): {vox_label.unique()}')

            predict_labels_vox, predict_labels_pts, *_ = my_model(img=imgs, img_metas=img_metas, points=val_grid_float)
            if cfg.lovasz_input == 'voxel':
                lovasz_input = predict_labels_vox
                lovasz_label = vox_label
            else:
                lovasz_input = predict_labels_pts
                lovasz_label = val_pt_labs

            if cfg.ce_input == 'voxel':
                ce_input = predict_labels_vox
                ce_label = vox_label
            else:
                ce_input = predict_labels_pts.squeeze(-1).squeeze(-1)
                ce_label = val_pt_labs.squeeze(-1)

            # loss = lovasz_softmax(
            #     torch.nn.functional.softmax(lovasz_input, dim=1).detach(),
            #     lovasz_label, ignore=ignore_label
            # ) + loss_func(ce_input.detach(), ce_label)
            loss = torch.tensor(-1)

            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1)  # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs = val_pt_labs.squeeze(-1).cpu()

            predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
            predict_labels_vox = predict_labels_vox.detach().cpu()
            if i_iter_val == 0:
                print(f'predict_labels_vox.unique(): {predict_labels_vox.unique()}')

            locations_for_vox_all = val_grid_int
            locations_for_vox_all_unique = torch.unique(val_grid_int, dim=1)
            locations_for_vox_visible = torch.stack(torch.where(vox_label != cfg.dataset_params['fill_label'])).T
            locations_for_vox_visible_unique = torch.unique(locations_for_vox_visible, dim=1)

            predict_labels_vox_agn = torch.zeros_like(predict_labels_vox, device=predict_labels_vox.device)
            predict_labels_vox_agn[predict_labels_vox != cfg.dataset_params['fill_label']] = 1

            val_pt_labs_agn = torch.ones_like(val_pt_labs, device=val_pt_labs.device)

            predict_labels_pts_agn = torch.ones_like(predict_labels_pts, device=predict_labels_pts.device)
            predict_labels_pts_agn[predict_labels_pts_agn == cfg.dataset_params['fill_label']] = 0

            vox_label_agn = torch.zeros_like(vox_label, device=vox_label.device)
            vox_label_agn[vox_label != cfg.dataset_params['fill_label']] = 1

            for count in range(len(val_grid_int)):
                CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count])
                CalMeanIou_pts_agn._after_step(predict_labels_pts_agn[count], val_pt_labs_agn[count])

                CalMeanIou_vox_all._after_step(predict_labels_vox[
                                                   count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                                   val_grid_int[count][:, 2]].flatten(), val_pt_labs[count])
                CalMeanIou_vox_agn_all._after_step(predict_labels_vox_agn[
                                                       count, val_grid_int[count][:, 0], val_grid_int[count][:, 1],
                                                       val_grid_int[count][:, 2]].flatten(), val_pt_labs_agn[count])

                if args.eval_visible:
                    # visible points
                    assert False, "Implement!!!"
                    # visible voxels
                    CalMeanIou_vox_unq_visible._after_step(
                        predict_labels_vox[count, locations_for_vox_visible[cur_indices, 1],
                                           locations_for_vox_visible[cur_indices, 2],
                                           locations_for_vox_visible[cur_indices, 3]].flatten().cuda(),
                        vox_label[count, locations_for_vox_visible[cur_indices, 1],
                                  locations_for_vox_visible[cur_indices, 2],
                                  locations_for_vox_visible[cur_indices, 3]].flatten().cuda()
                    )

                if args.unique_voxels:
                    cur_indices = locations_for_vox_all[:, 0] == count
                    # unique and agnostic
                    CalMeanIou_vox_agn_unq._after_step(
                        predict_labels_vox_agn[count, locations_for_vox_all_unique[cur_indices, 1],
                                               locations_for_vox_all_unique[cur_indices, 2],
                                               locations_for_vox_all_unique[cur_indices, 3]].flatten().cuda(),
                        vox_label_agn[count, locations_for_vox_all_unique[cur_indices, 1],
                                      locations_for_vox_all_unique[cur_indices, 2],
                                      locations_for_vox_all_unique[cur_indices, 3]].flatten().cuda()
                    )
                    # unique
                    CalMeanIou_vox_unq._after_step(
                        predict_labels_vox[count, locations_for_vox_all_unique[cur_indices, 1],
                                           locations_for_vox_all_unique[cur_indices, 2],
                                           locations_for_vox_all_unique[cur_indices, 3]].flatten().cuda(),
                        vox_label[count, locations_for_vox_all_unique[cur_indices, 1],
                                  locations_for_vox_all_unique[cur_indices, 2],
                                  locations_for_vox_all_unique[cur_indices, 3]].flatten().cuda()
                    )
                    if args.eval_visible:
                        # unique and visible
                        CalMeanIou_vox_unq_visible._after_step(
                            predict_labels_vox[count, locations_for_vox_visible_unique[cur_indices, 1],
                                               locations_for_vox_visible_unique[cur_indices, 2],
                                               locations_for_vox_visible_unique[cur_indices, 3]].flatten().cuda(),
                            vox_label[count, locations_for_vox_visible_unique[cur_indices, 1],
                                      locations_for_vox_visible_unique[cur_indices, 2],
                                      locations_for_vox_visible_unique[cur_indices, 3]].flatten().cuda()
                        )
            val_loss_list.append(loss.detach().cpu().numpy())
            if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                logger.info('[EVAL] Iter %5d: Loss: %.3f (%.3f)' % (
                    i_iter_val, loss.item(), np.mean(val_loss_list)))

    val_miou_pts = CalMeanIou_pts._after_epoch()
    val_miou_pts_agn = CalMeanIou_pts_agn._after_epoch()
    val_miou_vox = CalMeanIou_vox_all._after_epoch()
    val_miou_vox_agn = CalMeanIou_vox_agn_all._after_epoch()
    val_miou_vox_unq = CalMeanIou_vox_unq._after_epoch()
    val_miou_vox_agn_unq = CalMeanIou_vox_agn_unq._after_epoch()

    if args.eval_visible:
        val_miou_vox_unq_visible = CalMeanIou_vox_unq_visible._after_epoch()
        val_miou_pts_visible = CalMeanIou_pts_visible._after_epoch()

    logger.info('Current val miou pts is %.3f\n' % (val_miou_pts))
    logger.info('Current val miou vox is %.3f\n' % (val_miou_vox))
    logger.info('Current val miou pts AGNOSTIC is %.3f\n' % (val_miou_pts_agn))
    logger.info('Current val miou vox AGNOSTIC is %.3f\n' % (val_miou_vox_agn))
    logger.info('Current val miou vox UNIQUE is %.3f\n' % (val_miou_vox_unq))
    logger.info('Current val miou vox UNIQUE AGNOSTIC is %.3f\n' % (val_miou_vox_agn_unq))

    if args.eval_visible:
        logger.info('Current val miou vox UNIQUE VISIBLE is %.3f\n' % (val_miou_vox_unq_visible))
        logger.info('Current val miou points VISIBLE is %.3f\n' % (val_miou_pts_visible))

    logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = os.path.join('./eval_out', timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(CalMeanIou_pts.confusion_matrix.cpu(), os.path.join(output_dir, 'pts_conf.pth'))
    torch.save(CalMeanIou_pts_agn.confusion_matrix.cpu(), os.path.join(output_dir, 'pts_agn_conf.pth'))
    torch.save(CalMeanIou_vox_all.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_conf.pth'))
    torch.save(CalMeanIou_vox_agn_all.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_agn_conf.pth'))
    torch.save(CalMeanIou_vox_unq.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_unq_conf.pth'))
    torch.save(CalMeanIou_vox_agn_unq.confusion_matrix.cpu(), os.path.join(output_dir, 'vox_agn_unq_conf.pth'))
    print(f'Saved confusion matrices to {output_dir}')


if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--ckpt-path', type=str, default='')
    parser.add_argument('--unique-voxels', action='store_true')
    parser.add_argument('--max-iter', type=int, default=None)
    parser.add_argument('--agnostic', action='store_true')
    parser.add_argument('--eval-visible', action='store_true')

    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
