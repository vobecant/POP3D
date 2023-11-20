import argparse
import copy
import os
import socket
import time
from datetime import datetime
from math import inf

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.distributed as dist
import wandb
from mmcv import Config
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger
from timm.scheduler import CosineLRScheduler
from torch import nn
from tqdm import tqdm

from builder import loss_builder
from dataloader.dataset import get_nuScenes_label_name
from train import pass_print
from utils.load_save_util import revise_ckpt, revise_ckpt_linear_probe
from utils.metric_util import MeanIoU
from visualization.training import show3d


def linear_probe_gt_benchmark_bs(args, cfg, use_lovasz, batch_sizes):
    num_epochs = 1

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join('./linear_probing', f'{timestamp}_benchmark_bs.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    print(f'Log to {log_file}')
    print_freq = 50  # cfg.print_freq

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader
    grid_size = cfg.grid_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_linear_probe(cfg['model_params']).to(device)
    if cfg.ckpt_path is not None and os.path.isfile(cfg.ckpt_path):
        print('resume from: ', cfg.ckpt_path)
        map_location = 'cpu'
        ckpt = torch.load(cfg.ckpt_path, map_location=map_location)
        print(model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))

    loss_func, lovasz_softmax = loss_builder.build(ignore_label=ignore_label)

    timings = {}

    for bs in batch_sizes:

        run_start = time.time()

        train_dataloader_config_cur = copy.deepcopy(train_dataloader_config)
        train_dataloader_config_cur["batch_size"] = bs

        from builder import data_builder
        train_dataset_loader, val_dataset_loader = \
            data_builder.build(
                dataset_config,
                train_dataloader_config_cur,
                val_dataloader_config,
                grid_size=grid_size,
                version=version,
                scale_rate=cfg.get('scale_rate', 1),
                linear_probe=True
            )

        # get optimizer, loss, scheduler
        optimizer = build_optimizer(model, cfg.optimizer)
        num_iters = len(train_dataset_loader) * num_epochs
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_iters,
            lr_min=1e-6,
            warmup_t=min(500, int(0.1 * num_iters)),
            warmup_lr_init=1e-5,
            t_in_epochs=False
        )

        global_iter = 0

        # generate datasets
        SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
        unique_label = np.asarray(cfg.unique_label)
        unique_label_str = [SemKITTI_label_name[x] for x in unique_label] if len(unique_label) > 2 else ['empty',
                                                                                                         'occupied']
        CalMeanIou_pts_train = MeanIoU(unique_label, ignore_label, unique_label_str, 'train')

        for epoch in range(num_epochs):
            CalMeanIou_pts_train.reset()

            # train
            model.train()
            # preds = gt = []
            loss_list = []
            load_s = time.time()
            for i_iter, (dino_feats, gt_labels) in enumerate(tqdm(train_dataset_loader)):
                load_t = time.time() - load_s
                # move inputs to device
                dino_feats = dino_feats.to(device)
                gt_labels = gt_labels.to(device)

                # predict
                fwd_s = time.time()
                prediction = model(dino_feats)
                fwd_t = time.time() - fwd_s

                # loss
                loss_ce = loss_func(prediction, gt_labels)
                loss_lovasz = 0
                if use_lovasz:
                    loss_lovasz = lovasz_softmax(
                        torch.nn.functional.softmax(prediction, dim=1).squeeze(),
                        gt_labels.squeeze(), ignore=ignore_label
                    )
                loss = loss_ce + loss_lovasz

                # backprop
                bwd_s = time.time()
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
                optimizer.step()
                loss_list.append(loss.item())
                scheduler.step_update(global_iter)
                bwd_t = time.time() - bwd_s

                predicted_idx = prediction.argmax(1).flatten().detach().cpu()
                gt_idx = gt_labels.cpu().squeeze()

                CalMeanIou_pts_train._after_step(predicted_idx, gt_idx)

                lr = optimizer.param_groups[0]['lr']

                if i_iter % print_freq == 0:
                    loss_detail = []
                    if use_lovasz is not None:
                        loss_detail.append('Lovasz {:.3f}'.format(loss_lovasz))
                    loss_detail.append('CE {:.2f}'.format(loss_ce))
                    loss_detail = ', '.join(loss_detail)
                    logger.info(
                        '[TRAIN, BS={%s}] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f) (%s), grad_norm: %.1f, lr: %.7f, '
                        'time: fwd %.3f bwd %.3f data %.3f' % (
                            bs, epoch, i_iter, len(train_dataset_loader), loss.item(), np.mean(loss_list), loss_detail,
                            grad_norm, lr, fwd_t, bwd_t, load_t
                        ))

                load_s = time.time()
                global_iter += 1

        run_t = time.time() - run_start
        timings[bs] = run_t

    keys = list(timings.keys())
    vals = list(timings.values())

    print('Timings:')
    print(timings)

    vals_str = ';'.join([str(v) for v in vals])
    print('Values: {}\n'.format(vals_str))

    print('\nTimings sorted:')
    idx = np.argsort(vals)
    keys_sorted = [keys[i] for i in idx]
    vals_sorted = [vals[i] for i in idx]
    print_log = '\n '.join([f'{k}:{v}' for k, v in zip(keys_sorted, vals_sorted)])
    print(print_log)


def linear_probe_gt(local_rank, args, cfg, use_lovasz):
    # global settings
    torch.backends.cudnn.benchmark = True

    print(f'socket.gethostname(): {socket.gethostname()}')

    # init DDP
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

    log_wandb = global_rank == 0 and not args.no_wandb
    if log_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="TPVFormer-Open",
            # set the wandb run name
            name=args.name,
            # track hyperparameters and run metadata
            config=cfg
        )

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    # configure logger
    if dist.get_rank() == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(os.path.join(args.work_dir, os.path.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join('./linear_probing', f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    print(f'Log to {log_file}')
    print_freq = cfg.print_freq

    num_epochs = cfg['max_epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_linear_probe(cfg['model_params']).to(device)
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

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
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
            scale_rate=cfg.get('scale_rate', 1),
            linear_probe=True,
            dist=distributed
        )

    # get optimizer, loss, scheduler
    optimizer = build_optimizer(model, cfg.optimizer)
    loss_func, lovasz_softmax = loss_builder.build(ignore_label=ignore_label)
    num_iters = len(train_dataset_loader) * num_epochs
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_iters,
        lr_min=1e-6,
        warmup_t=min(500, int(0.1 * num_iters)),
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )

    best_miou_train = best_miou_val = 0.
    best_loss_train = best_loss_val = inf
    loss_list = []
    global_iter = 0

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label] if len(unique_label) > 2 else ['empty',
                                                                                                     'occupied']
    CalMeanIou_pts_train = MeanIoU(unique_label, ignore_label, unique_label_str, 'train', sub=0, extra_classes=1)
    CalMeanIou_pts_val = MeanIoU(unique_label, ignore_label, unique_label_str, 'val', sub=0, extra_classes=1)

    for epoch in range(num_epochs):
        CalMeanIou_pts_train.reset()
        CalMeanIou_pts_val.reset()

        # train
        model.train()
        # preds = gt = []
        loss_list = []
        load_s = time.time()
        train_s = time.time()
        iterable = tqdm(train_dataset_loader) if dist.get_rank() == 0 else train_dataset_loader
        for i_iter, (dino_feats, gt_labels) in enumerate(iterable):
            load_t = time.time() - load_s
            # move inputs to device
            dino_feats = dino_feats.to(device)
            gt_labels = gt_labels.to(device)

            # predict
            fwd_s = time.time()
            prediction = model(dino_feats)
            fwd_t = time.time() - fwd_s

            # loss
            loss_ce = loss_func(prediction, gt_labels)
            if log_wandb: wandb.log({'loss/train/ce': loss_ce}, commit=False)
            loss_lovasz = 0
            if use_lovasz:
                loss_lovasz = lovasz_softmax(
                    torch.nn.functional.softmax(prediction, dim=1).squeeze(),
                    gt_labels.squeeze(), ignore=ignore_label
                )
                if log_wandb: wandb.log({'loss/train/lovasz': loss_lovasz}, commit=False)
            loss = loss_ce + loss_lovasz

            # backprop
            bwd_s = time.time()
            optimizer.zero_grad()
            loss.backward()
            if i_iter % print_freq == 0: grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_max_norm)
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step_update(global_iter)
            bwd_t = time.time() - bwd_s

            predicted_idx = prediction.argmax(1).flatten().detach().cpu()
            gt_idx = gt_labels.cpu().squeeze()

            CalMeanIou_pts_train._after_step(predicted_idx, gt_idx)

            lr = optimizer.param_groups[0]['lr']

            if i_iter % print_freq == 0:
                loss_detail = []
                if use_lovasz is not None:
                    loss_detail.append('Lovasz {:.3f}'.format(loss_lovasz))
                loss_detail.append('CE {:.2f}'.format(loss_ce))
                loss_detail = ', '.join(loss_detail)
                logger.info(
                    '[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f) (%s), grad_norm: %.1f, lr: %.7f, '
                    'time: fwd %.3f bwd %.3f data %.3f' % (
                        epoch, i_iter, len(train_dataset_loader), loss.item(), np.mean(loss_list), loss_detail,
                        grad_norm, lr, fwd_t, bwd_t, load_t
                    ))

            if log_wandb:
                wandb.log({
                    'time/train/load': load_t, 'time/train/forward': fwd_t, 'time/train/backward': bwd_t,
                    'loss/train_total': loss, 'lr': lr
                })

            load_s = time.time()
            global_iter += 1

        train_miou, train_ious = CalMeanIou_pts_train._after_epoch(return_per_class=True)
        train_ious_dict = {f'ious/train/{name}': val for name, val in zip(unique_label_str, train_ious)}
        if train_miou > best_miou_train:
            print(f'New best train mIoU: {best_miou_train} -> {train_miou}')
            best_miou_train = train_miou
        else:
            print(f'No improvement in train mIoU: {best_miou_train} > {train_miou}')
        if log_wandb:
            wandb.log({'mIoU/train': train_miou}, step=global_iter)
            wandb.log(train_ious_dict, step=global_iter)

            wandb.log({'loss/train/total_average_epoch': np.mean(loss_list)}, step=global_iter)

            train_ep_t = (time.time() - train_s) / 60
            wandb.log({'time/epoch_train_mins': train_ep_t}, step=global_iter)

        # validation
        model.eval()
        loss_list_val = []
        loss_dict_val = {'ce': []}
        if use_lovasz:
            loss_dict_val['lovasz'] = []
        load_s = time.time()
        val_s = time.time()
        iterable = tqdm(val_dataset_loader) if dist.get_rank() == 0 else val_dataset_loader
        with torch.no_grad():
            for i_iter_val, (dino_feats, gt_labels) in enumerate(iterable):
                load_t = time.time() - load_s
                # move inputs to device
                dino_feats = dino_feats.to(device)
                gt_labels = gt_labels.to(device)

                # predict
                fwd_s = time.time()
                prediction = model(dino_feats)
                fwd_t = time.time() - fwd_s

                # loss
                loss_ce = loss_func(prediction, gt_labels)
                loss_dict_val['ce'].append(loss_ce.item())
                loss_lovasz = 0
                if use_lovasz:
                    loss_lovasz = lovasz_softmax(
                        torch.nn.functional.softmax(prediction, dim=1).squeeze(),
                        gt_labels.squeeze(), ignore=ignore_label
                    )
                    loss_dict_val['lovasz'].append(loss_lovasz.item())
                loss = loss_ce + loss_lovasz
                loss_list_val.append(loss.item())

                predicted_idx = prediction.argmax(1).flatten().detach().cpu()
                gt_idx = gt_labels.cpu().squeeze()

                CalMeanIou_pts_val._after_step(predicted_idx, gt_idx)

                if i_iter_val % print_freq == 0:
                    loss_detail = []
                    if use_lovasz is not None:
                        loss_detail.append('Lovasz {:.3f}'.format(loss_lovasz))
                    loss_detail.append('CE {:.2f}'.format(loss_ce))
                    loss_detail = ', '.join(loss_detail)
                    logger.info(
                        '[VAL] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f) (%s), time: fwd %.3f data %.3f' % (
                            epoch, i_iter_val, len(val_dataset_loader), loss.item(), np.mean(loss_list_val),
                            loss_detail, fwd_t, load_t
                        ))

                load_s = time.time()

            val_miou, val_iou_class = CalMeanIou_pts_val._after_epoch(return_per_class=True)
            val_ious_dict = {f'ious/val/{name}': val for name, val in zip(unique_label_str, val_iou_class)}
            if val_miou > best_miou_val:
                print(f'New best validation mIoU: {best_miou_val} -> {val_miou}')
                best_miou_val = val_miou
                if global_rank==0:
                    dict_to_save = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch + 1,
                        'global_iter': global_iter,
                        'best_val_miou': best_miou_val
                    }
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch + 1}.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = os.path.join(args.work_dir, 'latest.pth')
                    mmcv.symlink(save_file_name, dst_file)
            else:
                print(f'No improvement in validation mIoU: {best_miou_val} > {val_miou}')

            loss_val_mean = np.mean(loss_list_val)
            val_ep_t = (time.time() - val_s) / 60

            if log_wandb:
                wandb.log({'mIoU/val': val_miou}, step=global_iter)
                wandb.log(val_ious_dict, step=global_iter)

                wandb.log({'loss/val/total_average_epoch': loss_val_mean}, step=global_iter)
                # log different loss types
                log_loss_dict_val = {}
                for k, vals in loss_dict_val.items():
                    log_loss_dict_val['loss/val/' + k] = np.mean(vals)
                wandb.log(log_loss_dict_val, step=global_iter)

                wandb.log({'time/epoch_val_mins': val_ep_t}, step=global_iter)


def linear_probe_pred_setup1b(local_rank, args, cfg, use_lovasz, weights_path):
    # global settings
    torch.backends.cudnn.benchmark = True

    print(f'socket.gethostname(): {socket.gethostname()}')

    # init DDP
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

    log_wandb = global_rank == 0 and not args.no_wandb
    if log_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="TPVFormer-Open",
            # set the wandb run name
            name=args.name,
            # track hyperparameters and run metadata
            config=cfg
        )

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    # configure logger
    if dist.get_rank() == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(os.path.join(args.work_dir, os.path.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join('./linear_probing', f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    print(f'Log to {log_file}')
    print_freq = cfg.print_freq

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
    for param in model_occ.parameters():
        param.requires_grad = False
    n_parameters = sum(p.numel() for p in model_occ.parameters() if p.requires_grad)
    logger.info(f'Number of trainable params of the occupancy network: {n_parameters}')
    if distributed and n_parameters > 0:
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

    # resume and load
    assert os.path.isfile(args.ckpt_path)
    cfg.resume_from = args.ckpt_path
    print('ckpt path:', cfg.resume_from)

    map_location = 'cpu'
    ckpt = torch.load(weights_path, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(model.load_state_dict(revise_ckpt_linear_probe(ckpt, ddp=distributed and n_parameters_lin > 0), strict=True))
    print(f'successfully loaded ckpt to linear probe')

    map_location = 'cpu'
    ckpt = torch.load(cfg.resume_from, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(model_occ.load_state_dict(revise_ckpt_linear_probe(ckpt, ddp=distributed and n_parameters > 0), strict=True))
    print(f'successfully loaded ckpt to model_occ')

    model.eval()
    model_occ.eval()

    n_parameters_occ = sum(p.numel() for p in model_occ.parameters() if p.requires_grad)
    logger.info(f'Number of trainable params in the occupancy model: {n_parameters_occ}')
    assert n_parameters_occ == 0
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of trainable params in the linear head: {n_parameters}')

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
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
            scale_rate=cfg.get('scale_rate', 1),
            dist=distributed,
            unique_features=args.unique_features
        )

    # get optimizer, loss, scheduler
    optimizer = build_optimizer(model, cfg.optimizer)
    loss_func, lovasz_softmax = loss_builder.build(ignore_label=ignore_label)
    num_iters = len(train_dataset_loader) * num_epochs
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_iters,
        lr_min=1e-6,
        warmup_t=min(500, int(0.1 * num_iters)),
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )

    best_miou_train = best_miou_val = 0.
    best_loss_train = best_loss_val = inf
    loss_list = []
    global_iter = 0

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label] if len(unique_label) > 2 else ['empty',
                                                                                                     'occupied']
    CalMeanIou_pts_train = MeanIoU(unique_label, ignore_label, unique_label_str, 'train')
    CalMeanIou_pts_val = MeanIoU(unique_label, ignore_label, unique_label_str, 'val')

    for epoch in range(num_epochs):
        CalMeanIou_pts_train.reset()
        CalMeanIou_pts_val.reset()

        if not args.eval_only:
            # train
            # model.train()
            # preds = gt = []
            loss_list = []
            load_s = time.time()
            train_s = time.time()
            iterable = tqdm(train_dataset_loader) if dist.get_rank() == 0 else train_dataset_loader
            for i_iter, loaded_data in enumerate(iterable):
                imgs, img_metas, train_vox_label, train_grid, train_pt_labs, train_vox_label_cls, train_grid_fts_gt, train_fts, _ = loaded_data
                train_grid_fts_gt_int = train_grid_fts_gt.long()
                load_t = time.time() - load_s
                # assert False, "Need to load also GT labels here!"

                # move inputs to device
                imgs = imgs.cuda()
                train_grid = train_grid.to(torch.float32).cuda()

                try:
                    feature_learning = cfg.feature_learning
                except:
                    feature_learning = False
                train_grid_fts = torch.stack(torch.where(train_vox_label > -1)[1:]).T.unsqueeze(
                    0).cuda() if feature_learning else None

                # predict
                fwd_s = time.time()
                # get occupancy and features
                with torch.no_grad():
                    predict_labels_vox, predict_labels_pts, predict_fts_pts = model_occ(img=imgs, img_metas=img_metas,
                                                                                        points=train_grid,
                                                                                        features=train_grid_fts)
                dino_feats_list = []
                dino_targets_list = []
                xyz_pred_occ = []
                xyz_pred_idx = []
                xyz_gt_occ = []
                gt_idx = []
                for bi in range(predict_labels_vox.shape[0]):
                    if args.unique_features:
                        occupied_voxels_gt, unq_fts = train_grid_fts_gt_int[bi].unique(dim=0, return_inverse=True)
                    else:
                        occupied_voxels_gt = train_grid_fts_gt_int[bi]
                    xyz_gt_occ.append(occupied_voxels_gt)
                    # get the predicted occupancy at the positions of GT-occupied voxels
                    occupied_voxels_pred = predict_labels_vox.argmax(1)[
                        bi, occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1], occupied_voxels_gt[:, 2]
                    ]
                    # get locations where it is predicted correctly
                    occupied_voxels_pred_occ_bool = occupied_voxels_pred == 1
                    xyz_pred_idx.extend(occupied_voxels_pred_occ_bool)
                    occupied_voxels_pred_occ = torch.where(occupied_voxels_pred_occ_bool)[0]
                    # get the location of the correctly predicted occupied voxels
                    xyz_pred_occ.append(occupied_voxels_gt[occupied_voxels_pred_occ])
                    # get the GT DINO features at locations of correctly predicted occupied voxels
                    dino_feats = train_fts[bi][occupied_voxels_pred_occ]
                    dino_feats_list.append(dino_feats)
                    gt_idx.append(train_vox_label_cls[bi][occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1],
                                                          occupied_voxels_gt[:, 2]])
                    dino_targets_list.append(train_vox_label_cls[bi][occupied_voxels_gt[:, 0], occupied_voxels_gt[:, 1],
                                                                     occupied_voxels_gt[:, 2]][
                                                 occupied_voxels_pred_occ])
                dino_feats = torch.cat(dino_feats_list).T.unsqueeze(0).cuda().float()
                dino_targets = torch.cat(dino_targets_list).unsqueeze(0).cuda()
                xyz_pred_idx = torch.tensor(xyz_pred_idx)
                del dino_feats_list, dino_targets_list

                # get predictions
                # don't get gradients since we do not train anything
                with torch.no_grad():
                    prediction = model(dino_feats)
                fwd_t = time.time() - fwd_s

                # assert False, "Check if the predict_labels_vox outputs change with the change of 'points' and 'features'."

                # loss
                loss_ce = loss_func(prediction, dino_targets)
                if log_wandb: wandb.log({'loss/train/ce': loss_ce}, commit=False)
                loss_lovasz = 0
                if use_lovasz:
                    loss_lovasz = lovasz_softmax(
                        torch.nn.functional.softmax(prediction, dim=1).squeeze(),
                        dino_targets.squeeze(), ignore=ignore_label
                    )
                    if log_wandb: wandb.log({'loss/train/lovasz': loss_lovasz}, commit=False)
                loss = loss_ce + loss_lovasz

                # no optimization
                if False:
                    # backprop
                    bwd_s = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
                    optimizer.step()
                    loss_list.append(loss.item())
                    scheduler.step_update(global_iter)
                    bwd_t = time.time() - bwd_s
                else:
                    bwd_t = 0.
                    grad_norm = 0.

                predicted_idx_occupied = prediction.argmax(1).flatten().detach().cpu()

                if args.show:
                    # xyz_pred = torch.where(predict_labels_vox[0].argmax(0))[0].T
                    # xyz_gt = torch.where(train_vox_label[0].flatten())[0].T
                    fig = plt.figure()
                    show3d(xyz_pred_occ[0], fig, 1, 1, 1, s=1., labels=predicted_idx_occupied)
                    plt.show()
                    fig = plt.figure()
                    show3d(xyz_pred_occ[0], fig, 1, 2, 1, s=1., labels=predicted_idx_occupied)
                    show3d(xyz_gt_occ[0], fig, 1, 2, 2, s=1., labels=gt_idx[0])
                    plt.show()

                full_targets = train_vox_label_cls[:, train_grid_fts_gt_int[..., 0], train_grid_fts_gt_int[..., 1],
                               train_grid_fts_gt_int[..., 2]].flatten()
                predicted_idx = torch.ones_like(full_targets) * 17  # initialize with "empty" predictions
                predicted_idx[xyz_pred_idx.cpu()] = predicted_idx_occupied

                CalMeanIou_pts_train._after_step(predicted_idx, full_targets)

                lr = optimizer.param_groups[0]['lr']

                if i_iter % print_freq == 0:
                    loss_detail = []
                    if use_lovasz is not None:
                        loss_detail.append('Lovasz {:.3f}'.format(loss_lovasz))
                    loss_detail.append('CE {:.2f}'.format(loss_ce))
                    loss_detail = ', '.join(loss_detail)
                    logger.info(
                        '[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f) (%s), grad_norm: %.1f, lr: %.7f, '
                        'time: fwd %.3f bwd %.3f data %.3f' % (
                            epoch, i_iter, len(train_dataset_loader), loss.item(), np.mean(loss_list), loss_detail,
                            grad_norm, lr, fwd_t, bwd_t, load_t
                        ))

                if log_wandb:
                    wandb.log({
                        'time/train/load': load_t, 'time/train/forward': fwd_t, 'time/train/backward': bwd_t,
                        'loss/train_total': loss, 'lr': lr
                    })

                load_s = time.time()
                global_iter += 1

            train_miou, train_ious = CalMeanIou_pts_train._after_epoch(return_per_class=True)
            train_ious_dict = {f'ious/train/{name}': val for name, val in zip(unique_label_str, train_ious)}
            if train_miou > best_miou_train:
                print(f'New best train mIoU: {best_miou_train} -> {train_miou}')
                best_miou_train = train_miou
            else:
                print(f'No improvement in train mIoU: {best_miou_train} > {train_miou}')
            if log_wandb:
                wandb.log({'mIoU/train': train_miou}, step=global_iter)
                wandb.log(train_ious_dict, step=global_iter)

                wandb.log({'loss/train/total_average_epoch': np.mean(loss_list)}, step=global_iter)

                train_ep_t = (time.time() - train_s) / 60
                wandb.log({'time/epoch_train_mins': train_ep_t}, step=global_iter)

        # validation
        model.eval()
        loss_list_val = []
        loss_dict_val = {'ce': []}
        if use_lovasz:
            loss_dict_val['lovasz'] = []
        load_s = time.time()
        val_s = time.time()
        iterable = tqdm(val_dataset_loader) if dist.get_rank() == 0 else val_dataset_loader
        with torch.no_grad():
            for i_iter, loaded_data in enumerate(iterable):
                imgs, img_metas, val_vox_label, val_grid, val_pt_labs, val_vox_label_cls, val_grid_fts_gt, val_fts, _ = loaded_data
                val_grid_fts_gt_int = val_grid_fts_gt.long()
                load_t = time.time() - load_s
                # move inputs to device
                imgs = imgs.cuda()
                val_grid = val_grid.to(torch.float32).cuda()

                try:
                    feature_learning = cfg.feature_learning
                except:
                    feature_learning = False
                val_grid_fts = torch.stack(torch.where(val_vox_label > -1)[1:]).T.unsqueeze(
                    0).cuda() if feature_learning else None

                # predict
                fwd_s = time.time()
                # get occupancy and features
                with torch.no_grad():
                    predict_labels_vox, predict_labels_pts, predict_fts_pts = model_occ(img=imgs, img_metas=img_metas,
                                                                                        points=val_grid,
                                                                                        features=val_grid_fts)
                dino_feats_list = []
                dino_targets_list = []
                xyz_pred_occ = []
                xyz_pred_idx = []
                xyz_gt_occ = []
                gt_idx = []
                for bi in range(predict_labels_vox.shape[0]):
                    if args.unique_features:
                        occupied_voxels_gt, unq_fts = val_grid_fts_gt_int[bi].unique(dim=0, return_inverse=True)
                    else:
                        occupied_voxels_gt = val_grid_fts_gt_int[bi]
                    xyz_gt_occ.append(occupied_voxels_gt)
                    # get the predicted occupancy at the positions of GT-occupied voxels
                    occupied_voxels_pred = predict_labels_vox.argmax(1)[
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
                del dino_targets_list, dino_feats_list

                # get predictions
                # don't get gradients since we do not train anything
                with torch.no_grad():
                    prediction = model(dino_feats)
                fwd_t = time.time() - fwd_s

                if args.show:
                    locs = xyz_pred_occ[0].cpu()
                    labels = prediction[0].argmax(0).cpu()

                    show3d(locs, plt.figure(), 1, 1, 1, labels=labels, s=1.)
                    plt.show()

                # assert False, "Check if the predict_labels_vox outputs change with the change of 'points' and 'features'."

                # loss
                loss_ce = loss_func(prediction, dino_targets)
                if log_wandb: wandb.log({'loss/val/ce': loss_ce}, commit=False)
                loss_lovasz = 0
                if use_lovasz:
                    loss_lovasz = lovasz_softmax(
                        torch.nn.functional.softmax(prediction, dim=1).squeeze(),
                        dino_targets.squeeze(), ignore=ignore_label
                    )
                    if log_wandb: wandb.log({'loss/val/lovasz': loss_lovasz}, commit=False)
                loss = loss_ce + loss_lovasz

                # no optimization
                bwd_t = 0.
                grad_norm = 0.

                predicted_idx_occupied = prediction.argmax(1).flatten().detach().cpu()

                if args.show:
                    # xyz_pred = torch.where(predict_labels_vox[0].argmax(0))[0].T
                    # xyz_gt = torch.where(train_vox_label[0].flatten())[0].T
                    fig = plt.figure()
                    show3d(xyz_pred_occ[0], fig, 1, 1, 1, s=1., labels=predicted_idx_occupied)
                    plt.show()
                    fig = plt.figure()
                    show3d(xyz_pred_occ[0], fig, 1, 2, 1, s=1., labels=predicted_idx_occupied)
                    show3d(xyz_gt_occ[0], fig, 1, 2, 2, s=1., labels=gt_idx[0])
                    plt.show()

                full_targets = val_vox_label_cls[:, val_grid_fts_gt_int[..., 0], val_grid_fts_gt_int[..., 1],
                               val_grid_fts_gt_int[..., 2]].flatten()
                predicted_idx = torch.ones_like(full_targets) * 17  # initialize with "empty" predictions
                predicted_idx[xyz_pred_idx.cpu()] = predicted_idx_occupied

                CalMeanIou_pts_val._after_step(predicted_idx, full_targets)

                lr = optimizer.param_groups[0]['lr']

                if i_iter % print_freq == 0:
                    loss_detail = []
                    if use_lovasz is not None:
                        loss_detail.append('Lovasz {:.3f}'.format(loss_lovasz))
                    loss_detail.append('CE {:.2f}'.format(loss_ce))
                    loss_detail = ', '.join(loss_detail)
                    logger.info(
                        '[VAL] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f) (%s), grad_norm: %.1f, lr: %.7f, '
                        'time: fwd %.3f bwd %.3f data %.3f' % (
                            epoch, i_iter, len(val_dataset_loader), loss.item(), np.mean(loss_list), loss_detail,
                            grad_norm, lr, fwd_t, bwd_t, load_t
                        ))

                if log_wandb:
                    wandb.log({
                        'time/val/load': load_t, 'time/val/forward': fwd_t, 'time/val/backward': bwd_t,
                        'loss/val_total': loss, 'lr': lr
                    })

                load_s = time.time()
                global_iter += 1

            val_miou, val_iou_class = CalMeanIou_pts_val._after_epoch(return_per_class=True)
            val_ious_dict = {f'ious/val/{name}': val for name, val in zip(unique_label_str, val_iou_class)}
            if val_miou > best_miou_val:
                print(f'New best validation mIoU: {best_miou_val} -> {val_miou}')
                best_miou_val = val_miou
                dict_to_save = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'global_iter': global_iter,
                    'best_val_miou': best_miou_val
                }
                save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch + 1}.pth')
                torch.save(dict_to_save, save_file_name)
                dst_file = os.path.join(args.work_dir, 'latest.pth')
                mmcv.symlink(save_file_name, dst_file)
            else:
                print(f'No improvement in validation mIoU: {best_miou_val} > {val_miou}')

            loss_val_mean = np.mean(loss_list_val)
            val_ep_t = (time.time() - val_s) / 60

            if log_wandb:
                wandb.log({'mIoU/val': val_miou}, step=global_iter)
                wandb.log(val_ious_dict, step=global_iter)

                wandb.log({'loss/val/total_average_epoch': loss_val_mean}, step=global_iter)
                # log different loss types
                log_loss_dict_val = {}
                for k, vals in loss_dict_val.items():
                    log_loss_dict_val['loss/val/' + k] = np.mean(vals)
                wandb.log(log_loss_dict_val, step=global_iter)

                wandb.log({'time/epoch_val_mins': val_ep_t}, step=global_iter)


def build_linear_probe(model_cfg):
    def build_block(ind, outd):
        block = nn.Sequential(
            nn.Conv1d(ind, outd, 1, 1),
            nn.BatchNorm1d(outd),
            nn.ReLU(),
        )
        return block

    input_dim = model_cfg['input_dim']
    hidden_dim = model_cfg['hidden_dim']
    num_hidden = model_cfg['num_hidden']
    num_classes = model_cfg['nbr_class']

    if hidden_dim == 0:
        return nn.Sequential(
            nn.Conv1d(input_dim, num_classes, 1, 1)
        )

    layers = [build_block(input_dim, hidden_dim)]
    for _ in range(num_hidden - 1):
        layers.append(build_block(hidden_dim, hidden_dim))
    layers.append(nn.Conv1d(hidden_dim, num_classes, 1, 1))

    model = nn.Sequential(*layers)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/linear_mini_7ep_gt.py')
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--use-lovasz', action='store_true')
    parser.add_argument('--work-dir', type=str, default='./out/linear_probe')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--run-mode', type=str, default='linear_gt')
    parser.add_argument('--batch-sizes', type=int, nargs='+')
    parser.add_argument('--num-workers', default=None, type=int)
    parser.add_argument('--num-epochs', default=None, type=int)
    parser.add_argument('--train-bs', default=None, type=int)
    parser.add_argument('--weights-path-linear', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--unique-features', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    cfg = Config.fromfile(args.py_config)

    if args.train_bs is not None:
        cfg.train_data_loader["batch_size"] = args.train_bs

    if args.lr is not None:
        cfg.optimizer['lr'] = args.lr

    if args.num_epochs is not None:
        cfg.max_epochs = args.num_epochs

    if args.num_workers is not None:
        cfg.train_data_loader["num_workers"] = args.num_workers
        print(f'Set cfg.train_data_loader["num_workers"] to {cfg.train_data_loader["num_workers"]} '
              f'(check {args.num_workers})')

    if args.name is None:
        args.name = args.work_dir.split(os.path.sep)[-1]

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    lr = cfg.optimizer['lr']
    args.name += f'_lr{lr}_{cfg.max_epochs}ep_{timestamp}'
    args.work_dir += f'_lr{lr}_{cfg.max_epochs}ep_{timestamp}'

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.run_mode == 'linear_gt':
        # linear_probe_gt(cfg, args.use_lovasz)
        torch.multiprocessing.spawn(linear_probe_gt, args=(args, cfg, args.use_lovasz,), nprocs=args.gpus)
    elif args.run_mode == 'linear_pred':
        torch.multiprocessing.spawn(linear_probe_pred_setup1b,
                                    args=(args, cfg, args.use_lovasz, args.weights_path_linear,), nprocs=args.gpus)
    elif args.run_mode == 'bechmark_bs':
        assert len(
            args.batch_sizes) > 0, "Batch sizes to explore need to be specified in the args.batch_sizes argument!"
        # linear_probe_gt_benchmark_bs(cfg, args.use_lovasz, args.batch_sizes)
        torch.multiprocessing.spawn(linear_probe_gt_benchmark_bs, args=(args, cfg, args.use_lovasz, args.batch_sizes,),
                                    nprocs=args.gpus)
    else:
        raise NotImplementedError(f'Behavior for args.run_mode={args.run_mode} is not implemented!')
