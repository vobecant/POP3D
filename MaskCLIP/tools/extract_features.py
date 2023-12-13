# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)   
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test #, vis_output
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path', nargs='?',  default=None,
                        # default="configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_test_ciirc.py", 
                        # default="configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainval_ciirc.py"
                        )
    parser.add_argument('--save-dir', type=str, default=None, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
             'not be supported in version v0.22.0. Override some settings in the '
             'used config, the key-value pair in xxx=yyy format will be merged '
             'into config file. If the value to be overwritten is a list, it '
             'should be like key="[a,b]" or key=a,b It also allows nested '
             'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
             'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    # parser.add_argument('--paths-file', type=str,
    #                     default='/home/vobecant/PhD/MaskCLIP/paths_file.txt'
    #                     )
    parser.add_argument('--test-on-train', action='store_true')

    parser.add_argument('--projections-dir', 
                        default=None
                        )
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--complete', help='Load all images regardless the split.', action='store_true')
    parser.add_argument('--paths-file', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-dir', type=str, default=None)

    args = parser.parse_args()
    args.complete = True

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def limit_dataset(dataset, paths_file, start, end):
    print(f'Limiting dataset to images with paths in {paths_file}')

    paths = None
    if paths_file is not None:
        with open(paths_file, 'r') as f:
            paths = [os.path.split(p.strip())[-1] for p in f.readlines()]
        paths = set(paths)
    print(paths)

    nimgs = len(dataset.img_infos)
    print(f'dataset has {nimgs} images')

    if paths is not None:
        new_img_infos = []
        for i,img_info in enumerate(dataset.img_infos):
            path = os.path.split(img_info['filename'])[-1]
            if i<10: print(i,path)
            if path in paths:
                new_img_infos.append(img_info)
            else:
                pass
                # print(f'{path} is not among paths')
        print(i)
        dataset.img_infos = new_img_infos
        new_len = len(dataset.img_infos)
        # assert new_len == len(paths), f"new_len: {new_len}, len(paths): {len(paths)}"
        print(f'The limited dataset has {new_len} samples based on paths.')

    if start is not None and end is not None:
        dataset.img_infos = dataset.img_infos[start:end]
        new_len = len(dataset.img_infos)
        print(f'The limited dataset has {new_len} samples by taking indices {start}-{end}.')

    return dataset


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.complete:
        cfg.data.test = cfg.data.complete
    elif args.test_on_train:
        cfg.data.test = cfg.data.test_train
        
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(work_dir, 'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    tmp_dir = None
    # if args.paths_file is not None:
    #     # cfg.data.test, tmpdir = ...
    #     cfg.data.test.data_root = '/home/vobecant'
    #     cfg.data.test.img_dir = 'to_extract'

    print(f'Build dataset from config: {cfg.data.test}')
    dataset = build_dataset(cfg.data.test)
    if args.paths_file is not None or (args.start is not None and args.end is not None):
        dataset = limit_dataset(dataset, args.paths_file, args.start, args.end)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu if args.num_workers is None else args.num_workers,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = False
    tmpdir = None

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])

        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            #pre_eval=args.eval is not None and not eval_on_format_results,
            #format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs,
            start=args.start, end=args.end,
            save_dir=args.save_dir,
            projections_dir=args.projections_dir
        )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            #pre_eval=args.eval is not None and not eval_on_format_results,
            #format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)


if __name__ == '__main__':
    main()
