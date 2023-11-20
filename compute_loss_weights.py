import argparse
import os
import pickle

import numpy as np
import torch
try:
    from mmcv import Config
except:
    from mmengine.config import Config
from tqdm import tqdm


def main(args):
    cfg = Config.fromfile(args.py_config)

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader
    grid_size = cfg.grid_size

    # setup configuration and loss for feature learning
    # if the feature learning is ON
    try:
        feature_learning = cfg.feature_learning
    except:
        feature_learning = False

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=False,
            scale_rate=cfg.get('scale_rate', 1),
            num_workers=args.num_workers
        )

    num_free = num_occ = 0

    for loaded_data in tqdm(train_dataset_loader):
        if not feature_learning:
            imgs, img_metas, train_vox_label, train_grid, train_pt_labs, train_pt_labs_cls = loaded_data
            train_grid_fts = None
        else:
            imgs, img_metas, train_vox_label, train_grid, train_pt_labs, train_pt_labs_cls, train_grid_fts, train_pt_fts, _ = loaded_data

        unq, cnt = torch.unique(train_vox_label, return_counts=True)
        for u, c in zip(unq, cnt):
            if u == 0:
                num_free += c.item()
            elif u == 1:
                num_occ += c.item()

    print(f'num_free={num_free}, num_occ={num_occ}')

    name = os.path.split(args.py_config)[-1].split('.')[0] + '_counts'

    total_count = num_free + num_occ
    free_prob = float(num_free) / float(total_count)
    free_weight = 1 / np.log(1.02 + free_prob)
    occ_prob = float(num_occ) / float(total_count)
    occ_weight = 1 / np.log(1.02 + occ_prob)

    d = {
        'num_free': num_free, 'num_occ': num_occ,
        'weight_free': free_weight, 'weight_occ': occ_weight
    }
    print(d)

    savepath = os.path.join(args.save_dir, name + '.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(d, f)

    class_weights = torch.tensor([free_weight, occ_weight])
    savepath = os.path.join(args.save_dir, name + '__weights.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(class_weights, f)
    print(f'Saved to {savepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_small_occupancy_mini_agnostic_also.py')
    parser.add_argument('--save-dir', default='./data')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    main(args)
