import argparse

from mmcv import Config
from tqdm import tqdm

MISSING = [
    '/mnt/proj1/open-26-3/datasets/nuscenes/features/vit_small_8/matched/CAM_BACK/n015-2018-09-25-11-10-38+0800__CAM_BACK__1537845547787525__k.pth',
]


def main(args):
    # load config
    cfg = Config.fromfile(args.py_config)

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader
    grid_size = cfg.grid_size
    distributed = False  # True
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
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1),
            num_workers=args.num_workers
        )

    unextracted_files, unextracted_files_train, unextracted_files_val = [], [], []

    assert feature_learning

    for i_iter, loaded_data in enumerate(tqdm(train_dataset_loader)):
        *_, nonexistent = loaded_data
        if len(nonexistent) > 0:
            for n in nonexistent:
                unextracted_files.append(n)
                unextracted_files_train.append(n)

    for i_iter, loaded_data in enumerate(tqdm(val_dataset_loader)):
        *_, nonexistent = loaded_data
        if len(nonexistent) > 0:
            for n in nonexistent:
                unextracted_files.append(n)
                unextracted_files_val.append(n)

    print(unextracted_files)
    print()
    print(unextracted_files_train)
    print()
    print(unextracted_files_val)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy_50_7ep_agnostic_noLovasz_features.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--eval-at-start', action='store_true')
    parser.add_argument('--compute-weights', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--agnostic', action='store_true')
    parser.add_argument('--class-weights-path',
                        # default='./class_weights.pkl',
                        default=None,
                        type=str)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--num-workers', default=10, type=int)

    args = parser.parse_args()

    main(args)
