_base_ = [
    './_base_/dataset_25.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

max_epochs = 21

dataset_params = dict(
    version="v1.0-trainval",
    ignore_label=0,
    fill_label=17,
    fixed_volume_space=True,
    label_mapping="./config/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space=[51.2, 51.2, 3],
    min_volume_space=[-51.2, -51.2, -5],
    linear_gt=True,
    features_path="/mnt/proj1/open-26-3/datasets/nuscenes/features/vit_small_8/matched",
    projections_path='/mnt/proj1/open-26-3/datasets/nuscenes/features/projections'
)

model_params = dict(
    input_dim=384,
    hidden_dim=384,
    num_hidden=2,
    nbr_class=18
)
