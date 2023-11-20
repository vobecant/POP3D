_base_ = [
    './_base_/dataset_100.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

max_epochs = 7

dataset_params = dict(
    version="v1.0-trainval",
    ignore_label=0,
    fill_label=17,
    fixed_volume_space=True,
    label_mapping="./config/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space=[51.2, 51.2, 3],
    min_volume_space=[-51.2, -51.2, -5],
    linear_gt=True
)

model_params = dict(
    input_dim=384,
    hidden_dim=384,
    num_hidden=2,
    nbr_class=18
)
