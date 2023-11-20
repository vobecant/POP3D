_base_ = [
    './_base_/dataset_25.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

max_epochs = 7
feature_learning = True
print_freq = 20

dataset_params = dict(
    version="v1.0-trainval",
    ignore_label=0,
    fill_label=17,
    fixed_volume_space=True,
    label_mapping="./config/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space=[51.2, 51.2, 3],
    min_volume_space=[-51.2, -51.2, -5],
    linear_gt=True,
    dataset_type="ImagePoint_NuScenes_withFeatures",
    features_type="k",
    features_path="/mnt/proj1/open-26-3/datasets/nuscenes/features/vit_small_8/matched",
    projections_path='/mnt/proj1/open-26-3/datasets/nuscenes/features/projections'
)

train_data_loader = dict(
    batch_size=8,
    num_workers=4
)

model_params = dict(
    input_dim=384,
    hidden_dim=384,
    num_hidden=2,
    nbr_class=18
)

tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]
