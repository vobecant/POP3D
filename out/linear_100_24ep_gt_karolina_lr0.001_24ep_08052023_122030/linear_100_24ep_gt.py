dataset_params = dict(
    version='v1.0-trainval',
    ignore_label=0,
    fill_label=17,
    fixed_volume_space=True,
    label_mapping='./config/label_mapping/nuscenes-noIgnore.yaml',
    max_volume_space=[51.2, 51.2, 3],
    min_volume_space=[-51.2, -51.2, -5],
    linear_gt=True)
train_data_loader = dict(
    data_path='data/nuscenes/',
    imageset='./data/nuscenes_infos_100_train.pkl',
    batch_size=1,
    shuffle=True,
    num_workers=0)
val_data_loader = dict(
    data_path='data/nuscenes/',
    imageset='./data/nuscenes_infos_100_val.pkl',
    batch_size=1,
    shuffle=False,
    num_workers=1)
unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
optimizer = dict(
    type='AdamW',
    lr=0.001,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
grad_max_norm = 35
print_freq = 50
print_freq_wandb_train = 300
print_freq_wandb_val = 150
max_epochs = 24
load_from = './ckpts/r101_dcn_fcos3d_pretrain.pth'
model_params = dict(input_dim=384, hidden_dim=384, num_hidden=2, nbr_class=17)
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [100, 100, 8]
gpu_ids = range(0, 1)
