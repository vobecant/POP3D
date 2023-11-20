train_bs = 1
num_workers = 0
dataset_params = dict(
    version='v1.0-mini',
    ignore_label=255,
    fill_label=0,
    fixed_volume_space=True,
    label_mapping='./config/label_mapping/nuscenes-agnostic.yaml',
    max_volume_space=[51.2, 51.2, 3],
    min_volume_space=[-51.2, -51.2, -5],
    dataset_type='ImagePoint_NuScenes_withFeatures',
    features_type='clip',
    features_path='/nfs/datasets/nuscenes/maskclip_features_projections_all',
    projections_path='/nfs/datasets/nuscenes/features/projections',
    class_agnostic=True)
train_data_loader = dict(
    data_path='data/nuscenes/',
    imageset='./data/nuscenes_infos_train_mini.pkl',
    batch_size=1,
    shuffle=True,
    num_workers=0)
val_data_loader = dict(
    data_path='data/nuscenes/',
    imageset='./data/nuscenes_infos_val_mini.pkl',
    batch_size=1,
    shuffle=False,
    num_workers=0)
unique_label = [0, 1]
class_weights_path = '/home/vobecant/PhD/TPVFormer-OpenSet/class_weights.pth'
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
grad_max_norm = 35
print_freq = 50
max_epochs = 24
load_from = './ckpts/moco_v2_800ep_weights.pth.tar'
scale_rate = 0.5
occupancy = True
lovasz_input = 'none'
ce_input = 'voxel'
voxel_feature_loss = 'l2'
feature_learning = True
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
_num_cams_ = 6
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [100, 100, 8]
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
nbr_class = 2
feature_dim = 512
hidden_dims_ft = 1024
out_dims_ft = 1024
model = dict(
    type='TPVFormer',
    use_grid_mask=True,
    tpv_aggregator=dict(
        type='TPVAggregator',
        tpv_h=100,
        tpv_w=100,
        tpv_z=8,
        nbr_classes=2,
        in_dims=256,
        hidden_dims=512,
        out_dims=256,
        scale_h=1,
        scale_w=1,
        scale_z=1,
        feature_dim=512,
        hidden_dims_ft=1024,
        out_dims_ft=1024),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    tpv_head=dict(
        type='TPVFormerHead',
        tpv_h=100,
        tpv_w=100,
        tpv_z=8,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        num_feature_levels=4,
        num_cams=6,
        embed_dims=256,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=100,
            col_num_embed=100),
        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=100,
            tpv_w=100,
            tpv_z=8,
            num_layers=3,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            num_points_in_pillar=[4, 32, 32],
            return_intermediate=False,
            transformerlayers=dict(
                type='TPVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TPVCrossViewHybridAttention',
                        embed_dims=256,
                        num_levels=1),
                    dict(
                        type='TPVImageCrossAttention',
                        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        deformable_attention=dict(
                            type='TPVMSDeformableAttention3D',
                            embed_dims=256,
                            num_points=[8, 64, 64],
                            num_z_anchors=[4, 32, 32],
                            num_levels=4,
                            floor_sampling_offset=False,
                            tpv_h=100,
                            tpv_w=100,
                            tpv_z=8),
                        embed_dims=256,
                        tpv_h=100,
                        tpv_w=100,
                        tpv_z=8)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')))))
work_dir = 'out/tmp_04052023_234419'
