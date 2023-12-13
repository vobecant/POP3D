# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/jx_vit_base_p16_224-80ecf9dd.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        patch_bias=True,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        pre_norm = False,
        final_norm=True,
        return_qkv=False,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=-1,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0,
        num_classes=20,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)  # yapf: disable
