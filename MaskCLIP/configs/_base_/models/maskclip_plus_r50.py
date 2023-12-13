# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='MaskClipPlusHead',
        in_channels=2048,
        channels=1024,
        num_classes=0,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        decode_module_cfg=dict(
            type='ASPPHeadV2',
            input_transform=None,
            dilations=(6, 12, 18, 24)
        ),
        text_categories=20,
        text_channels=1024,
        text_embeddings_path='pretrain/voc_RN50_clip_text.pth',
        cls_bg=False,
        norm_feat=False,
        clip_unlabeled_cats=list(range(0, 20)),
        clip_cfg=dict(
            type='ResNetClip',
            depth=50,
            norm_cfg=norm_cfg,
            contract_dilation=True
        ),
        clip_weights_path='pretrain/RN50_clip_weights.pth',
        reset_counter=True,
        start_clip_guided=(1, -1),
        start_self_train=(-1, -1)
    ),
    feed_img_to_decode_head=True,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
