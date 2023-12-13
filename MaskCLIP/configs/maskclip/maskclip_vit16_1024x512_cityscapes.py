_base_ = [
    '../_base_/models/maskclip_vit16.py', '../_base_/datasets/cityscapes.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(
        num_classes=19,
        text_categories=19, 
        text_channels=512, 
        text_embeddings_path='pretrain/city_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth',
    ),
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline)
)