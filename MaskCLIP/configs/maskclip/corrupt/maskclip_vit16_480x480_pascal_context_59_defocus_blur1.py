_base_ = '../maskclip_vit16_480x480_pascal_context_59.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(520, 520),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # possible choices for Corrupt `name`: gaussian_noise, shot_noise, impulse_noise
            #   speckle_noise, gaussian_blur, defocus_blur, spatter, jpeg_compression
            # possible choices for Corrupt `level`: 1, 2, 3, 4, 5
            dict(type='Corrupt', name='defocus_blur', level=1),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    test=dict(
        pipeline=test_pipeline
    )
)