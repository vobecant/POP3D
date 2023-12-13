_base_ = './maskclip_r50_480x480_pascal_context_59.py'

model = dict(
    backbone=dict(
        stem_channels=96,
        base_channels=96,
        depth='50x16'
    ),
    decode_head=dict(
        in_channels=3072,
        text_channels=768, 
        text_embeddings_path='pretrain/context_RN50x16_clip_text.pth',
        visual_projs_path='pretrain/RN50x16_clip_weights.pth',
    )
)