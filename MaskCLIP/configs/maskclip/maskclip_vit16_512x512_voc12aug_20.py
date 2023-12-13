_base_ = [
    '../_base_/models/maskclip_vit16.py', '../_base_/datasets/pascal_voc12_aug_20.py', 
    # '../_base_/models/maskclip_vit16.py', '../_base_/datasets/pascal_voc12_aug.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(
        num_classes=20,
        text_categories=20, 
        text_channels=512, 
        text_embeddings_path='pretrain/voc_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth',
    ),
)