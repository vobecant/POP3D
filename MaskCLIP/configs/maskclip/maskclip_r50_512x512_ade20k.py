_base_ = [
    '../_base_/models/maskclip_r50.py', '../_base_/datasets/ade20k.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(
        num_classes=150,
        text_categories=150, 
        text_channels=1024, 
        text_embeddings_path='pretrain/ade_RN50_clip_text.pth',
        visual_projs_path='pretrain/RN50_clip_weights.pth',
    )
)