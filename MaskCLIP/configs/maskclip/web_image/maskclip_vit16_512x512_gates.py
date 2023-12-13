_base_ = [
    '../../_base_/models/maskclip_vit16.py', '../../_base_/datasets/web_image.py', 
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_20k.py'
]
img_dir = 'gates'
num_class = 7
model = dict(
    decode_head=dict(
        num_classes=num_class,
        text_categories=num_class, 
        text_embeddings_path=f'pretrain/{img_dir}_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth',
    )
)
data = dict(
    samples_per_gpu=4,
    test=dict(img_dir=img_dir, split=f'{img_dir}.txt', data_name=img_dir)
)