# dataset settings
dataset_type = "ADE20KDataset"
data_root = "../data/nuscenes"

# class_names = ['adult pedestrian', 'child pedestrian', 'wheelchair', 'stroller', 'personal mobility',
#                'police officer', 'construction worker', 'animal', 'car', 'motorcycle', 'bicycle', 'bendy bus',
#                'rigid bus', 'truck', 'construction vehicle', 'ambulance vehicle', 'police car', 'trailer',
#                'barrier', 'traffic cone', 'debris', 'bicycle rack', 'driveable surface', 'sidewalk', 'terrain',
#                'other flat', 'manmade', 'vegetation']
class_names = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
               'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow',
               'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
               'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
               'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
               'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
palette = PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                     [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                     [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                     [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                     [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                     [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                     [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                     [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                     [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                     [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                     [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                     [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                     [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                     [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                     [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                     [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                     [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                     [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                     [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                     [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                     [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                     [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                     [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                     [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                     [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                     [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                     [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                     [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                     [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                     [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                     [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                     [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                     [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                     [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                     [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                     [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                     [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                     [102, 255, 0], [92, 0, 255]][:len(class_names)]
#allow_custom_classes = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (768, 768)
image_size = (900, 1600)
max_ratio = 2
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1600, 900), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size),  # , cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(900 * max_ratio, 900),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024 * max_ratio, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir="images/train",
        ann_dir=None,#"labels_png/train",
        pipeline=train_pipeline,
        #classes=class_names,
        palette=palette,
        #allow_custom_classes=True
    ),
    val=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir="images/val",
        ann_dir=None,#"labels_png/val",
        pipeline=test_pipeline,
        #classes=class_names,
        palette=palette,
        #allow_custom_classes=True
    ),
    test=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir="images/trainval",
        # img_dir='../samples/debug/scene-0003',
        ann_dir=None,#"labels_png/val",
        pipeline=test_pipeline,
        #classes=class_names,
        palette=palette,
        #allow_custom_classes=True
    ),
    test_train=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir="images/trainval",
        # img_dir='../samples/debug/scene-0003',
        ann_dir=None,#"labels_png/val",
        pipeline=test_pipeline,
        #classes=class_names,
        palette=palette,
        #allow_custom_classes=True
    ),
    complete=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir="samples",
        # img_dir='../samples/debug/scene-0003',
        ann_dir=None,#"labels_png/val",
        pipeline=test_pipeline,
        #classes=class_names,
        palette=palette,
        #allow_custom_classes=True
    )
)
