# dataset settings
dataset_type = "CustomDatasetMy"  # "NuscenesDataset"
data_root = "/nfs/datasets/nuscenes/mmseg"
img_infos_path_train = '/nfs/datasets/nuscenes/mmseg/img_infos_train20.pkl'  # '/nfs/datasets/nuscenes/mmseg/img_infos_train.pkl'
img_infos_path_val = '/nfs/datasets/nuscenes/mmseg/img_infos_val100.pkl'  # '/nfs/datasets/nuscenes/mmseg/img_infos_val.pkl'

seg_map_suffix = ".png"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (768, 768)
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/train",
        ann_dir="labels_png/train",
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        img_infos_path=img_infos_path_train,
        label_mapping_path='segm/data/mapping_class_index_nuscenes.npy',
    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["images/train", "images/val"],
        ann_dir=["labels_png/train", "labels_png/val"],
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        label_mapping_path='segm/data/mapping_class_index_nuscenes.npy',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="labels_png/val",
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        img_infos_path=img_infos_path_val,
        label_mapping_path='segm/data/mapping_class_index_nuscenes.npy',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/test",
        ann_dir="labels_png/test",
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        label_mapping_path='segm/data/mapping_class_index_nuscenes.npy',
    ),
)
