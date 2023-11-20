import torch
from nuscenes import NuScenes

from dataloader.dataset import ImagePoint_NuScenes, ImagePoint_NuScenes_withFeatures, \
    ImagePoint_NuScenes_withFeatures_openseg
from dataloader.dataset_wrapper import custom_collate_fn, DatasetWrapper_NuScenes, custom_collate_fn_linear_gt, \
    DatasetWrapper_NuScenes_LinearGT, custom_collate_fn_dino

DATASET_LUT = {"ImagePoint_NuScenes_withFeatures": ImagePoint_NuScenes_withFeatures,
               "ImagePoint_NuScenes_withFeatures_openseg": ImagePoint_NuScenes_withFeatures_openseg,
               "ImagePoint_NuScenes": ImagePoint_NuScenes}


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[200, 200, 16],
          version='v1.0-trainval',
          dist=False,
          scale_rate=1,
          num_workers=None,
          linear_probe=False,
          with_projections=False,
          unique_features=False,
          train_noaug=False,
          eval_mode=False,
          semantic_points=False,
          class_agnostic=None,
          fill_label_gt=None,
          dataset_type=None,
          no_nusc=False,
          retrieval=False,
          merged=False,
          return_test=False
          ):
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    label_mapping = dataset_config["label_mapping"]
    dataset_type = dataset_type if dataset_type is not None else dataset_config.get("dataset_type",
                                                                                    "ImagePoint_NuScenes")
    no_nusc = no_nusc or dataset_config.get('no_nusc', False)
    features_type = dataset_config.get("features_type", None)
    features_path = dataset_config.get("features_path", None)
    projections_path = dataset_config.get("projections_path", None)
    class_agnostic = dataset_config.get("class_agnostic", False) if class_agnostic is None else class_agnostic
    gt_mode = dataset_config.get("gt_mode", 'tpvformer')
    ignore_label = dataset_config["ignore_label"]
    linear_gt = dataset_config.get("linear_gt", False)
    label_mapping_gt = dataset_config.get("label_mapping_gt", "./config/label_mapping/nuscenes-noIgnore.yaml")

    dino_features = dataset_config.get('dino_features', False)
    features_type_dino = dataset_config.get('features_type_dino', None)
    features_path_dino = dataset_config.get('features_path_dino', None)
    projections_path_dino = dataset_config.get('projections_path_dino', projections_path)
    cam2tok_path = dataset_config.get('cam2tok', None)

    max_features = dataset_config.get('max_points', None)
    labels_path = dataset_config.get('labels_path', None)

    dataset_fn = DATASET_LUT[dataset_type]

    nusc = None
    if not no_nusc:
        nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    train_dataset = dataset_fn(data_path, imageset=train_imageset, label_mapping=label_mapping, nusc=nusc,
                               features_type=features_type, features_path=features_path,
                               projections_path=projections_path, class_agnostic=class_agnostic,
                               linear_probe=linear_probe, label_mapping_gt=label_mapping_gt,
                               dino_features=dino_features, features_type_dino=features_type_dino,
                               features_path_dino=features_path_dino, projections_path_dino=projections_path_dino,
                               retrieval=retrieval, cam2tok_path=cam2tok_path, split='train',
                               labels_path=labels_path
                               )
    val_dataset = dataset_fn(data_path, imageset=val_imageset, label_mapping=label_mapping, nusc=nusc,
                             features_type=features_type, features_path=features_path,
                             projections_path=projections_path, class_agnostic=class_agnostic,
                             linear_probe=linear_probe, label_mapping_gt=label_mapping_gt,
                             dino_features=dino_features, features_type_dino=features_type_dino,
                             features_path_dino=features_path_dino, projections_path_dino=projections_path_dino,
                             retrieval=retrieval, cam2tok_path=cam2tok_path, split='val',
                             labels_path=labels_path
                             )
    test_dataset = dataset_fn(data_path, imageset=val_imageset.replace('_val_','_test_'), label_mapping=label_mapping, nusc=nusc,
                             features_type=features_type, features_path=features_path,
                             projections_path=projections_path, class_agnostic=class_agnostic,
                             linear_probe=linear_probe, label_mapping_gt=label_mapping_gt,
                             dino_features=dino_features, features_type_dino=features_type_dino,
                             features_path_dino=features_path_dino, projections_path_dino=projections_path_dino,
                             retrieval=retrieval, cam2tok_path=cam2tok_path, split='test',
                             labels_path=labels_path
                             )
    if merged:
        train_dataset.merge(val_dataset)
        train_dataset.merge(test_dataset)

    print(f'\nDataset sizes:\n\tTRAIN:\t{len(train_dataset)}\n\tVAL:\t{len(val_dataset)}')
    if merged:
        print(f'\tTEST: {len(test_dataset)}\n')
    else:
        print()


    wrapper_cls = DatasetWrapper_NuScenes_LinearGT if linear_gt else DatasetWrapper_NuScenes

    if train_noaug:
        train_dataset = wrapper_cls(
            train_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            fill_label=dataset_config["fill_label"],
            phase='train_noaug',
            scale_rate=scale_rate,
            gt_mode=gt_mode,
            ignore_label=ignore_label,
            linear_probe=linear_probe,
            with_projections=with_projections,
            unique_features=unique_features,
            eval_mode=False,  # eval_mode,
            semantic_points=semantic_points,
            max_features=max_features,
            fill_label_gt=fill_label_gt
        )
    else:
        train_dataset = wrapper_cls(
            train_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            fill_label=dataset_config["fill_label"],
            phase='train',
            scale_rate=scale_rate,
            gt_mode=gt_mode,
            ignore_label=ignore_label,
            linear_probe=linear_probe,
            with_projections=with_projections,
            unique_features=unique_features,
            eval_mode=False,  # eval_mode,
            semantic_points=semantic_points,
            max_features=max_features,
            fill_label_gt=fill_label_gt
        )

    val_dataset = wrapper_cls(
        val_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        fill_label=dataset_config["fill_label"],
        phase='val',
        scale_rate=scale_rate,
        ignore_label=ignore_label,
        linear_probe=linear_probe,
        with_projections=with_projections,
        unique_features=unique_features,
        semantic_points=semantic_points,
        eval_mode=eval_mode,
        max_features=max_features,
        fill_label_gt=fill_label_gt
    )

    test_dataset = wrapper_cls(
        test_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        fill_label=dataset_config["fill_label"],
        phase='test',
        scale_rate=scale_rate,
        ignore_label=ignore_label,
        linear_probe=linear_probe,
        with_projections=with_projections,
        unique_features=unique_features,
        semantic_points=semantic_points,
        eval_mode=eval_mode,
        max_features=max_features,
        fill_label_gt=fill_label_gt
    )


    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        if merged:
            merged_sampler = torch.utils.data.distributed.DistributedSampler(merged_dataset, shuffle=False, drop_last=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None
        merged_sampler = None
        test_sampler = None

    collate_fn = custom_collate_fn_linear_gt if linear_gt else custom_collate_fn

    collate_fn = None
    if linear_gt:
        collate_fn = custom_collate_fn_linear_gt
    elif dino_features:
        collate_fn = custom_collate_fn_dino
    else:
        collate_fn = custom_collate_fn

    num_workers_train = num_workers if num_workers is not None else train_dataloader_config["num_workers"]
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn,
                                                       shuffle=False if (dist or merged) else train_dataloader_config["shuffle"],
                                                       sampler=sampler,
                                                       num_workers=num_workers_train)

    if train_noaug:
        return train_dataset_loader

    num_workers_val = num_workers if num_workers is not None else val_dataloader_config["num_workers"]
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=val_sampler,
                                                     num_workers=num_workers_val)
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=test_sampler,
                                                     num_workers=num_workers_val)
    if return_test:
        return test_dataset_loader

    return train_dataset_loader, val_dataset_loader
