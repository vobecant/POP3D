import random
from pathlib import Path

import mmcv
import numpy as np
from mmcv.utils import Config
from torch.utils.data import Dataset


class BaseMMSeg(Dataset):
    def __init__(
            self,
            image_size,
            crop_size,
            split,
            config_path,
            normalization='deit',
            **kwargs,
    ):
        super().__init__()

        from mmseg.datasets import build_dataset
        from mmseg.datasets.utils import STATS

        self.image_size = image_size
        self.crop_size = crop_size
        self.split = split
        self.normalization = STATS[normalization].copy()
        self.ignore_label = None
        for k, v in self.normalization.items():
            v = np.round(255 * np.array(v), 2)
            self.normalization[k] = tuple(v)
        print(f"Use normalization: {self.normalization}")

        config = Config.fromfile(config_path)

        self.ratio = config.max_ratio
        self.dataset = None
        self.config = self.update_default_config(config)
        self.dataset = build_dataset(getattr(self.config.data, f"{self.split}"))

    def update_default_config(self, config):

        acdc_train_splits = [f"train_{split}" for split in ['fog', 'night', 'rain', 'snow']]
        acdc_val_splits = [f"val_{split}" for split in ['fog', 'night', 'rain', 'snow']]
        train_splits = ["train", "trainval"] + acdc_train_splits
        if self.split in train_splits:
            config_pipeline = getattr(config, f"train_pipeline")
        else:
            if self.split in acdc_val_splits:
                split = self.split.split('_')[0]
            else:
                split = self.split
            config_pipeline = getattr(config, f"{split}_pipeline")

        img_scale = (self.ratio * self.image_size, self.image_size)
        if self.split not in train_splits:
            assert config_pipeline[1]["type"] == "MultiScaleFlipAug"
            config_pipeline = config_pipeline[1]["transforms"]
        for i, op in enumerate(config_pipeline):
            op_type = op["type"]
            if op_type == "Resize":
                op["img_scale"] = img_scale
            elif op_type == "RandomCrop":
                op["crop_size"] = (
                    self.crop_size,
                    self.crop_size,
                )
            elif op_type == "Normalize":
                op["mean"] = self.normalization["mean"]
                op["std"] = self.normalization["std"]
            elif op_type == "Pad":
                op["size"] = (self.crop_size, self.crop_size)
            config_pipeline[i] = op
        if self.split in acdc_train_splits:
            config.data.get(self.split).pipeline = config_pipeline
        elif self.split == "train":
            config.data.train.pipeline = config_pipeline
        elif self.split == "trainval":
            config.data.trainval.pipeline = config_pipeline
        elif self.split in acdc_val_splits:
            config.data.get(self.split).pipeline[1]["img_scale"] = img_scale
            config.data.get(self.split).pipeline[1]["transforms"] = config_pipeline
        elif self.split == "val":
            config.data.val.pipeline[1]["img_scale"] = img_scale
            config.data.val.pipeline[1]["transforms"] = config_pipeline
        elif self.split == "test":
            config.data.test.pipeline[1]["img_scale"] = img_scale
            config.data.test.pipeline[1]["transforms"] = config_pipeline
            config.data.test.test_mode = True
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return config

    def set_multiscale_mode(self):
        self.config.data.val.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.val.pipeline[1]["flip"] = True
        self.config.data.test.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.test.pipeline[1]["flip"] = True
        self.dataset = build_dataset(getattr(self.config.data, f"{self.split}"))

    def __getitem__(self, idx):
        data = self.dataset[idx]

        acdc_train_splits = [f"train_{split}" for split in ['fog', 'night', 'rain', 'snow']]
        train_splits = ["train", "trainval"] + acdc_train_splits

        if self.split in train_splits:
            im = data["img"].data
            seg = data["gt_semantic_seg"].data.squeeze(0)
        else:
            im = [im.data for im in data["img"]]
            seg = None

        out = dict(im=im)
        if self.split in train_splits:
            out["segmentation"] = seg
        else:
            im_metas = [meta.data for meta in data["img_metas"]]
            out["im_metas"] = im_metas
            out["colors"] = self.colors

        return out

    def get_gt_seg_maps(self, mapping=None, max_images=None, n_cls=None):
        dataset = self.dataset
        gt_seg_maps = {}
        kept_img_infos = []

        if max_images is not None:
            random.shuffle(dataset.img_infos)

        for img_info in dataset.img_infos:
            seg_map = Path(dataset.ann_dir) / img_info["ann"]["seg_map"]
            gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
            gt_seg_map[gt_seg_map == self.ignore_label] = IGNORE_LABEL
            if mapping is not None:
                gt_seg_map_remapped = gt_seg_map.copy()
                for old_id, new_id in mapping.items():
                    gt_seg_map_remapped[gt_seg_map == old_id] = new_id
                gt_seg_map = gt_seg_map_remapped
                del gt_seg_map_remapped
            if self.reduce_zero_label:
                zero_locations = gt_seg_map == 0
                gt_seg_map[gt_seg_map != IGNORE_LABEL] -= 1
                gt_seg_map[zero_locations] = IGNORE_LABEL
            gt_seg_maps[img_info["filename"]] = gt_seg_map
            kept_img_infos.append(img_info)
            if max_images is not None and len(gt_seg_maps) >= max_images:
                break
        dataset.img_infos = kept_img_infos
        return gt_seg_maps

    def __len__(self):
        return len(self.dataset)

    @property
    def unwrapped(self):
        return self

    def set_epoch(self, epoch):
        pass

    def get_diagnostics(self, logger):
        pass

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return
