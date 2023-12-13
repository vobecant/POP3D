import os.path as osp
import pickle
import warnings

import mmcv
from mmseg.datasets import CustomDataset, DATASETS


# from mmcv.builder import DATASETS

# DATASETS = Registry('dataset')


@DATASETS.register_module()
class CustomDatasetMy(CustomDataset):
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 img_infos_path=None,
                 label_mapping_path=None):
        self.img_infos_path = img_infos_path
        self.label_mapping = {25: 255}
        # if label_mapping_path is not None:
        #     self.label_mapping = {
        #         idx: new_idx for idx, new_idx in enumerate(np.load(label_mapping_path).astype(np.uint8) - 1)
        #     }
        super().__init__(pipeline, img_dir, img_suffix, ann_dir, seg_map_suffix, split, data_root, test_mode,
                         ignore_index, reduce_zero_label, classes, palette)

    def load_annotations(self, *args):
        with open(self.img_infos_path, 'rb') as f:
            img_infos = pickle.load(f)
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

        if self.label_mapping is not None:
            results['label_map'] = self.label_mapping
        elif self.custom_classes:
            results['label_map'] = self.label_map

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            yield gt_seg_map
