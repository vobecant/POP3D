# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
import numpy as np

DISTINCT_COLORS = [[255, 179, 0], [128, 62, 117], [255, 104, 0], [166, 189, 215], [193, 0, 32], [206, 162, 98], [129, 112, 102], [0, 125, 52], [246, 118, 142], [0, 83, 138], [255, 122, 92], [83, 55, 122], [255, 142, 0], [179, 40, 81], [244, 200, 0], [127, 24, 13], [147, 170, 0], [89, 51, 21], [241, 58, 19], [35, 44, 22]]

@DATASETS.register_module()
class MaskCLIPDemoDataset(CustomDataset):
    def __init__(self, **kwargs):
        fg_classes = kwargs.pop('fg_classes', None)
        bg_classes = kwargs.pop('bg_classes', None)

        MaskCLIPDemoDataset.CLASSES = fg_classes + bg_classes

        if len(fg_classes) <= len(DISTINCT_COLORS):
            MaskCLIPDemoDataset.PALETTE = DISTINCT_COLORS[:len(fg_classes)] \
                + [[0 ,0, 0]] * len(bg_classes)
        else:
            state = np.random.get_state()
            np.random.seed(42)
            MaskCLIPDemoDataset.PALETTE = DISTINCT_COLORS \
                + np.random.randint(0, 255, size=(len(fg_classes)-len(DISTINCT_COLORS), 3)).tolist() \
                + [[0 ,0, 0]] * len(bg_classes)
            np.random.set_state(state)

        super(MaskCLIPDemoDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        pass