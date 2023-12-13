import numpy as np

try:
    import cityscapesscripts.helpers.labels as CSLabels
except:
    pass

from pathlib import Path
from mmseg.datasets.base import BaseMMSeg
from mmseg.datasets import utils, DATASETS

NUSCENES_CONFIG_PATH = Path(__file__).parent / "config" / "nuscenes.py"
NUSCENES_CATS_PATH = Path(__file__).parent / "config" / "nuscenes.yml"

name_mapping = {'noise': 'ignore',
                'human.pedestrian.adult': 'adult pedestrian',
                'human.pedestrian.child': 'child pedestrian',
                'human.pedestrian.wheelchair': 'wheelchair',
                'human.pedestrian.stroller': 'stroller',
                'human.pedestrian.personal_mobility': 'personal mobility',
                'human.pedestrian.police_officer': 'police officer',
                'human.pedestrian.construction_worker': 'construction worker',
                'animal': 'animal',
                'vehicle.car': 'car',
                'vehicle.motorcycle': 'motorcycle',
                'vehicle.bicycle': 'bicycle',
                'vehicle.bus.bendy': 'bendy bus',
                'vehicle.bus.rigid': 'rigid bus',
                'vehicle.truck': 'truck',
                'vehicle.construction': 'construction vehicle',
                'vehicle.emergency.ambulance': 'ambulance vehicle',
                'vehicle.emergency.police': 'police car',
                'vehicle.trailer': 'trailer',
                'movable_object.barrier': 'barrier',
                'movable_object.trafficcone': 'traffic cone',
                # 'movable_object.pushable_pullable': 'ignore',
                'movable_object.debris': 'debris',
                'static_object.bicycle_rack': 'bicycle rack',
                'flat.driveable_surface': 'driveable surface',
                'flat.sidewalk': 'sidewalk',
                'flat.terrain': 'terrain',
                'flat.other': 'other flat',
                'static.manmade': 'manmade',
                'static.vegetation': 'vegetation',
                # 'static.other': 'ignore',
                # 'vehicle.ego': 'ignore'
                }


@DATASETS.register_module()
class NuscenesDataset(BaseMMSeg):
    CLASSES = ('')

    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(image_size, crop_size, split, NUSCENES_CONFIG_PATH, **kwargs)
        self.names, self.colors = utils.dataset_cat_description(NUSCENES_CATS_PATH)
        self.n_cls = 16
        self.ignore_label = 255
        self.reduce_zero_label = False
        self.label_map = {25: 255}

    def update_default_config(self, config):
        # root_dir = dataset_dir()
        # path = Path(root_dir) / "nuscenes"
        # config.data_root = path
        #
        # config.data[self.split]["data_root"] = path
        # config = super().update_default_config(config)

        return config

    def test_post_process(self, labels):
        labels_copy = np.copy(labels)
        # cats = np.unique(labels_copy)
        # for cat in cats:
        #     labels_copy[labels == cat] = CSLabels.trainId2label[cat].id
        return labels_copy
