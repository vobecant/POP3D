import os

import numpy as np
from detectron2.structures import BitMasks
from detectron2.data.detection_utils import read_image
from segment_anything import SamPredictor, sam_model_registry
from torch import nn


class SAMPredictor(nn.Module):
    def __init__(self, sam_model, sam_path, multimask_output):
        super(SAMPredictor, self).__init__()
        sam = sam_model_registry[sam_model](checkpoint=sam_path).cuda()
        self.predictor = SamPredictor(sam)
        self.multimask_output = multimask_output

    def sam_single_image(self, image, points):
        self.predictor.set_image(image)
        pred_masks, scores = [], []
        points = points[:, None, [1, 0]]
        n_queries = points.shape[0]
        for point in points:
            masks_cur, confs, _ = self.predictor.predict(point_coords=point, point_labels=np.array([1]),
                                                         multimask_output=self.multimask_output
                                                         )
            pred_masks.extend(masks_cur)
            scores.extend(confs)

        pred_masks = np.row_stack(pred_masks)
        pred_masks = BitMasks(pred_masks)

        return pred_masks


if __name__ == '__main__':
    sam_model = "vit_h"
    sam_path = "/home/vobecant/PhD/weights/sam_vit_h_4b8939.pth"
    multimask_output = True

    sam_predictor = SAMPredictor(sam_model, sam_path, multimask_output)

    image_paths = [
        '/nfs/datasets/nuscenes/samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984239412460.jpg'
    ]
    projections_dir = '/nfs/datasets/nuscenes/features/projections'

    for path in image_paths:
        # load image
        image = read_image(path, format="BGR")
        # load points
        img_name = os.path.split(path)[-1].split('.')[0]
        cam_name = img_name.split('__')[1]
        assert cam_name.startswith('CAM_')
        points_path = os.path.join(projections_dir, cam_name, f'{img_name}__pixels.npy')
        points = np.load(points_path)

        # predict
        pred_masks = sam_predictor.sam_single_image(image, points)

        # visualize
        pass

        # save
        pass
