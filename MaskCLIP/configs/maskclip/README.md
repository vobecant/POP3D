# MaskCLIP: Extract Free Dense Labels from CLIP

## Abstract
<!-- [ABSTRACT] -->
Contrastive Language-Image Pre-training (CLIP) has made a remarkable breakthrough in open-vocabulary zero-shot image recognition. Many recent studies leverage the pre-trained CLIP models for image-level classification and manipulation. In this paper, we wish examine the intrinsic potential of CLIP for pixel-level dense prediction, specifically in semantic segmentation. To this end, with minimal modification, we show that MaskCLIP yields compelling segmentation results on open concepts across various datasets in the absence of annotations and fine-tuning. By adding pseudo labeling and self-training, MaskCLIP+ surpasses SOTA transductive zero-shot semantic segmentation methods by large margins, e.g., mIoUs of unseen classes on PASCAL VOC/PASCAL Context/COCO Stuff are improved from 35.6/20.7/30.3 to 86.1/66.7/54.7. We also test the robustness of MaskCLIP under input corruption and evaluate its capability in discriminating fine-grained objects and novel concepts. Our finding suggests that MaskCLIP can serve as a new reliable source of supervision for dense prediction tasks to achieve annotation-free segmentation.

## Results and models of annotation-free segmentation

#### Pascal VOC 2012 + Aug (w/o Background)

| Method    | CLIP Model | Crop Size | mIoU | config                                           |
| --------- | ---------- | --------- |----: | -------------------------------------------------|
| MaskCLIP | R-50       | 512x512   | 41.5 | [config](maskclip_r50_512x512_voc12aug_20.py)   |
| MaskCLIP | R-50x16    | 512x512   | 50.9 | [config](maskclip_r50x16_512x512_voc12aug_20.py)|
| MaskCLIP | ViT-B/16   | 512x512   | 49.5 | [config](maskclip_vit16_512x512_voc12aug_20.py)|


#### Pascal Context (w/o Background)

| Method    | CLIP Model | Crop Size | mIoU | config                                                 |
| --------- | ---------- | --------- |----: | -------------------------------------------------------|
| MaskCLIP | R-50       | 520x520   | 18.5 | [config](maskclip_r50_520x520_pascal_context_59.py)   |
| MaskCLIP | R-50x16    | 520x520   | 20.3 | [config](maskclip_r50x16_520x520_pascal_context_59.py)|
| MaskCLIP | ViT-B/16   | 520x520   | 21.7 | [config](maskclip_vit16_520x520_pascal_context_59.py)|


#### COCO-Stuff 164k

| Method    | CLIP Model | Crop Size | mIoU | config                                              |
| --------- | ---------- | --------- |----: | ----------------------------------------------------|
| MaskCLIP | R-50       | 512x512   | 10.2 | [config](maskclip_r50_512x512_coco-stuff164k.py)   |
| MaskCLIP | R-50x16    | 512x512   | 13.6 | [config](maskclip_r50x16_512x512_coco-stuff164k.py)|
| MaskCLIP | ViT-B/16   | 512x512   | 12.5 | [config](maskclip_vit16_512x512_coco-stuff164k.py)|


#### Pascal VOC 2012 + Aug (Under Corruption)

| Method    | CLIP Model | Crop Size | Corrupt Type | Corrupt Level | mIoU | config                                                   |
| --------- | ---------- | --------- | ------------ | ------------- |----: | ---------------------------------------------------------|
| MaskCLIP | R-50       | 512x512   | Speckle Noise| 1             | 30.9 | [config](corrupt/maskclip_r50_512x512_voc12aug_20_corrupt.py)   |
