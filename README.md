# Code for paper "POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images"

Welcome to the code for paper "POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images" ([project page](https://vobecant.github.io/POP3D/))

## environment setup
Please, have GCC 5 or higher.

### POP-3D
Run the following script to prepare the `pop3d` conda environment:
```shell
conda env create -f conda_env.yaml
```
Download weights from [this link](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  and put them to `./ckpts`

### MaskCLIP
**Step 0.** Create a conda environment, activate it and install requirements
```shell
cd MaskCLIP
conda create -n maskclip python=3.9
conda activate maskclip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir opencv-python
```

**Step 1.** Install PyTorch and Torchvision following [official instructions](https://pytorch.org/get-started/locally/), e.g., fo4 PyTorch 1.10 with CUDA 10.2:

```shell
pip install --no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

**Step 2.** Install MMCV:
```shell
pip install --no-cache-dir mmcv-full==1.5.0
```

**Step 3.** Install [CLIP](https://github.com/openai/CLIP).
```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

**Step 4.** Install MaskCLIP.
```shell
pip install --no-cache-dir -v -e .
```

## Data preparation

### Download nuScenes
Download and extract the nuScenes dataset ([link](https://www.nuscenes.org/nuscenes#download 'Link to the nuScenes downloads page.')) and **place it to the `./data/nuscenes` folder**. This means downloading all the nuScenes files, including both *trainval* and test *splits*, 

### Download "info" files:
We provide files for simpler manipulation with the nuScenes dataset. We use these files in our dataloaders. Again, please put these files to the `./data` folder (in the `POP3D` folder). To do this, please simply run:
```
bash scripts/download_info_files.sh
```

### Download retrieval benchmark files:
To download the data for our *Open-vocabulary language-driven retrieval dataset*, please run:
```
bash scripts/download_retrieval_benchmark.sh
```


### Prepare projection files.
To activate the environment, please run:
```
conda activate pop3d
```
Run the following script to prepare projection files. The default path to the directory with the nuScenes dataset is set to `./data/nuscenes`.
```
NUSC_ROOT=./data/nuscenes
PROJ_DIR=./data/nuscenes/features/projections
python3 generate_projections_nuscenes.py --nusc_root ${NUSC_ROOT} --proj-dir ${PROJ_DIR}
```


### Generate MaskCLIP features.
**Switch** to MaskCLIP directory in this project (`cd MaskCLIP`).

0) Activate MaskCLIP environment:
```shell
conda activate maskclip
```

1) Prepare backbone weights by:
```
mkdir -p ./pretrain
python tools/maskclip_utils/convert_clip_weights.py --model ViT16 --backbone
python tools/maskclip_utils/convert_clip_weights.py --model ViT16
```

2) Download pre-trained weights from [this link](https://entuedu-my.sharepoint.com/:u:/g/personal/chong033_e_ntu_edu_sg/EZSrnPaNFwNNqpECYCkgpg4BEjN782MUD7ZUEPXFWSTEXA?e=mOaseS) and put them to `ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth`

3) Run feature extraction:
```bash
CFG_PATH=configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainvaltest.py
CKPT_PATH=ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth
PROJ_DIR=../data/nuscenes/features/projections/data/nuscenes
OUT_DIR=../data/nuscenes/maskclip_features_projections
python tools/extract_features.py ${CFG_PATH} --save-dir ${OUT_DIR} --checkpoint ${CKPT_PATH} --projections-dir ${PROJ_DIR} --complete
```
to generate the target MaskCLIP+ features to use in the training of our method. 

*Note*: the process of preparing the targets from MaskCLIP+ can be slow, depending on the speed of your file system. If you want to parallelize, we provide the following skeleton for launching multiple jobs using SLURM:
```bash:
NUM_GPUS=... # fill the number of nodes
ACCOUNT=... # name of your account, if any
HOURS_TOTAL=... # how long you expect the *WHOLE* extraction of features to last
MASKCLIP_DIR=/path/to/POP3D/MaskCLIP
bash generate_features_slurm.sh ${NUM_GPUS} ${HOURS_TOTAL} ${ACCOUNT} ${MASKCLIP_DIR}
``` 

## Training

Our model was trained on 8x NVIIDA A100 GPUs.

Please, modify the following variables in the training script ``:
```shell
PARTITION="..." # name of the parition on your cluser, e.g., "gpu"
ACCOUNT="..." # name of your account, if it is set on your cluster
USERNAME="..." # your username on the cluster, used just for printing of running jobs
```

Script to run the training using SLURM: **(NOT WORKING YET)**
```bash
POP3D_DIR=/path/to/POP3D
bash scripts/train_slurm.sh ${POP3D_DIR} 
```

## Pre-trained weights
Weights used for results in the paper are [here](https://data.ciirc.cvut.cz/public/projects/2023POP3D/pop3d_weights.pth) and used zero-shot weights from [here](https://data.ciirc.cvut.cz/public/projects/2023POP3D/zeroshot_weights.pth). Please, put both files to `${POP3D_DIR}/pretrained` folder for easier use.

## Evaluation
### Zero-shot open-vocabulary semantic segmentation
To obtain results from our paper, please run:

A) single-GPU (slow):
```shell
CFG=...
CKPT=...
ZEROSHOT_PTH=...
 python3 eval_maskclip.py --py-config ${CFG} --resume-from ${CKPT} --maskclip --no-wandb --text-embeddings-path ${ZEROSHOT_PTH}
```
If you followed the instructions above, you can run:
```shell
 python3 eval_maskclip.py --py-config config/pop3d_maskclip_12ep.py --resume-from ./pretrained/pop3d_weights.pth --maskclip --no-wandb --text-embeddings-path ./pretrained/zeroshot_weights.pth
```

B) multi-GPU using SLURM (faster), e.g.:
```shell
POP3D_DIR=`pwd`
CKPT="./pretrained/pop3d_weights.pth"
NUM_GPUS=8
HOURS=1
CFG="config/pop3d_maskclip_12ep.py"
EXTRA="--text-embeddings-path ./pretrained/zeroshot_weights.pth"
bash scripts/eval_zeroshot_slurm.sh ${POP3D_DIR} ${CKPT} ${NUM_GPUS} ${HOURS} ${CFG} ${EXTRA}
```

**EXPECTED RESULTS**:
```
val_miou_vox_clip_all (evaluated at the complete voxel space): 16.65827465887346
```

### Open-vocabulary language-driven retrieval
To obtain results from our paper, please run:
```shell
python retrieval.py
```
Expected results:
```
+-------------------------------+
|       train (42 samples)      |
+----------+------+-------------+
|  method  | mAP  | mAP visible |
+----------+------+-------------+
|  POP3D   | 15.3 |     15.6    |
| MaskCLIP | N/A  |     13.5    |
+----------+------+-------------+


+-------------------------------+
|        val (27 samples)       |
+----------+------+-------------+
|  method  | mAP  | mAP visible |
+----------+------+-------------+
|  POP3D   | 24.1 |     24.7    |
| MaskCLIP | N/A  |     18.7    |
+----------+------+-------------+


+-------------------------------+
|       test (36 samples)       |
+----------+------+-------------+
|  method  | mAP  | mAP visible |
+----------+------+-------------+
|  POP3D   | 12.6 |     13.6    |
| MaskCLIP | N/A  |     12.0    |
+----------+------+-------------+


+-------------------------------+
|      valtest (63 samples)     |
+----------+------+-------------+
|  method  | mAP  | mAP visible |
+----------+------+-------------+
|  POP3D   | 17.5 |     18.4    |
| MaskCLIP | N/A  |     14.9    |
+----------+------+-------------+
```
Results will be written to `./results/results_${TIMESTAMP}.txt` and to `./results/results_tables_${TIMESTAMP}.txt`



### Acknowledgements
Our code is based on [TPVFormer](https://github.com/wzzheng/TPVFormer) and [MaskCLIP](https://github.com/chongzhou96/MaskCLIP). Many thanks to the authors!
