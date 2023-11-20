import numpy as np
import os
import shutil
import tarfile
from spec2prompt import SPEC2PROMPT
from tqdm import tqdm
from benchmark_retrieval_final import get_args,get_img2scene_lut,get_dataloader,limit_dataloader
try:
    from mmcv import Config
except:
    from mmengine.config import Config

SPLITS=['train','val','test','eval','all']
PROJ_PATH="/nfs/datasets/nuscenes/features/projections"
BENCHMARK_TGT_DIR = "/nfs/datasets/nuscenes/retrieval_benchmark"

BENCHMARK_RELEASE_DIR = "/nfs/datasets/nuscenes/retrieval_benchmark_release"
if not os.path.exists(BENCHMARK_RELEASE_DIR):
    os.makedirs(BENCHMARK_RELEASE_DIR)

BENCHMARK_RELEASE_DIR_POINTS = "/nfs/datasets/nuscenes/retrieval_benchmark_release/matching_points"
if not os.path.exists(BENCHMARK_RELEASE_DIR_POINTS):
    os.makedirs(BENCHMARK_RELEASE_DIR_POINTS)

BENCHMARK_RELEASE_DIR_ANNOTATIONS = "/nfs/datasets/nuscenes/retrieval_benchmark_release/annotations"
if not os.path.exists(BENCHMARK_RELEASE_DIR_ANNOTATIONS):
    os.makedirs(BENCHMARK_RELEASE_DIR_ANNOTATIONS)

# for split in SPLITS:
#     split_dir = os.path.join(BENCHMARK_RELEASE_DIR, split)
#     if not os.path.exists(split_dir):
#         os.makedirs(split_dir)

BENCHMARK_RELEASE_FILE = "/nfs/datasets/nuscenes/retrieval.tar.gz"


def make_tarfile(output_filename, source_dir):
    print(f'\nCompressing {source_dir} to {output_filename}')
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print(f'DONE!\n')


def get_data_info(info):
    input_dict = dict(
        sample_idx=info['token'],
        pts_filename=info['lidar_path'],
        sweeps=info['sweeps'],
        ego2global_translation=info['ego2global_translation'],
        ego2global_rotation=info['ego2global_rotation'],
    )

    image_paths = []
    lidar2img_rts = []
    lidar2cam_rts = []
    cam_intrinsics = []
    for cam_type, cam_info in info['cams'].items():
        image_paths.append(cam_info['data_path'])
        # obtain lidar to image transformation matrix
        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
        lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)
        lidar2img_rts.append(lidar2img_rt)

        cam_intrinsics.append(viewpad)
        lidar2cam_rts.append(lidar2cam_rt.T)

    input_dict.update(
        dict(
            img_filename=image_paths,
            lidar2img=lidar2img_rts,
            cam_intrinsic=cam_intrinsics,
            lidar2cam=lidar2cam_rts,
        ))

    return input_dict


if __name__ == '__main__':
    args = get_args()
    cfg = Config.fromfile(args.py_config)

    new_infos, lut, lut_split, *_ = get_img2scene_lut()
    merged_dataloader, val_dataloader = get_dataloader(cfg, retrieval=True, no_nusc=True, merged=True)
    val_dataloader = merged_dataloader

    specs = [t[0] for t in SPEC2PROMPT]
    text_queries = [t[1] for t in SPEC2PROMPT]

    val_dataloader, image_paths = limit_dataloader(val_dataloader, specs, lut, new_infos)
    new_infos_loader = val_dataloader.dataset.imagepoint_dataset.nusc_infos
    splits = [lut_split[info['cams']['CAM_FRONT']['data_path'].split('/')[-1].split('.')[0]] for info in val_dataloader.dataset.imagepoint_dataset.nusc_infos]
    infos = [info for info in val_dataloader.dataset.imagepoint_dataset.nusc_infos]
    
    txt = []
    valid_count = 0

    txt_per_split = {split:[] for split in SPLITS}
    for info,split,spec,query in tqdm(zip(infos, splits, specs, text_queries),total=len(infos)):
        token = info['token']
        ann_fname = os.path.join( f'{spec}__retrieval.npy')
        ann_fpath = os.path.join(BENCHMARK_TGT_DIR, ann_fname)
        if not os.path.exists(ann_fpath):
            continue

        matching_points = []

        imgs_info = get_data_info(info)

        for filename in imgs_info['img_filename']:
            im_name = os.path.split(filename)[-1].split('.')[0] 
            camera_name = im_name.split('__')[1]
            # corresponding points
            matching_points_cam_path = os.path.join(PROJ_PATH, camera_name, f"{im_name}__points.npy")
            matching_points_cam = np.load(matching_points_cam_path)
            matching_points.append(matching_points_cam)
        matching_points = np.concatenate(matching_points)
        
        
        ann_points_fpath = os.path.join(BENCHMARK_TGT_DIR, f"{token}__points.npy")
        ann_points_fname = os.path.join('matching_points', os.path.split(ann_points_fpath)[-1])
        np.save(ann_points_fpath, matching_points)

        # try:
        ann_release_fpath = os.path.join(BENCHMARK_RELEASE_DIR_ANNOTATIONS, f'{token}__retrieval.npy')
        shutil.copyfile(ann_fpath, ann_release_fpath)

        ann_points_release_fpath = os.path.join(BENCHMARK_RELEASE_DIR_POINTS, f"{token}__points.npy")
        shutil.copyfile(ann_points_fpath, ann_points_release_fpath)

        ann_fname_release = os.path.join('annotations',os.path.split(ann_release_fpath)[-1])
        line = f'{token};{split};{ann_fname_release};{ann_points_fname};{query}'
        txt.append(line)
        txt_per_split['all'].append(line)
        txt_per_split[split].append(line)
        if split in ['val','test']:
            txt_per_split['eval'].append(line)
        print(line)

        valid_count += 1
        # except:
        #     print(f'invalid token {token}')

    print(f'valid: {valid_count}')

    make_tarfile(BENCHMARK_RELEASE_FILE, BENCHMARK_RELEASE_DIR)

    for split in SPLITS:
        csv_file = os.path.join(BENCHMARK_RELEASE_DIR, f'retrieval_anns_{split}.csv')    
        with open(csv_file, 'w') as f:
            for l in txt_per_split[split]:
                f.write(f'{l}\n')
        print(f'Saved data for split {split} to {csv_file}')