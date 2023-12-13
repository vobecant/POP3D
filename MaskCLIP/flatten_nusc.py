import os
import argparse
from tqdm import tqdm

CAM_NAMES = ['CAM_BACK','CAM_BACK_RIGHT','CAM_BACK_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc-root', type=str, default=None, required=True)
    args = parser.parse_args()

    save_dir = os.path.join(args.nusc_root, 'mmseg','images','trainvaltest_flatten')
    samples_dir = os.path.join(args.nusc_root, 'samples')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for cam_name in CAM_NAMES:
        cam_dir = os.path.join(samples_dir, cam_name)
        image_paths = [os.path.join(cam_dir, fname) for fname in os.listdir(cam_dir)]
        print(f'Found {len(image_paths)} paths in {cam_dir}, e.g., {image_paths[0]} .')

        for image_path in tqdm(image_paths):
            fname = os.path.split(image_path)[-1]
            tgt = os.path.join(save_dir, fname)
            os.symlink(image_path, tgt)
      
