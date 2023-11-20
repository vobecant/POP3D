import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir-path', default=None)
    args = parser.parse_args()

    dir_path = args.dir_path
    token = dir_path.split(os.path.sep)[-1]

    try:
        image_paths = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg') and 'CAM' in f
        ]
        image_paths = sorted([os.readlink(p) for p in image_paths])
    except:
        image_paths = sorted([
            os.readlink(os.path.join(dir_path, '../' + f)) for f in os.listdir(dir_path) if f.endswith('.jpg')
        ])

    top_idx = [4, 3, 5]
    bottom_idx = [1, 0, 2]

    top_images = np.concatenate([np.array(Image.open(image_paths[idx])) for idx in top_idx], axis=1)
    bottom_images = [np.array(Image.open(image_paths[idx])) for idx in bottom_idx]
    bottom_images = [np.fliplr(im) for im in bottom_images]
    bottom_images = np.concatenate(bottom_images, axis=1)
    images = np.concatenate((top_images, bottom_images), axis=0)

    save_path = os.path.join(dir_path, f'{token}.jpg')
    Image.fromarray(images).save(save_path)
    print(f'Saved to {save_path}')

    plt.figure(figsize=(10, 4))
    plt.imshow(images)
    plt.show()
