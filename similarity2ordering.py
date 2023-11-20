import argparse
import os

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--txt-path', default=None)
    args = parser.parse_args()

    xyz = []
    sim = []
    with open(args.txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        for l in lines:
            x, y, z, _sim = map(float, l.split(' '))
            xyz.append((x, y, z))
            sim.append(_sim)

    sorted2position = np.argsort(sim)
    position2sorted = -np.ones_like(sorted2position)
    position2sorted[sorted2position] = np.arange(len(sorted2position))

    d,filename = os.path.split(args.txt_path)
    filename = filename.split('.')[0]
    save_path = os.path.join(d,filename+'_sorted.txt')
    with open(save_path,'w') as f:
        for _xyz, _idx in zip(xyz, position2sorted):
            x,y,z = _xyz
            f.write(f'{x} {y} {z} {_idx}\n')
    print(save_path)

