import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil

PICKED = [
    '01176d20420b404ea52769d01ad61b0c',
    '0181321478f94908b4b668b47b492519',
    '02b68afcd7504587a003bffe899e4986',
    '02e64ee9e41b4e22898f027302a67849',
    '045ecf94b83648229500b23058af3117',
    '048a45dd2cf54aa5808d8ccc85731d44',
]


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def show3d_labels(ax, colors, labels, s, x, y, z):
    if labels is None:
        sc = ax.scatter(x, y, z, s=s)
    else:
        if colors is not None:
            for _unq in np.unique(labels):
                whr = labels == _unq
                # print(_unq)
                color = colors[_unq] / 255.
                # print(color)
                sc = ax.scatter(x[whr], y[whr], z[whr], color=color, s=s)
        else:
            sc = ax.scatter(x, y, z, c=labels, s=s, cmap='tab20')
    return sc


def show3d(points, fig, nrows, ncols, n, s=0.01, ax=None, colors=None, labels=None, ft_color=None, points_ft=None,
           title=None):
    d1, d2 = points.shape
    if d1 > d2:
        points = points.T

    if ax is None:
        ax = fig.add_subplot(nrows, ncols, n, projection='3d')

    try:
        x, y, z = points[:3]
    except:
        return ax

    if ft_color is None:
        # only labels
        sc = show3d_labels(ax, colors, labels, s, x, y, z)
        set_axes_equal(ax)
    else:
        ft_color = ft_color / 255
        if points_ft is not None:
            x, y, z = points_ft.cpu().T
        sc = ax.scatter(x, y, z, s=s, c=ft_color)
        set_axes_equal(ax)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_zaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])

    if title is not None:
        ax.set_title(title)

    return ax


def load_pc(fpath):
    xyz = []
    rgb = []
    with open(fpath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    for line in lines:
        try:
            x, y, z, r, g, b = map(float, line.split(' '))
            r, g, b = list(map(int, [r, g, b]))
            rgb.append([r, g, b])
        except:
            x, y, z, score = map(float, line.split(' '))
            rgb.append(score)
        xyz.append([x, y, z])

    xyz = np.array(xyz)
    rgb = np.array(rgb)
    return xyz, rgb


if __name__ == '__main__':
    dir_path = '/home/vobecant/PhD/TPVFormer-OpenSet/qualitative_trainval10_final_noIgnore_3xScale'
    dirs = sorted([os.path.join(dir_path, fpath) for fpath in os.listdir(dir_path)])[:40]

    picked_dir = '/home/vobecant/PhD/TPVFormer-OpenSet/qualitative_trainval10_final_noIgnore_3xScale_picked'
    if not os.path.exists(picked_dir):
        os.makedirs(picked_dir)

    paths = []
    for d in tqdm(dirs):
        token = d.split(os.path.sep)[-1]
        fpath = os.path.join(d, f'{token}.txt')
        paths.append(fpath)
        imname = os.path.split(fpath)[-1]
        title = imname.split('.')[0]
        if title in PICKED:
            shutil.copyfile(fpath,os.path.join(picked_dir,imname))
        continue
        try:
            xyz, rgb = load_pc(fpath)
        except:
            continue
        x, y, z = xyz.T

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        s = np.ones(len(x))
        ax.scatter(x, y, z, c=rgb / 255, s=s)
        set_axes_equal(ax)
        plt.suptitle(title)
        print(title)


        fpath = os.path.join(d, f'{token}_gt.txt')
        xyz, rgb = load_pc(fpath)
        x, y, z = xyz.T
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        s = np.ones(len(x))
        ax.scatter(x, y, z, c=rgb / 255, s=s)
        set_axes_equal(ax)

        plt.show()

    print(f'Saved to {picked_dir}')