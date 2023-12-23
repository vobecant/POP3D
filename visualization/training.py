import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.decomposition import PCA
from torchvision import transforms

from dataloader.dataset_wrapper import img_norm_cfg

STANDARDIZE_PRE_PCA = False
CLASS_COLORS = np.array(
    [
        [0, 0, 0, 255],  # ignore
        [255, 120, 50, 255],  # barrier              orange
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [255, 127, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],  # ego car
        [0, 191, 255, 255]  # ego car
    ]
).astype(np.uint8)

CLASS_COLORS_TINY = np.array(
    [
        [0, 0, 0, 255],  # ignore
        [0, 150, 245, 255],  # car                  blue
        [255, 0, 255, 255],  # driveable_surface    dark pink
        [255, 0, 0, 255],  # pedestrian           red
        [0, 175, 0, 255],  # vegetation           green
        [230, 230, 250, 255],  # manmade              white
    ]
).astype(np.uint8)

AGNOSTIC_COLORS = np.array(
    [
        [0, 0, 0, 255],
        [0, 0, 0, 255]
    ]
).astype(np.uint8)


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


def show3d_labels(ax, colors, labels, s, x, y, z, cmap=None,title=None):
    if labels is None:
        sc = ax.scatter(x, y, z, s=s)
    else:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if colors is not None:
            for _unq in np.unique(labels):
                whr = labels == _unq
                # print(_unq)
                color = colors[_unq] / 255.
                # print(color)
                sc = ax.scatter(x[whr], y[whr], z[whr], color=color, s=s)
        else:
            if cmap is None:
                if labels.dtype in [float, np.float]:
                    # scalar values
                    cmap = 'bwr'
                else:
                    cmap = 'tab20'
            sc = ax.scatter(x, y, z, c=labels, s=s, cmap=cmap)
        if title is not None:
            ax.set_title(title)
    return sc


def show3d(points, fig, nrows, ncols, n, s=0.01, ax=None, colors=None, labels=None, ft_color=None, points_ft=None,
           cmap_name=None, title=None):
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
        sc = show3d_labels(ax, colors, labels, s, x, y, z, cmap=cmap_name,title=title)
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


def torch2npy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


def show3d_wandb(points, name, global_step, colors=None, labels=None, ft_color=None, debug=False):
    d1, d2 = points.shape
    if max(d1, d2) < 4:
        return
    if d1 < d2:
        points = points.T

    target_name = f"point_cloud/{name}"
    points = torch2npy(points)

    if not len(points.flatten()) > 0:
        points = np.zeros((1, 3))
        wandb.log({target_name: wandb.Object3D(points)}, commit=False)  # step=global_step)
        if debug: print(f'Logged into {target_name}!')
        return

    if ft_color is not None:
        points_new = np.zeros((points.shape[0], 6))
        points_new[:, :3] = points
        ft_color = ft_color[:len(ft_color), :3]
        points_new[:, 3:] = ft_color
        points = points_new
    elif labels is not None:
        if colors is not None:
            points_new = np.zeros((points.shape[0], 6))
            points_new[:, :3] = points
            for _unq in np.unique(labels):
                whr = labels == _unq
                # print(_unq)
                color = colors[_unq][:3]
                # print(color)
                points_new[whr, 3:] = color
            points = points_new
        else:
            points_new = np.zeros((points.shape[0], 4))
            points_new[:, :3] = points
            points_new[:, 3] = labels
            points = points_new

    wandb.log({target_name: wandb.Object3D(points)},
              # step=global_step,
              commit=False)
    if debug: print(f'Logged into {target_name}!')


def plot_comparison_occupancy(gt_grid, pred_grid, free_label, ignore_label, agnostic):
    fig = plt.figure()

    if agnostic:
        nrows = 2
        ncols = 2
        gt_occ = torch.stack(torch.where(torch.bitwise_and(gt_grid > 0, gt_grid < ignore_label)))
        pred_occ = torch.stack(torch.where(torch.bitwise_and(pred_grid > 0, pred_grid < ignore_label)))
        show3d(gt_occ, fig, nrows, ncols, 1, labels=np.ones(gt_occ.shape[1], dtype=np.uint8), colors=AGNOSTIC_COLORS)
        show3d(pred_occ, fig, nrows, ncols, 2, labels=np.ones(pred_occ.shape[1], dtype=np.uint8),
               colors=AGNOSTIC_COLORS)
    else:
        nrows = 3
        ncols = 2
        gt_grid_agnostic = torch.tensor(torch.where(torch.bitwise_and(gt_grid != free_label, gt_grid != ignore_label)))
        pred_grid_agnostic = torch.tensor(torch.where(torch.bitwise_and(pred_grid != free_label,
                                                                        pred_grid != ignore_label)))
        # [[GT    , pred    ],
        show3d(gt_grid, fig, nrows, ncols, 1, colors=CLASS_COLORS)
        show3d(pred_grid, fig, nrows, ncols, 2, colors=CLASS_COLORS)
        #  [GT agn, pred agn]]
        show3d(gt_grid_agnostic, fig, nrows, ncols, 3, colors=AGNOSTIC_COLORS)
        show3d(pred_grid_agnostic, fig, nrows, ncols, 4, colors=AGNOSTIC_COLORS)

    return fig, nrows, ncols


def plot_comparison_occupancy_semantics_wandb(occ_grid, free_label, ignore_label, split, global_step, mode):
    occupied = torch.stack(torch.where(torch.bitwise_and(occ_grid != free_label,
                                                         occ_grid != ignore_label)))
    occ_labels = occ_grid[occupied[0], occupied[1], occupied[2]]
    occ_grid = occupied
    name = f'{split}/{mode}_occupancy_semantics'
    show3d_wandb(occ_grid, name, global_step, colors=CLASS_COLORS, labels=occ_labels)


def plot_comparison_occupancy_wandb(occ_grid, free_label, ignore_label, agnostic, split, global_step, mode,
                                    debug=False, gt_cls_labels=None):
    if agnostic:
        gt_occ = torch.stack(torch.where(torch.bitwise_and(occ_grid > 0, occ_grid < ignore_label)))
    else:
        gt_occ = torch.stack(torch.where(torch.bitwise_and(occ_grid != free_label,
                                                           occ_grid != ignore_label)))
        # name = f'{split}/{mode}_occupancy'
        # show3d_wandb(gt_occ, name, global_step, colors=CLASS_COLORS, debug=debug)
        #  [GT agn, pred agn]]
        # name = f'{split}/{mode}_occupancy_agnostic'
        # show3d_wandb(gt_occ, name, global_step, colors=AGNOSTIC_COLORS, debug=debug)
    name = f'{split}/{mode}_occupancy_agnostic'
    show3d_wandb(gt_occ, name, global_step, labels=np.ones(gt_occ.shape[1], dtype=np.uint8),
                 colors=AGNOSTIC_COLORS, debug=debug)

    if gt_cls_labels is not None:
        name = f'{split}/{mode}_semantic'
        points = torch.stack(torch.where(gt_cls_labels < 17))
        labels = gt_cls_labels[points[0], points[1], points[2]]
        show3d_wandb(points, name, global_step, labels=labels, colors=CLASS_COLORS, debug=debug)


def pca_rgb_projection(features, standardize=STANDARDIZE_PRE_PCA,
                       mean=None, std=None, pca=None, reuse_pca=None):
    import cv2
    # features are of shape [N,dim]
    # X = features.transpose(1, 2, 0)
    # print(f'features.shape: {features.shape}')
    ndim = len(features.shape)
    if ndim == 4:
        # shape [B, C, H, W]
        features = features[0]
        C, H, W = features.shape
        X = features.view(C, H * W).T
    elif ndim == 3:
        C, H, W = features.shape
        X = features.view(C, H * W).T
    else:
        X = features
        H, W = None, None

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # Reshape to have shape (nb_pixel x dim)
    Xt_all = X
    Xt = Xt_all

    # print(f'Xt.shape: {Xt.shape}')

    # Normalize
    if standardize:
        if not reuse_pca or mean is None:
            mean = np.mean(Xt, axis=-1)[:, np.newaxis]
        if not reuse_pca or std is None:
            std = np.std(Xt, axis=-1)[:, np.newaxis]
        Xtn = (Xt - mean) / std
        Xtn_all = Xtn
    else:
        mean = std = None
        Xtn = Xt
        Xtn_all = Xt_all

    Xtn = np.nan_to_num(Xtn)
    Xtn_all = np.nan_to_num(Xtn_all)

    nb_comp = 3
    if not reuse_pca or pca is None:
        # Apply PCA
        pca = PCA(n_components=nb_comp)
        pca.fit(Xtn)
    else:
        # print('Using precomputed PCA.')
        pass
    projected = pca.fit_transform(Xtn_all)
    projected = projected.reshape(X.shape[0], nb_comp)

    # normalizing between 0 to 255
    PC_n = np.zeros((X.shape[0], nb_comp))
    for i in range(nb_comp):
        PC_n[:, i] = cv2.normalize(projected[:, i],
                                   np.zeros((X.shape[0])), 0, 255, cv2.NORM_MINMAX).squeeze().clip(0, 256)
    PC_n = PC_n.astype(np.uint8)

    if H is not None and W is not None:
        PC_n.resize(H, W, 3)

    if not reuse_pca:
        mean = std = pca = None
    return PC_n, (mean, std, pca)


def plot_comparison_features(fig, gt_features, pred_features, points, reuse_pca, nrows, ncols):
    ft_loc = points
    # print(f'pred_features.shape: {pred_features.shape}, ft_loc.shape: {ft_loc.shape}')
    gt_ft_color, (mean, std, pca) = pca_rgb_projection(gt_features, reuse_pca=reuse_pca)
    pred_ft_color, *_ = pca_rgb_projection(pred_features, mean=mean, std=std, pca=pca)

    show3d(ft_loc, fig, nrows, ncols, nrows * ncols - 1, ft_color=gt_ft_color)
    show3d(ft_loc, fig, nrows, ncols, nrows * ncols, ft_color=pred_ft_color)
    return fig


def plot_comparison_features_wandb(gt_features, pred_features, points, reuse_pca, split, global_step, debug=False):
    ft_loc = points
    # print(f'pred_features.shape: {pred_features.shape}, ft_loc.shape: {ft_loc.shape}')
    gt_ft_color, (mean, std, pca) = pca_rgb_projection(gt_features, reuse_pca=reuse_pca)
    pred_ft_color_reused, *_ = pca_rgb_projection(pred_features, mean=mean, std=std, pca=pca)
    pred_ft_color_ownPCA, *_ = pca_rgb_projection(pred_features)

    name = f'{split}/gt_features_pca'
    show3d_wandb(ft_loc, name, global_step, ft_color=gt_ft_color, debug=debug)
    name = f'{split}/pred_features_pca_reused'
    show3d_wandb(ft_loc, name, global_step, ft_color=pred_ft_color_reused, debug=debug)
    name = f'{split}/pred_features_pcaOwn'
    show3d_wandb(ft_loc, name, global_step, ft_color=pred_ft_color_ownPCA, debug=debug)


def plot_comparison(gt_grid, pred_grid, free_label, ignore_label, agnostic, gt_features=None, pred_features=None,
                    points=None, reuse_pca=True):
    fig, nrows, ncols = plot_comparison_occupancy(gt_grid, pred_grid, free_label, ignore_label, agnostic)
    fig = plot_comparison_features(fig, gt_features, pred_features, points, reuse_pca, nrows, ncols)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def log_comparison_wandb(images, gt_grid, pred_grid, free_label, ignore_label, agnostic, split, global_step,
                         gt_features=None, pred_features=None, points=None, reuse_pca=True, debug=False,
                         gt_cls_labels=None, commit=False):
    if debug: print('In WandB logging!')
    # occupancy
    plot_comparison_occupancy_wandb(gt_grid, free_label, ignore_label, agnostic, split, global_step, mode='gt',
                                    debug=debug, gt_cls_labels=gt_cls_labels)
    plot_comparison_occupancy_wandb(pred_grid, free_label, ignore_label, agnostic, split, global_step, mode='pred',
                                    debug=debug)
    if not agnostic:
        plot_comparison_occupancy_semantics_wandb(gt_grid, free_label, ignore_label, split, global_step, mode='gt')
        plot_comparison_occupancy_semantics_wandb(pred_grid, free_label, ignore_label, split, global_step, mode='pred')
    # features
    if gt_features is not None and pred_features is not None and points is not None:
        plot_comparison_features_wandb(gt_features, pred_features, points, reuse_pca, split, global_step, debug=debug)

    # images
    mean_neg = [-m for m in img_norm_cfg['mean']]
    std_inv = [1 / s for s in img_norm_cfg['std']]
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=std_inv),
                                   transforms.Normalize(mean=mean_neg,
                                                        std=[1., 1., 1.]),
                                   ])
    images_t = invTrans(images)

    # TODO: downsample by interpolation
    images_t = torch.nn.functional.interpolate(images_t, scale_factor=(1 / 4), mode='bilinear')

    images_t = images_t.permute(0, 2, 3, 1).detach().cpu().numpy()[..., ::-1]
    images_2rows_color = np.concatenate((np.concatenate(images_t[[2, 0, 1]], axis=1),
                                         np.concatenate(images_t[[4, 3, 5]], axis=1)), axis=0)
    images_2rows_color = images_2rows_color.clip(0, 255).astype(np.uint8)
    images_t = wandb.Image(
        images_2rows_color,
        caption="6 input images"
    )
    name = f'point_cloud/{split}/inputs'
    wandb.log({name: images_t}, commit=commit)


def log_comparison_clip_wandb(gt_grid, pred_grid, free_label, ignore_label, split, global_step, debug=False):
    if debug: print('In WandB logging!')

    plot_comparison_occupancy_semantics_wandb(gt_grid, free_label, ignore_label, split, global_step, mode='gt_clip')
    plot_comparison_occupancy_semantics_wandb(pred_grid, free_label, ignore_label, split, global_step, mode='pred_clip')
