from sklearn.decomposition import PCA

import argparse, torch, os
import shutil
import numpy as np
import mmcv
try:
    from mmcv import Config
except:
    from mmengine.config import Config
from collections import OrderedDict

try:
    from pyvirtualdisplay import Display

    display = Display(visible=False, size=(2560, 1440))
    display.start()

    # from mayavi import mlab
    #
    # mlab.options.offscreen = True
    # print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
except:
    pass

STANDARDIZE_PRE_PCA = True
# STANDARDIZE_PRE_PCA = False
COLORS = np.array(
    [
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
        # [175,   0,  75, 255],       # other_flat           dark red
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


def pca_rgb_projection(features, standardize=STANDARDIZE_PRE_PCA,
                       mean=None, std=None, pca=None, reuse_pca=None):
    import cv2
    # features are of shape [N,dim]
    # X = features.transpose(1, 2, 0)
    print(f'features.shape: {features.shape}')
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

    print(f'Xt.shape: {Xt.shape}')

    # Normalize
    if standardize:
        if mean is None:
            mean = np.mean(Xt, axis=-1)[:, np.newaxis]
        if std is None:
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
    if pca is None:
        # Apply PCA
        pca = PCA(n_components=nb_comp)
        pca.fit(Xtn)
    else:
        print('Using precomputed PCA.')
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


def show3d(points, fig=None, s=0.01, savepath=None, colors=None, labels=None, ft_color=None, ft_loc=None, show_colorbar=False):
    from matplotlib import pyplot as plt
    d1, d2 = points.shape
    if d1 > d2:
        points = points.T
    x, y, z = points[:3]

    if fig is None:
        fig = plt.figure()
    if ft_color is None:
        # only labels
        ax = fig.add_subplot(projection='3d')
        sc = show3d_labels(ax, colors, labels, s, x, y, z)
        if show_colorbar:
            fig.colorbar(sc)
        set_axes_equal(ax)
    else:
        ft_color = ft_color / 255
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        show3d_labels(ax, colors, labels, s, x, y, z)
        set_axes_equal(ax)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        if ft_loc is not None:
            x, y, z = ft_loc.cpu().T
        sc = ax.scatter(x, y, z, s=s, c=ft_color)
        if show_colorbar:
            fig.colorbar(sc)
        set_axes_equal(ax)

    plt.tight_layout()

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f'Saved to {savepath}')


def show3d_labels(ax, colors, labels, s, x, y, z):
    if labels is None:
        sc = ax.scatter(x, y, z, s=s)
    else:
        if colors is not None:
            for _unq in np.unique(labels):
                whr = labels == _unq
                print(_unq)
                color = colors[_unq] / 255.
                # print(color)
                sc = ax.scatter(x[whr], y[whr], z[whr], color=color, s=s)
        else:
            sc = ax.scatter(x, y, z, c=labels, s=s)
    return sc


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0])  # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1])  # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2])  # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw(
        voxels,  # semantic occupancy predictions
        pred_pts,  # lidarseg predictions
        vox_origin,
        voxel_size=0.2,  # voxel size in the real world
        grid=None,  # voxel coordinates of point cloud
        pt_label=None,  # label of point cloud
        save_dirs=None,
        cam_positions=None,
        focal_positions=None,
        timestamp=None,
        mode=0,
        voxel_gt=None,
        agnostic=False,
        gt_features=None,
        grid_features=None,
        pred_features=None,
        reuse_pca=False
):
    w, h, z = voxels.shape
    grid = grid.astype(np.int)

    if agnostic:
        occupied_gt = np.array(np.where(voxel_gt)).T
        gt_labels = np.ones(len(occupied_gt))
        _whr = np.ones(len(gt_labels), dtype=bool)
        colors = np.array([(0, 0, 0, 255), (0, 0, 0, 255)])
    else:
        _whr = np.array(np.where((voxel_gt > 0) & (voxel_gt < 17)))
        occupied_gt = _whr
        gt_labels = voxel_gt[_whr]
        colors = COLORS

    # print(f'colors: {colors}')

    print(f'gt_features is None: {gt_features is None}')
    if gt_features is None:
        gt_ft_color = mean = std = pca = ft_loc = None
        print(f'occupied_gt.shape: {occupied_gt.shape}')
        occupied_gt = occupied_gt.T
        oc0, oc1, oc2 = occupied_gt.T

        print(f'voxel_gt: {voxel_gt.shape}')
        gt_labels = voxel_gt[oc0, oc1, oc2]
        print(f'gt_labels: {gt_labels.shape}')

    else:
        unq_grid_fts, inverse_indices = torch.unique(grid_features[0], dim=0, return_inverse=True)
        _whr = torch.unique(inverse_indices)
        occupied_gt = unq_grid_fts.cpu()
        oc0, oc1, oc2 = occupied_gt.T.cpu()

        print(f'voxel_gt: {voxel_gt.shape}')
        gt_labels = voxel_gt[oc0, oc1, oc2]
        print(f'gt_labels: {gt_labels.shape}')

        ft_loc = grid_features[0]
        gt_features = gt_features
        gt_ft_color, (mean, std, pca) = pca_rgb_projection(gt_features, reuse_pca=reuse_pca)

    show3d(occupied_gt, s=0.1, savepath=os.path.join(save_dirs, 'gt.png'), colors=colors, labels=gt_labels,
           ft_color=gt_ft_color, ft_loc=ft_loc)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    print(f'grid_coords: {grid_coords.shape}')

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
        # draw a simple car at the middle
        # car_vox_range = np.array([
        #     [w//2 - 2 - 4, w//2 - 2 + 4],
        #     [h//2 - 2 - 4, h//2 - 2 + 4],
        #     [z//2 - 2 - 3, z//2 - 2 + 3]
        # ], dtype=np.int)
        # car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
        # car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
        # car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
        # car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
        # car_label = np.zeros([8, 8, 6], dtype=np.int)
        # car_label[:3, :, :2] = 17
        # car_label[3:6, :, :2] = 18
        # car_label[6:, :, :2] = 19
        # car_label[:3, :, 2:4] = 18
        # car_label[3:6, :, 2:4] = 19
        # car_label[6:, :, 2:4] = 17
        # car_label[:3, :, 4:] = 19
        # car_label[3:6, :, 4:] = 17
        # car_label[6:, :, 4:] = 18
        # car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
        # car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
        # grid_coords[car_indexes, 3] = car_label.flatten()

    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    if agnostic:
        grid_coords[grid_coords[:, 3] == 0, 3] = 20
    else:
        grid_coords[grid_coords[:, 3] == 17, 3] = 20

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
        ]
    print('num voxels:', len(fov_voxels))
    print('shape fov_voxels:', fov_voxels.shape)

    if agnostic:
        whr = np.array(fov_voxels[:, 3] > 0)
        pred_labels = np.ones(len(whr)).astype(np.uint8)
    else:
        _whr = (fov_voxels[:, 3] > 0) & (fov_voxels[:, 3] < 20)
        whr = np.array(_whr)
        pred_labels = fov_voxels[_whr, 3].astype(np.uint8)
    print(f'whr shape {whr.shape}')

    if pred_features is None:
        pred_ft_color = None
        ft_loc = None
    else:
        ft_loc = grid_features[0]
        pred_features = pred_features[0]
        print(f'pred_features.shape: {pred_features.shape}, ft_loc.shape: {ft_loc.shape}')
        pred_ft_color, *_ = pca_rgb_projection(pred_features, mean=mean, std=std, pca=pca)
        print(f'pred_features.shape: {pred_features.shape}')
        print(f'pred_ft_color.shape: {pred_ft_color.shape}')

    xyz = fov_voxels[whr, :3]
    print(f'xyz shape: {xyz.shape}')
    savepath = os.path.join(frame_dir, 'tmp2.png')
    show3d(xyz[:, [1, 0, 2]], savepath=savepath, s=0.1, colors=colors, labels=pred_labels, ft_color=pred_ft_color,
           ft_loc=ft_loc)
    print(f'Saved to {savepath}')

    if pred_features is not None and gt_features is not None:
        l2_diff = torch.sqrt(((pred_features.cpu() - gt_features) ** 2).sum(1))
        savepath = os.path.join(frame_dir, 'diff.png')
        show3d(ft_loc.cpu(), savepath=savepath, s=0.1, labels=l2_diff, show_colorbar=True)

    if agnostic:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = 1, 19
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=vmin,  # 1,
        vmax=vmax,  # 19,  # 16
    )
    print('After draw')

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = COLORS

    scene = figure.scene
    scene.camera.position = [0.75131739, -35.08337438, 16.71378558]
    scene.camera.focal_point = [0.75131739, -34.21734897, 16.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    print('After rendering')

    mlab.savefig(os.path.join(frame_dir, 'tmp.png'))
    print('After save.')

    # scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
    # scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
    # scene.camera.view_angle = 40.0
    # scene.camera.view_up = [0., 1., 0.]
    # scene.camera.clipping_range = [0.01, 400.]
    # scene.camera.compute_view_plane_normal()
    # scene.render()
    mlab.show()
    print('After show.')


if __name__ == "__main__":

    features = np.load('/home/vobecant/PhD/TPVFormer-OpenSet/sample_1xScale/0046092508b14f40a86760d11f9896bb/ft.npy')
    xyz = np.load('/home/vobecant/PhD/TPVFormer-OpenSet/sample_1xScale/0046092508b14f40a86760d11f9896bb/xyz.npy')

    rgb = pca_rgb_projection(features)[0]
    with open('/home/vobecant/PhD/TPVFormer-OpenSet/sample_1xScale/0046092508b14f40a86760d11f9896bb/xyz_rgb.txt',
              'w') as f:
        for _xyz, _rgb in zip(xyz.T, rgb):
            x, y, z = _xyz
            r, g, b = _rgb
            f.write(f'{x} {y} {z} {r} {g} {b}\n')

    import sys;

    sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    ## prepare config
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--work-dir', type=str, default='out/tpv_occupancy')
    parser.add_argument('--ckpt-path', type=str, default='out/tpv_occupancy/latest.pth')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--save-path', type=str, default='out/tpv_occupancy/frames')
    parser.add_argument('--frame-idx', type=int, default=0, nargs='+',
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    parser.add_argument('--mode', type=int, default=0, help='0: occupancy, 1: predicted point cloud, 2: gt point cloud')
    parser.add_argument('--agnostic', action='store_true')
    parser.add_argument('--reuse-pca', action='store_true')

    args = parser.parse_args()
    print(args)

    cfg = Config.fromfile(args.py_config)
    dataset_config = cfg.dataset_params

    # prepare model
    logger = mmcv.utils.get_logger('mmcv')
    logger.setLevel("WARNING")
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model).to(device)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(revise_ckpt(ckpt)))
    my_model.eval()

    # prepare data
    from nuscenes import NuScenes
    from visualization.dataset import ImagePoint_NuScenes_vis, DatasetWrapper_NuScenes_vis, \
        ImagePoint_NuScenes_vis_wFeats

    if args.vis_train:
        pkl_path = cfg.train_data_loader['imageset']
        # pkl_path = 'data/nuscenes_infos_train.pkl'
    else:
        pkl_path = cfg.val_data_loader['imageset']
        # pkl_path = 'data/nuscenes_infos_val.pkl'

    try:
        feature_learning = cfg.feature_learning
    except:
        feature_learning = False

    data_path = 'data/nuscenes'
    label_mapping = dataset_config['label_mapping']

    version = dataset_config['version']
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    if not feature_learning:
        pt_dataset = ImagePoint_NuScenes_vis(
            data_path, imageset=pkl_path,
            label_mapping=label_mapping, nusc=nusc)
    else:
        features_type = dataset_config.get("features_type", None)
        features_path = dataset_config.get("features_path", None)
        projections_path = dataset_config.get("projections_path", None)
        pt_dataset = ImagePoint_NuScenes_vis_wFeats(
            data_path, imageset=pkl_path,
            label_mapping=label_mapping, nusc=nusc,
            features_type=features_type, features_path=features_path, projections_path=projections_path)

    print(f"type(pt_dataset): {type(pt_dataset)}")

    dataset = DatasetWrapper_NuScenes_vis(
        pt_dataset,
        grid_size=cfg.grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["fill_label"],
        phase='val'
    )
    print(len(dataset))

    for index in args.frame_idx:
        print(f'processing frame {index}')
        batch_data, filelist, scene_meta, timestamp = dataset[index]
        if not feature_learning:
            imgs, img_metas, vox_label, grid, pt_label = batch_data
            pt_fts = grid_fts = None
        else:
            imgs, img_metas, vox_label, grid, pt_label, grid_fts, pt_fts = batch_data
            grid_fts = torch.from_numpy(grid_fts).unsqueeze(0).to(device)
        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)
        with torch.no_grad():
            if not feature_learning:
                outputs_vox, outputs_pts, *_ = my_model(img=imgs, img_metas=[img_metas], points=grid.clone())
                predict_fts_pts = None
            else:
                outputs_vox, outputs_pts, predict_fts_pts = my_model(img=imgs, img_metas=[img_metas],
                                                                     points=grid.clone(), features=grid_fts)

            predict_vox = torch.argmax(outputs_vox, dim=1)  # bs, w, h, z
            predict_vox = predict_vox.squeeze(0).cpu().numpy()  # w, h, z

            predict_pts = torch.argmax(outputs_pts, dim=1)  # bs, n, 1, 1
            predict_pts = predict_pts.squeeze().cpu().numpy()  # n

        voxel_origin = dataset_config['min_volume_space']
        voxel_max = dataset_config['max_volume_space']
        grid_size = cfg.grid_size
        resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

        frame_dir = os.path.join(args.save_path, str(index))
        os.makedirs(frame_dir, exist_ok=True)

        for filename in filelist:
            shutil.copy(filename, os.path.join(frame_dir, os.path.basename(filename)))

        draw(predict_vox,
             predict_pts,
             voxel_origin,
             resolution,
             grid.squeeze(0).cpu().numpy(),
             pt_label.squeeze(-1),
             frame_dir,
             img_metas['cam_positions'],
             img_metas['focal_positions'],
             timestamp=timestamp,
             mode=args.mode,
             voxel_gt=vox_label,
             agnostic=args.agnostic,
             gt_features=pt_fts,
             grid_features=grid_fts,
             pred_features=predict_fts_pts,
             reuse_pca=args.reuse_pca)
