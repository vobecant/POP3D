import numpy as np
from sklearn.neighbors import NearestNeighbors


class GridRemapping:
    def __init__(self, mode, orig_extent_xyz, target_extent_xyz, orig_num_steps_xyz, target_num_steps_xyz):
        mode = mode.lower()
        assert mode in ['nearest', 'linear']
        self.mode = mode
        self.target_grid_size = target_num_steps_xyz

        # !!! NOTE: The grid is first shifted such that it starts at [0,0,0]
        grid_orig_x = np.linspace(orig_extent_xyz[0], orig_extent_xyz[3], orig_num_steps_xyz[0])
        grid_orig_y = np.linspace(orig_extent_xyz[1], orig_extent_xyz[4], orig_num_steps_xyz[1])
        grid_orig_z = np.linspace(orig_extent_xyz[2], orig_extent_xyz[5], orig_num_steps_xyz[2])

        grid_orig_xyz = np.meshgrid(grid_orig_x, grid_orig_y, grid_orig_z)
        grid_orig_xyz = np.stack((grid_orig_xyz[0], grid_orig_xyz[1], grid_orig_xyz[2]))
        grid_orig_xyz_shape = grid_orig_xyz.shape
        grid_orig_xyz = grid_orig_xyz.reshape(
            (3, orig_num_steps_xyz[0] * orig_num_steps_xyz[1] * orig_num_steps_xyz[2])).T

        x, y, z = np.arange(orig_num_steps_xyz[0]), np.arange(orig_num_steps_xyz[1]), np.arange(orig_num_steps_xyz[2])
        xyz_idx = np.meshgrid(x, y, z)
        xyz_idx = np.stack((xyz_idx[0], xyz_idx[1], xyz_idx[2]))
        self.orig_flatten2xyz = xyz_idx.reshape(
            (3, orig_num_steps_xyz[0] * orig_num_steps_xyz[1] * orig_num_steps_xyz[2]))

        # TODO: We first need to shift the target grid such that it is in the same origin as the original grid.
        grid_tgt_x = np.linspace(target_extent_xyz[0], target_extent_xyz[3], target_num_steps_xyz[0])
        grid_tgt_y = np.linspace(target_extent_xyz[1], target_extent_xyz[4], target_num_steps_xyz[1])
        grid_tgt_z = np.linspace(target_extent_xyz[2], target_extent_xyz[5], target_num_steps_xyz[2])

        grid_tgt_xyz = np.meshgrid(grid_tgt_x, grid_tgt_y, grid_tgt_z)
        grid_tgt_xyz = np.stack((grid_tgt_xyz[0], grid_tgt_xyz[1], grid_tgt_xyz[2]))
        grid_tgt_xyz_shape = grid_tgt_xyz.shape
        grid_tgt_xyz = grid_tgt_xyz.reshape(
            (3, target_num_steps_xyz[0] * target_num_steps_xyz[1] * target_num_steps_xyz[2])).T

        # nearest neighbor search
        knn = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(grid_orig_xyz)
        _, self.orig2tgt_4nearest = knn.kneighbors(grid_tgt_xyz)
        self.orig2tgt_nearest = self.orig2tgt_4nearest[:, 0]

        print('Grid Remapper initialization finished.')

    def __call__(self, orig_grid):
        orig_shape = orig_grid.shape
        bs, nfts = orig_shape[:2]
        num_spatial = np.prod(orig_shape[-3:])
        orig_grid_flatten = orig_grid.reshape((orig_shape[0], orig_shape[1], num_spatial))
        if self.mode == 'nearest':
            resampled = orig_grid_flatten[..., self.orig2tgt_nearest]
        else:
            # linear
            resampled = orig_grid_flatten[..., self.orig2tgt_nearest[:, 0]] + orig_grid_flatten[
                ..., self.orig2tgt_nearest[:, 1]] + orig_grid_flatten[..., self.orig2tgt_nearest[:, 2]] + \
                        orig_grid_flatten[..., self.orig2tgt_nearest[:, 3]]
            resampled /= 4

        target_grid_size = [bs, nfts] + self.target_grid_size
        resampled_xyz = np.zeros(target_grid_size)
        resampled_xyz[self.orig_flatten2xyz[0], self.orig_flatten2xyz[1], self.orig_flatten2xyz[2]] = resampled


if __name__ == '__main__':
    orig_extent_xyz = [-51.2, -51.2, -5, 51.2, 51.2, 3]
    orig_num_steps_xyz = [100, 100, 8]
    tgt_extent_xyz = [-40, -40, -1, 40, 40, 5.4]
    target_num_steps_xyz = [200, 200, 16]

    bs = 1
    ft_size = 512
    orig_feature_size = [bs, ft_size] + orig_num_steps_xyz

    grid_remapping = GridRemapping('nearest', orig_extent_xyz, tgt_extent_xyz, orig_num_steps_xyz, target_num_steps_xyz)
    orig_fts_grid = np.random.random(size=orig_feature_size)
    grid_remapping(orig_fts_grid)
