import numpy as np
import torch as th
import trimesh
import open3d as o3d
import torch


def get_volume_points(resolution=256):
    voxel_min = torch.tensor([-1.0, -1.0, -1.0]).cuda()  # Minimum coordinates of the voxel space
    voxel_max = torch.tensor([1.0, 1.0, 1.0]).cuda()  # Maximum coordinates of the voxel space
    voxel_size = 2.0 / resolution
    indices = torch.arange(resolution, dtype=torch.float32).cuda()
    x, y, z = torch.meshgrid(indices, indices, indices)
    voxel_centers_x = voxel_min[0] + (x + 0.5) * voxel_size
    voxel_centers_y = voxel_min[1] + (y + 0.5) * voxel_size
    voxel_centers_z = voxel_min[2] + (z + 0.5) * voxel_size
    voxel_centers = torch.stack((voxel_centers_x, voxel_centers_y, voxel_centers_z), dim=-1)
    voxel_centers = voxel_centers.reshape(-1, 3)
    return voxel_centers
