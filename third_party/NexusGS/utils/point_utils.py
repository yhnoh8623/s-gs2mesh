import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math
import open3d as o3d
import numpy as np


def depths_to_points(view, depthmap, resolution = 1): #resolution = 2 구현 필요
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    W /= resolution
    H /= resolution
    intrins = view.intrinsic[:3, :3].clone()
    for i in range(2):
        for j in range(3):
            intrins[i, j] /= resolution
    #print(intrins)
    #print(f"point_utils.py : depths to points")
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    
    # pts = points.detach().cpu().numpy()

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts)

    #o3d.io.write_point_cloud('depths_to_points.ply', pcd)

    return points

# def depths_to_points(view, depthmap):
#     c2w = (view.world_view_transform.T).inverse()
#     W, H = view.image_width, view.image_height
#     ndc2pix = torch.tensor([
#         [W / 2, 0, 0, (W) / 2],
#         [0, H / 2, 0, (H) / 2],
#         [0, 0, 0, 1]]).float().cuda().T
#     projection_matrix = c2w.T @ view.full_proj_transform
#     intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
#     grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
#     points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
#     rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
#     rays_o = c2w[:3,3]
#     points = depthmap.reshape(-1, 1) * rays_d + rays_o
#     return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output