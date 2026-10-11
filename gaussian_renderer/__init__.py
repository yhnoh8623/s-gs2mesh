#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
import diff_gaussian_rasterization as rast3d
import diff_surfel_rasterization as rast2d

from utils.point_utils import depth_to_normal


def _raster_settings(module, viewpoint_camera, pc, bg_color, scaling_modifier, debug):
    return module.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
    )


def _screenspace_points(pc):
    # zero tensor whose gradient gives the screen-space gradients of the 2D means (densification)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    return screenspace_points


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """3DGS rendering: color, accumulated (alpha-weighted) depth and alpha. bg_color must be on the GPU."""
    screenspace_points = _screenspace_points(pc)
    rasterizer = rast3d.GaussianRasterizer(
        raster_settings=_raster_settings(rast3d, viewpoint_camera, pc, bg_color, scaling_modifier, pipe.debug))

    rendered_image, radii, depth, alpha = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha}


def render_2dgs(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """2DGS rendering with the surface depth / normal maps used for regularization. bg_color must be on the GPU."""
    screenspace_points = _screenspace_points(pc)
    rasterizer = rast2d.GaussianRasterizer(
        raster_settings=_raster_settings(rast2d, viewpoint_camera, pc, bg_color, scaling_modifier, False))

    rendered_image, radii, allmap = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None)

    render_alpha = allmap[1:2]

    # normal map in world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)

    # median and expected depth
    render_depth_median = torch.nan_to_num(allmap[5:6], 0, 0)
    render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)

    # depth distortion map
    render_dist = allmap[6:7]

    # surface depth: median (depth_ratio = 1, bounded scenes) or expected (depth_ratio = 0)
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
    # pseudo surface normal from the depth map (render_normal is not normalized, hence the alpha factor)
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1)
    surf_normal = surf_normal * (render_alpha).detach()

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal}
