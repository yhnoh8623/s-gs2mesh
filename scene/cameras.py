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

import numpy as np
import torch
from torch import nn

from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, K, FoVx, FoVy, image, image_name, uid, feats, flow, mask=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K.transpose()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.flow = flow
        self.alpha_mask = None if mask is None else torch.tensor(mask > 127 if mask.dtype == np.uint8 else mask > 0.5).bool()

        # Vis-MVSNet features of the full (stride 2) and the half-resolution image (stride 4)
        self.feat = [
            feats[0][:, :math.ceil(image.shape[1]), :math.ceil(image.shape[2])].to('cuda'),
            feats[1][:, :math.ceil(image.shape[1] / 2), :math.ceil(image.shape[2] / 2)].to('cuda')
        ]

        self.data_device = torch.device(data_device)
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        fx = self.image_width / (2 * math.tan(self.FoVx / 2.))
        fy = self.image_height / (2 * math.tan(self.FoVy / 2.))
        self.intrinsic = torch.tensor(
            [[fx, 0., self.image_width / 2., 0.],
             [0., fy, self.image_height / 2., 0.],
             [0., 0., 1.0, 0.],
             [0., 0., 0., 1.0]]
        ).float().cuda()

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # set by Scene
        self.radius = None
        self.stereo_cam_list = None

    def set_flow_depth(self, flow_depth, flow_depth_mask):
        self.flow_depth = flow_depth.clone().squeeze().cpu()
        self.flow_depth_mask = flow_depth_mask.clone().squeeze().cpu()


class StereoCamera:
    """
    Right (stereo) view of a train camera for a baseline given in % of the scene radius (Eq. 1):
    R_R = R_L, C_R = C_L + R_L [b, 0, 0], i.e. T_R = T_L - [b, 0, 0] in world-to-camera form.
    """

    def __init__(self, left_cam, baseline_percentage, radius):
        self.baseline = baseline_percentage
        self.radius = radius
        self.baseline_abs = baseline_percentage * radius / 100.
        self.image_name = left_cam.image_name
        self.uid = left_cam.uid
        self.R = left_cam.R
        self.T = left_cam.T - np.array([self.baseline_abs, 0., 0.])
        self.FoVx = left_cam.FoVx
        self.FoVy = left_cam.FoVy
        self.image_width = left_cam.image_width
        self.image_height = left_cam.image_height
        self.intrinsic = left_cam.intrinsic
        self.znear = left_cam.znear
        self.zfar = left_cam.zfar
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, left_cam.trans, left_cam.scale)).transpose(0, 1).cuda()
        self.projection_matrix = left_cam.projection_matrix
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
