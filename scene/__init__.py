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

import os
import random

import numpy as np

from scene.cameras import StereoCamera
from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
from scene.surfel_model import SurfelModel
from utils.camera_utils import cameraList_from_camInfos
from utils.flow_utils import compute_depth_by_flow, construct_pcd

__all__ = ["Scene", "GaussianModel", "SurfelModel"]


class Scene:
    """
    Sparse input views with their epipolar depth priors. A freshly created 3DGS (GaussianModel) or
    2DGS (SurfelModel) is initialized from the epipolar-depth point cloud.
    """

    def __init__(self, args, gaussians, shuffle=True):
        self.model_path = args.model_path
        self.gaussians = gaussians

        assert os.path.exists(os.path.join(args.source_path, "sparse")), "Could not recognize scene type!"
        scene_info = readColmapSceneInfo(args.source_path, args.images, args.n_views)
        if shuffle:
            random.shuffle(scene_info.train_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)

        compute_depth_by_flow(self.train_cameras, args.valid_dis_threshold, args.near_n)
        self.pcd = construct_pcd(self.train_cameras)
        if isinstance(self.gaussians, GaussianModel):
            self.gaussians.create_from_pcd(self.pcd, self.cameras_extent, args.drop_rate)
        else:
            self.gaussians.create_from_pcd(self.pcd, self.cameras_extent)

        self.radius = self.compute_stereo_radius(args, self.train_cameras)
        for cam in self.train_cameras:
            cam.radius = self.radius
            cam.stereo_cam_list = [StereoCamera(cam, b, self.radius) for b in args.renderer_baseline_candidate]

    @staticmethod
    def compute_stereo_radius(args, cameras):
        """Scene radius used for the stereo baseline (GS2Mesh definition, doubled for DTU)."""
        ts = np.stack([cam.camera_center.detach().cpu().numpy() for cam in cameras])
        radius = np.median(np.linalg.norm(ts - ts.mean(axis=0), axis=1))
        if args.dataset_type.lower() == "dtu":
            radius *= 2
        return float(radius)

    def save(self, iteration):
        self.gaussians.save_ply(os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras
