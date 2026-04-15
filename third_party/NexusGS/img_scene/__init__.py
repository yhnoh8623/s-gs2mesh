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
import json
import numpy as np
# import torch
from utils.system_utils import searchForMaxIteration
from img_scene.dataset_readers import sceneLoadTypeCallbacks
from img_scene.gaussian_model import GaussianModel
from img_scene.cameras import Camera
#from scene.hf_gaussian_model import HFGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_huggingfaceModel
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360
# from gaussian_renderer import render
from utils.flow_utils import compute_depth_by_flow, construct_pcd
from utils.graphics_utils import BasicPointCloud
from utils.colmap_utils import poses_from_file
from utils.transformation_utils import rotm2eul, eul2rotm, intrinsic_from_camera_params, RT_from_rot_pos, convert_R_T_to_GS, convert_GS_to_R_T, calculate_right_camera_pose
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        self.eval_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, dataset_type=args.dataset_type)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.point_cloud)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Eval Cameras")
            self.eval_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.eval_cameras, resolution_scale, args)


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            for resolution_scale in resolution_scales:
                compute_depth_by_flow(self.train_cameras[resolution_scale], args.valid_dis_threshold, args.near_n)
            self.pcd = construct_pcd(self.train_cameras[1.0])
            self.gaussians.create_from_pcd(self.pcd, None, self.cameras_extent, args.drop_rate)
        
        print(f'renderer baseline : {args.renderer_baseline_candidate}')
        for resolution_scale in resolution_scales:
            self.stereo_cams = self.generateStereoCams(args, self.train_cameras[resolution_scale], args.renderer_baseline_candidate)
    
    def generateStereoCams(self, args, cameras, baseline_candidates):
        stereo_cams = []
        self.poses = poses_from_file(os.path.join(args.source_path, "sparse/0/images.txt"))
        
        poses_inv = [np.linalg.inv(np.vstack((pose, np.array([0, 0, 0, 1])))) for pose in self.poses]
        camera_rotations = [rotm2eul(pose[:3,:3]) for pose in poses_inv]
        for i in range(len(camera_rotations)):
            rotation = eul2rotm(camera_rotations[i])
            rotation[:,1:]*=-1
            camera_rotations[i] = rotm2eul(rotation)

        camera_locations = [pose[:3,3].tolist() for pose in poses_inv]
        radius = None
        ts = np.array(camera_locations)
        radius = np.median(np.linalg.norm(ts - ts.mean(axis=0),axis=1))
        if args.dataset_type == "DTU" or args.dataset_type == "dtu": # For backwards compatibility with paper results
            radius *= 2
            
        for cam in cameras:
            stereo_cam_list = []
            #change_T = [cam.T[0], -1*cam.T[1], -1*cam.T[2]]
            for baseline in baseline_candidates:
                
               # print(f"before : {cam.T}")
                left_R, left_T = convert_GS_to_R_T(cam.R, cam.T)
                right_rot, right_pos = calculate_right_camera_pose(left_R, left_T, baseline*radius/100)
                right_R, right_T = convert_R_T_to_GS(right_rot, right_pos)
                stereo_cam = Camera(0, right_R, right_T, cam.K, cam.FoVx, cam.FoVy, cam.original_image, cam.alpha_mask, cam.image_name, cam.uid, baseline=baseline, original_cam=cam)
                #print(f"after : {right_T}")
                stereo_cams.append(stereo_cam)
                stereo_cam_list.append(stereo_cam)
            cam.stereo_cam_list = stereo_cam_list
                
        return stereo_cams
    
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getEvalCameras(self, scale=1.0):
        return self.eval_cameras[scale]
