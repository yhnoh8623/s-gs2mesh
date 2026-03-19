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

from scene.cameras import Camera
import numpy as np
import cv2
from tqdm import tqdm
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, pcd=None):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    mask = None if cam_info.mask is None else cv2.resize(cam_info.mask, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None


    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    K = cam_info.K
    W, H = K[0, 2] * 2, K[1, 2] * 2
    K[0, 2], K[1, 2] = resolution[0] / 2., resolution[1] / 2.
    K[0, 0] = K[0, 0] * resolution[0] / W
    K[1, 1] = K[1, 1] * resolution[1] / H
    R = cam_info.R
    T = cam_info.T

    width, height = resolution

    
    if cam_info.flow is []:
        flow = None
    else:
        flow = []
        for i, f in cam_info.flow:
            if f.shape[0] != resolution[1] or f.shape[1] != resolution[0]:
                scale_x = (f.shape[1] / resolution[0])
                scale_y = (f.shape[0] / resolution[1])
                f = cv2.resize(f, resolution)
                f[:, :, 0] = f[:, :, 0] / scale_x
                f[:, :, 1] = f[:, :, 1] / scale_y
            flow.append((i, f))
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, K=K,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,  image=gt_image, gt_alpha_mask=loaded_mask,
                  uid=id, data_device=args.data_device, image_name=cam_info.image_name,
                  mask=mask, bounds=cam_info.bounds,
                  flow=flow)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, pcd=None):
    camera_list = []

    for id, c in tqdm((enumerate(cam_infos))):
        camera_list.append(loadCam(args, id, c, resolution_scale, pcd))

    return camera_list

def cameraList_from_huggingfaceModel(args, model, prefix="train"):
    camera_list = []

    for idx, cam_info in tqdm((enumerate(getattr(model, prefix + "_cameras_config")))):
        
        R = np.asarray(cam_info["R"])
        T = np.asarray(cam_info["T"])
        K = np.asarray(cam_info["K"])
        trans = np.asarray(cam_info["trans"])
        bounds = np.asarray(cam_info["bounds"])

        image = getattr(model, prefix + "_cam{}_original_image".format(str(idx).zfill(3))).data
        alpha_mask = getattr(model, prefix + "_cam{}_alpha_mask".format(str(idx).zfill(3))).data

        cam = Camera(colmap_id=cam_info["uid"], R=R, T=T, K=K,
                  FoVx=cam_info["FoVx"], FoVy=cam_info["FoVy"],  image=image, gt_alpha_mask=alpha_mask,
                  uid=cam_info["uid"], data_device=args.data_device, image_name=cam_info["image_name"],
                  mask=alpha_mask, bounds=bounds, scale=cam_info["scale"], trans=trans)
        
        depth = getattr(model, prefix + "_cam{}_depth".format(str(idx).zfill(3))).data
        depth_mask = getattr(model, prefix + "_cam{}_depth_mask".format(str(idx).zfill(3))).data
        cam.set_flow_depth(depth, depth_mask)
        camera_list.append(cam)

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
