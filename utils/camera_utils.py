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

import cv2
from tqdm import tqdm

from scene.cameras import Camera
from utils.general_utils import PILtoTorch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution))
    else:
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

    gt_image = PILtoTorch(cam_info.image, resolution)[:3, ...]

    mask = cam_info.mask
    if mask is not None:
        if mask.shape[0] > orig_h and mask.shape[1] > orig_w:
            # DTU masks are 1600x1200 while the images are cropped around the principal point
            # (1554x1162 = [38:, 46:]), so crop instead of resizing
            mask = mask[mask.shape[0] - orig_h:, mask.shape[1] - orig_w:]
        mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_NEAREST)

    # rescale the intrinsics to the loaded resolution (principal point at the image center)
    K = cam_info.K
    W, H = K[0, 2] * 2, K[1, 2] * 2
    K[0, 2], K[1, 2] = resolution[0] / 2., resolution[1] / 2.
    K[0, 0] = K[0, 0] * resolution[0] / W
    K[1, 1] = K[1, 1] * resolution[1] / H

    flow = []
    for i, f in cam_info.flow:
        if f.shape[0] != resolution[1] or f.shape[1] != resolution[0]:
            scale_x = (f.shape[1] / resolution[0])
            scale_y = (f.shape[0] / resolution[1])
            f = cv2.resize(f, resolution)
            f[:, :, 0] = f[:, :, 0] / scale_x
            f[:, :, 1] = f[:, :, 1] / scale_y
        flow.append((i, f))

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, K=K, FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, image_name=cam_info.image_name, uid=id, feats=cam_info.feat, flow=flow,
                  mask=mask, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    return [loadCam(args, id, c, resolution_scale) for id, c in tqdm(enumerate(cam_infos))]
