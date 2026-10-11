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
import sys
import glob
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import getWorld2View2, focal2fov
from utils.general_utils import PILtoTorch
from utils.flow_utils import readFlow
from utils.feat_utils import FeatExt

# DTU images are (cropped) 1600x1200: features are extracted on a zero-padded 1600x1200 canvas
DTU_W, DTU_H = 1600, 1200


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    flow: list
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    feat: list


class SceneInfo(NamedTuple):
    train_cameras: list
    nerf_normalization: dict


def getNerfppNorm(cam_info):
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    cam_centers = np.hstack(cam_centers)
    center = np.mean(cam_centers, axis=1, keepdims=True)
    diagonal = np.max(np.linalg.norm(cam_centers - center, axis=0, keepdims=True))
    return {"translate": -center.flatten(), "radius": diagonal * 1.1}


@torch.no_grad()
def extract_features(image_paths):
    """Vis-MVSNet features of every image at full (scale 1) and half (scale 2) resolution."""
    feat_ext = FeatExt().cuda()
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406]).float()
    std = torch.tensor([0.229, 0.224, 0.225]).float()
    feats_list = []
    for scale in [1, 2]:
        rgb = torch.zeros(len(image_paths), 3, DTU_H // scale, DTU_W // scale)
        for i, path in enumerate(image_paths):
            image_pil = Image.open(path)
            resolution = (int(image_pil.size[0] / scale), int(image_pil.size[1] / scale))
            image = torch.cat([PILtoTorch(im, resolution) for im in image_pil.split()[:3]], dim=0)
            rgb[i, :, :image.shape[1], :image.shape[2]] = image
        rgb = (rgb / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        feats = torch.cat([feat_ext(rgb[s:s + 20].cuda())[2].detach().cpu() for s in range(0, len(image_paths), 20)], dim=0)
        feats_list.append(feats[..., :(DTU_H // 2) // scale, :(DTU_W // 2) // scale])
    return feats_list


def readColmapCameras(cam_extrinsics, cam_intrinsics, path, image_paths, n_views):
    mask_paths = sorted(glob.glob(os.path.join(path, "mask", "*.png")))
    feats_list = extract_features(image_paths)

    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        K = np.zeros((3, 3)).astype(np.float32)
        K[2, 2] = 1.
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            K[0, 0] = intr.params[0]
            K[1, 1] = intr.params[0]
            K[0, 2] = intr.params[1]
            K[1, 2] = intr.params[2]
            FovY = focal2fov(intr.params[0], height)
            FovX = focal2fov(intr.params[0], width)
        elif intr.model == "PINHOLE":
            K[0, 0] = intr.params[0]
            K[1, 1] = intr.params[1]
            K[0, 2] = intr.params[2]
            K[1, 2] = intr.params[3]
            FovY = focal2fov(intr.params[1], height)
            FovX = focal2fov(intr.params[0], width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_name = os.path.basename(extr.name).split(".")[0]
        rgb_path = image_paths[idx]
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        image = Image.open(rgb_path)

        # optical flows to the other input views: flow/<name>_<i>_<j>.flo
        flow = []
        for i in range(n_views):
            for j in range(n_views):
                if i == j:
                    continue
                flow_path = os.path.join(path, 'flow', rgb_name + f'_{i}_{j}.flo')
                if os.path.exists(flow_path):
                    flow.append((j, readFlow(flow_path)))

        # object mask of the i-th input view (sorted mask/*.png)
        mask = None
        if int(image_name) < len(mask_paths):
            mask = np.array(Image.open(mask_paths[int(image_name)]).convert("L"))

        cam_infos.append(CameraInfo(uid=intr.id, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, flow=flow,
                                    image_path=rgb_path, image_name=image_name, width=width, height=height,
                                    mask=mask, feat=[feats[idx] for feats in feats_list]))
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path, images, n_views):
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0", "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0", "cameras.txt"))

    image_paths = [f for f in sorted(glob.glob(os.path.join(path, images or "images", '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, path, image_paths, n_views)
    train_cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    return SceneInfo(train_cameras=train_cam_infos, nerf_normalization=getNerfppNorm(train_cam_infos))
