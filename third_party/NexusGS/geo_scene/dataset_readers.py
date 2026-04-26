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
from PIL import Image
from typing import NamedTuple
from geo_scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import imageio
import glob
import skimage
from geo_scene.gaussian_model import BasicPointCloud
import cv2
import torch
from utils.flow_utils import readFlow


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
    bounds: np.array
    feat: list = None
    pair: list = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, n_views=3):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        one_mask = torch.ones((height, width), dtype=torch.float32).numpy()

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        K = np.zeros((3,3)).astype(np.float32)
        K[2, 2] = 1.
        try:
            bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]
        except:
            bounds = None

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            K[0,0] = intr.params[0]
            K[1,1] = intr.params[0]
            K[0,2] = intr.params[1]
            K[1,2] = intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            K[0,0] = intr.params[0]
            K[1,1] = intr.params[1]
            K[0,2] = intr.params[2]
            K[1,2] = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        rgb_name = os.path.basename(image_path).split(".")[0]
        #print(f'rgb_name: {rgb_name}')

        flow = []
        for i in range(n_views):
            for j in range(n_views):
                if i == j:
                    continue
                flow_path = os.path.join(path, 'flow', rgb_name + f'_{i}_{j}.flo')
                if os.path.exists(flow_path):
                    flow.append((j, readFlow(flow_path)))
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, mask = one_mask, flow = flow,
                              image_path=image_path, image_name=image_name, width=width, height=height, bounds = bounds)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = None
    #normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, n_views = 3, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), n_views = n_views, path = path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        total_cam = len(cam_infos)
        #jump = total_cam // 5
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for cam in train_cam_infos:
    #     print(cam.image_name)
    #     print('dd')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    #ply_path = os.path.join(path, "dense/fused.ply")
    
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs

def load_rgb(path):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img

def load_mask(path):
    alpha = Image.open(path)
    return alpha

def read_monoData(path):
    mono = np.load(path)
    if len(mono.shape) == 4:
        mono = mono[0]
    elif len(mono.shape) == 2:
        mono = mono[None]
    return mono


def readIDRCameras(path, image_list):
    # copy from IDR: https://github.com/lioryariv/idr/

    assert os.path.exists(path), "Data directory is empty"

    image_dir = '{0}/images'.format(path)
    image_paths = sorted(glob_imgs(image_dir))
    mask_dir = '{0}/mask'.format(path)
    mask_paths = sorted(glob_imgs(mask_dir))
    cam_file = '{0}/cameras_sphere.npz'.format(path)


    camera_dict = np.load(cam_file)
    n_images = len(image_paths)

    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        pose_all.append(P)


    rgb_images = []
    scene_name = path.split('/')[-1]
    #image_list = image_list[scene_name]
    #n_images = len(image_list)

    for image_num in range(n_images):
        rgb = load_rgb(image_paths[image_num])
        # print(rgb.shape)
        # print(type(rgb))
        # print(rgb.dtype)
        rgb_images.append(rgb)

    object_masks = []
    for image_num in range(n_images):
        object_mask = load_mask(mask_paths[image_num])
        #print(object_mask.shape)
        #print(object_mask.dtype)
        object_masks.append(object_mask)

    if len(object_masks) == 0:
        object_masks = [np.ones_like(i[:1]) for i in rgb_images]

    # feats_list = []
    # scale_list = [1, 2]
    # ori_w, ori_h = 768, 576
    # #ori_w , ori_h = 1600, 1200
    # for scale in scale_list:
    #     feat_ext = FeatExt().cuda()
    #     feat_ext.eval()
    #     for p in feat_ext.parameters():
    #         p.requires_grad = False

    #     size_w = ori_w // scale
    #     size_h = ori_h // scale

    #     rgb_2xd = torch.zeros(n_images, 3, size_h, size_w)

    #     for i in range(n_images):
    #         image_pil = Image.open(image_paths[i])
    #         resolution = (int(image_pil.size[0] / scale), int(image_pil.size[1] / scale))
    #         image = torch.cat([PILtoTorch(im, resolution) for im in image_pil.split()[:3]], dim=0)
    #         rgb_2xd[i, :, :image.shape[1], :image.shape[2]] = image

    #     mean = torch.tensor([0.485, 0.456, 0.406]).float()
    #     std = torch.tensor([0.229, 0.224, 0.225]).float()
    #     rgb_2xd = (rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    #     feats = []
    #     feat_eval_bs = 20
    #     #print(f"n_images = {n_images}")
    #     for start_i in range(0, n_images, feat_eval_bs):
    #         #print(f"start_i = {start_i}")
    #         eval_batch = rgb_2xd[start_i:start_i + feat_eval_bs]
    #         #print(f"eval_batch.shape = {eval_batch.shape}")
    #         feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu()
    #         #print(f"feat2.shape = {feat2.shape}")
    #         feats.append(feat2)
    #     feats = torch.cat(feats, dim=0)
        
    #     åfeats = feats[..., :(ori_h // 2) // scale, :(ori_w // 2) // scale]
            
    #     #print(feats.shape)
    #     feats_list.append(feats)



    cam_infos = []
    for i in range(n_images):
        P = pose_all[i]
        K, R, t = cv2.decomposeProjectionMatrix(P[:3, :4])[:3]
        #print(f"k1 = {K}")
        K = K / K[2, 2]
        K[0,1] = 0.0
        #print(f"k2 = {K}")
        t = t[:3, :] / t[3:, :]
        T = -R @ t
        T = T[:, 0]
        R = R.T

        # print(R, T)
        # exit()
        
        image_path = image_paths[i]
        image_name = image_path.split('.')[0].split('/')[-1]
        print(f"image name : {image_name}")
        uid = int(image_name.split('/')[-1])
        image = (rgb_images[i].transpose([1, 2, 0]) * 0.5 + 0.5) * 255
        
        # mono_depth_path = os.path.join(path, 'depth_npy', f"{image_num:08d}_pred.npy")
        # mono_depth = np.load(mono_depth_path)
        #mono_depth = None
        FovY = focal2fov(K[1, 1], image.shape[0])
        FovX = focal2fov(K[0, 0], image.shape[1])
        
        if image.shape[-1] == 4:
            alpha = image[..., 3:] / 255
            object_masks[i] *= alpha.transpose([2, 0, 1])
            image = image[..., :3]
        image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")
        prcppoint = K[:2, 2] / image.size[:2]
        mask = np.array(object_masks[i])
        # print(K[:2, 2], image.size[:2])
        # exit()
        # flow = []
        # for i in range(3):
        #     for j in range(3):
        #         if i == j:
        #             continue
        #         flow_path = os.path.join(path, 'flow', image_name + f'_{i}_{j}.flo')
        #         if os.path.exists(flow_path):
        #             print(f"Reading flow from {flow_path}")
        #             flow.append((j, readFlow(flow_path)))
        #             x = readFlow(flow_path)
        #             x -= x.min()
        #             x /= x.max()
        #             x *= 255
        #             y = np.zeros((x.shape[0], x.shape[1], 3))
        #             y[:, :, :2] = x
        #             #print(x.shape)
        #             cv2.imwrite(f"test.png", y)

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                        mask = mask))
    return cam_infos

def readIDRSceneInfo(path, eval, image_list, testskip=8):
    cam_infos = readIDRCameras(path, image_list)

    if eval:
        # test_cams = [i for i in range(len(cam_infos)) if i % testskip == 0]
        # test split following NeuS2
        test_cams = [8, 13, 16, 21, 26, 31, 34]
        if len(cam_infos) > 56:
            test_cams.append(56)

        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in test_cams]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_cams]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "points3d.ply")
    ply_path = os.path.join(path, "pcd", path.split('/')[-1]+".ply")
    print(ply_path)
    if os.path.exists(ply_path):
        pcd = fetchPly(ply_path)
        print(f"Featching points3d.ply...")
    else:
        num_pts = 100_0000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        rand_scale = 1.2
        normal = np.random.random((num_pts, 3)) - 0.5
        normal /= np.linalg.norm(normal, 2, 1, True)
        xyz = normal * 0.5 #- rand_scale / 2

        # normal = np.repeat(np.array([[0, 1, 0.0]]), num_pts, 0)
        # normal = np.random.random((num_pts, 3)) - 0.5
        rand_scale *= 2
        xyz = np.random.random((num_pts, 3)) * rand_scale - rand_scale / 2

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normal)

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cam_infos,
                        test_cameras=test_cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path)
    
    return scene_info




sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "IDR": readIDRSceneInfo
}