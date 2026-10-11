#
# Sparse-GS2Mesh mesh extraction
#   For every train view, render the stereo view with the fine-tuned 3DGS (baseline B* of Sec 3.2),
#   estimate the train-view depth with the stereo network, complete the left-right inconsistent
#   (occluded) regions with the epipolar depth prior (Sec 4.2) and fuse the depths with TSDF.
#

import os
import sys
import copy
import json
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import StereoCamera
from train import prepare_train_cameras, stereo_depth
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, fov2focal
from utils.stereo_utils import Stereo


def resize_to(t, H, W, mode):
    if t.shape[-2:] == (H, W):
        return t
    return F.interpolate(t[None, None].float(), size=(H, W), mode=mode)[0, 0]


@torch.no_grad()
def compute_depths(args, scene, gaussians, stereo, pipe, background, baselines, out_dir):
    results = []
    for cam in scene.getTrainCameras():
        stereo_cam = StereoCamera(cam, baselines[cam.image_name], scene.radius)
        stereo_image = render(stereo_cam, gaussians, pipe, background)["render"]
        depth, consistent = stereo_depth(stereo, cam, stereo_image, stereo_cam.baseline, args, with_grad=False)
        depth = resize_to(depth, cam.image_height, cam.image_width, 'bilinear')
        consistent = resize_to(consistent, cam.image_height, cam.image_width, 'nearest') > 0.5

        # left-right consistent stereo depth, occluded regions completed with the epipolar depth prior
        depth = torch.where(consistent, depth, torch.zeros_like(depth))
        flow_depth = cam.flow_depth.squeeze().cuda().float()
        flow_valid = cam.flow_depth_mask.squeeze().cuda().bool() & (flow_depth > 0)
        depth = torch.where((~consistent) & flow_valid, flow_depth, depth)
        depth = depth * cam.object_mask

        np.save(os.path.join(out_dir, f"{cam.image_name}_depth.npy"), depth.cpu().numpy())
        print(f"{cam.image_name}: baseline {stereo_cam.baseline}%  lr-consistent {consistent[cam.object_mask].float().mean():.3f}")
        results.append((cam, depth.cpu().numpy()))
    return results


def tsdf_fusion(args, results):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.tsdf_voxel,
        sdf_trunc=args.tsdf_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for cam, depth in results:
        H, W = cam.image_height, cam.image_width
        image = (cam.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(image)), o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0, depth_trunc=args.tsdf_depth_trunc, convert_rgb_to_intensity=False)
        # principal point at the pixel-center convention of the rasterizer
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fov2focal(cam.FoVx, W), fov2focal(cam.FoVy, H),
                                                      (W - 1) / 2., (H - 1) / 2.)
        extrinsic = getWorld2View2(cam.R, cam.T).astype(np.float64)
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def clean_mesh(mesh, min_ratio):
    """Remove connected components smaller than min_ratio * (largest component)."""
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) == 0:
        return mesh
    cleaned = copy.deepcopy(mesh)
    cleaned.remove_triangles_by_mask(cluster_n_triangles[triangle_clusters] < min_ratio * cluster_n_triangles.max())
    cleaned.remove_unreferenced_vertices()
    return cleaned


if __name__ == "__main__":
    parser = ArgumentParser(description="Sparse-GS2Mesh mesh extraction")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mesh_stereo_resolution", type=float, default=1.0, help="input scale of the stereo network")
    parser.add_argument("--mesh_occlusion_threshold", type=float, default=3.0, help="left-right consistency threshold (pixels)")
    parser.add_argument("--tsdf_voxel", type=float, default=0.004)
    parser.add_argument("--tsdf_sdf_trunc", type=float, default=0.008)
    parser.add_argument("--tsdf_depth_trunc", type=float, default=5.0)
    parser.add_argument("--clean_min_ratio", type=float, default=0.05)
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    final_dir = os.path.join(args.model_path, "final")
    out_dir = os.path.join(args.model_path, "mesh")
    os.makedirs(out_dir, exist_ok=True)
    args.stereo_resolution = args.mesh_stereo_resolution
    args.stereo_occlusion_threshold = args.mesh_occlusion_threshold

    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    scene_args = copy.copy(args)
    scene_args.model_path = out_dir
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(scene_args, gaussians, shuffle=False)
    (model_params, _) = torch.load(os.path.join(final_dir, "chkpnt_3d.pth"), weights_only=False)
    gaussians.restore(model_params, op.extract(args))
    prepare_train_cameras(scene, args)

    with open(os.path.join(final_dir, "baselines.json")) as f:
        baselines = json.load(f)

    stereo = Stereo(args, device='cuda')
    results = compute_depths(args, scene, gaussians, stereo, pp.extract(args), background, baselines, out_dir)
    del stereo
    torch.cuda.empty_cache()

    mesh = tsdf_fusion(args, results)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.ply"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh_cleaned.ply"), clean_mesh(mesh, args.clean_min_ratio))
    print(f"Saved {os.path.join(out_dir, 'mesh_cleaned.ply')}")
