#
# Sparse-view DTU protocol: keep the mesh parts that project inside the (dilated) object masks of the
# input views, then transform the mesh from the normalized frame to the DTU world frame (scale_mat).
#
import os
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K, R, t = out[0], out[1], out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose


def cull_scan(scan_dir, mesh_path, result_mesh_file):
    """scan_dir: scan folder with images/, mask/ (input views) and cameras.npz."""
    n_images = len(sorted(glob.glob(os.path.join(scan_dir, "images", "*.png"))))
    camera_dict = np.load(os.path.join(scan_dir, "cameras.npz"))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        intrinsics, pose = load_K_Rt_from_P((world_mat @ scale_mat)[:3, :4])
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    masks = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(scan_dir, "mask", "*.png")))]
    W, H = 1600, 1200

    mesh = trimesh.load(mesh_path)
    vertices = torch.from_numpy(mesh.vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1).permute(1, 0).float()

    sampled_masks = []
    for i in tqdm(range(n_images)):
        w2c = torch.inverse(pose_all[i]).cuda()
        intrinsic = intrinsics_all[i].cuda()
        with torch.no_grad():
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1.) & (pix_coords < 1.)).all(dim=-1).float()

            # dilate the mask as in UNISURF
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]
            sampled_masks.append(sampled_mask + (1. - valid))

    mask = (torch.stack(sampled_masks, -1) > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)

    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cull a mesh with the input-view masks and move it to the DTU frame.")
    parser.add_argument('--scan_dir', type=str, required=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    cull_scan(args.scan_dir, args.mesh, args.out)
