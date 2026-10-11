#
# Epipolar depth initialization from optical flow (NexusGS, CVPR 2025).
#
import numpy as np
import torch

from utils.graphics_utils import BasicPointCloud


def readFlow(fn):
    """Read a .flo file (Middlebury format, little-endian)."""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        return np.resize(data, (int(h), int(w), 2))


def _pose(cam):
    R = torch.tensor(cam.R.transpose()).cuda().float()
    T = torch.tensor(cam.T.transpose()).cuda().float().view(3, 1)
    return R, T


def compute_depth_by_flow(cams, valid_dis_threshold, n):
    """
    For every camera, triangulate a depth map from the flows to its n nearest cameras: the flow target is
    projected onto the epipolar line, and among the valid estimates (distance to the epipolar line below
    valid_dis_threshold) the one with the lowest geometric uncertainty is kept.
    """
    for i, cam_s in enumerate(cams):
        H, W = cam_s.image_height, cam_s.image_width
        R_s, T_s = _pose(cam_s)
        K_s = torch.tensor(cam_s.K.transpose()).cuda().float()

        cams_near = []
        for j, flow in cam_s.flow:
            R_t, T_t = _pose(cams[j])
            cam_dis = torch.sqrt(((R_s.inverse() @ T_s - R_t.inverse() @ T_t) ** 2).sum(0))
            cams_near.append((j, float(cam_dis), flow))
        cams_near = sorted(cams_near, key=lambda x: x[1])

        final_confidence = None
        final_depth = None
        final_mask = None

        for j, _, flow in cams_near[:min(n, len(cams_near))]:
            flow = torch.tensor(flow).cuda()
            cam_t = cams[j]
            R_t, T_t = _pose(cam_t)
            K_t = torch.tensor(cam_t.K.transpose()).cuda().float()

            y, x = torch.meshgrid(torch.arange(0, H).cuda(), torch.arange(0, W).cuda(), indexing='ij')
            y, x = y.unsqueeze(2), x.unsqueeze(2)
            xyz_s = torch.cat([x, y, torch.ones_like(x).cuda().float()], dim=0).view(3, -1)

            # fundamental matrix
            R = R_t @ R_s.inverse()
            T = R_s @ R_t.inverse() @ T_t - T_s
            T = T.squeeze()
            S = torch.zeros((3, 3)).cuda()
            S[0, 1] = - T[2]
            S[1, 0] = T[2]
            S[0, 2] = T[1]
            S[2, 0] = - T[1]
            S[1, 2] = - T[0]
            S[2, 1] = T[0]
            E = R @ S
            F = K_t.inverse().transpose(0, 1) @ E @ K_s.inverse()

            epipolar_line = F @ xyz_s
            a = epipolar_line[0:1]
            b = epipolar_line[1:2]
            c = epipolar_line[2:3]

            # flow target and its projection onto the epipolar line
            u = (x.contiguous().view(H, W, 1) + flow[:, :, :1]).contiguous().view(1, -1)
            v = (y.contiguous().view(H, W, 1) + flow[:, :, 1:2]).contiguous().view(1, -1)
            x_t_near = (b * b * u - a * b * v - a * c) / (a ** 2 + b ** 2)
            y_t_near = (a * a * v - a * b * u - b * c) / (a ** 2 + b ** 2)
            xyz_t_near = torch.cat([x_t_near, y_t_near, torch.ones_like(x_t_near).cuda()], dim=0)

            dis = torch.abs((a * u + b * v + c) / torch.sqrt(a ** 2 + b ** 2))
            mask = (dis < valid_dis_threshold)

            # triangulation
            CD = R.inverse() @ K_t.inverse() @ xyz_t_near
            CA = - T.view(3, 1).repeat(1, CD.shape[1])
            AB = CE = K_s.inverse() @ xyz_s
            CB = CA + AB
            AD = CD - CA

            ACD = torch.cross(CD, CA, dim=0)
            CDE = torch.cross(CE, CD, dim=0)
            SACD = torch.sqrt((ACD ** 2).sum(dim=0))
            SCDE = torch.sqrt((CDE ** 2).sum(dim=0))
            depth = SACD / SCDE

            # geometric uncertainty of the triangulated depth
            len_CA = torch.sqrt((CA ** 2).sum(0, keepdim=True))
            len_AB = torch.sqrt((AB ** 2).sum(0, keepdim=True))
            len_CD = torch.sqrt((CD ** 2).sum(0, keepdim=True))
            len_CB = torch.sqrt((CB ** 2).sum(0, keepdim=True))
            len_AD = torch.sqrt((AD ** 2).sum(0, keepdim=True))
            cos_alpha = (len_CD ** 2 + len_CA ** 2 - len_AD ** 2) / (2 * len_CA * len_CD)
            cos_beta = (len_AB ** 2 + len_CA ** 2 - len_CB ** 2) / (2 * len_CA * len_AB)
            sin_alpha = torch.sqrt(1 - cos_alpha ** 2)
            sin_beta = torch.sqrt(1 - cos_beta ** 2)
            confidence = len_CA * sin_beta / ((sin_beta * cos_alpha + sin_alpha * cos_beta) ** 2)

            Ost = - R_t @ R_s.inverse() @ T_s + T_t
            Ost = K_t @ Ost
            AF = R.inverse() @ K_t.inverse() @ (Ost / Ost[2:3, :])
            CF = CA + AF
            len_CF = torch.sqrt((CF ** 2).sum(0, keepdim=True))
            DF = AF - AD
            len_DF = torch.sqrt((DF ** 2).sum(0, keepdim=True))
            cos_theta = (len_CF ** 2 + len_DF ** 2 - len_CD ** 2) / (2 * len_CF * len_DF)
            sin_theta = torch.sqrt(1 - cos_theta ** 2)
            confidence *= ((sin_alpha * cos_theta + cos_alpha * sin_theta) ** 2) / (len_CF * sin_theta)
            confidence = torch.abs(confidence)

            if final_confidence is None:
                final_confidence = confidence
                final_depth = depth
                final_mask = mask
            else:
                newly_valid = torch.logical_and(final_mask == False, mask == True)
                final_confidence = torch.where(newly_valid, confidence, final_confidence)
                final_depth = torch.where(newly_valid, depth, final_depth)
                final_mask = torch.where(newly_valid, mask, final_mask)

                final_confidence = torch.where(confidence < final_confidence, confidence, final_confidence)
                final_depth = torch.where(confidence < final_confidence, depth, final_depth)
                final_mask = torch.where(confidence < final_confidence, mask, final_mask)

        cams[i].set_flow_depth(final_depth.view(H, W, 1), final_mask.view(H, W, 1))


def construct_pcd(cams):
    """Back-project the valid epipolar depths of all cameras (inside the object masks) to a point cloud."""
    zs = []
    colors = []
    Rs = []
    Ts = []
    coords = []
    H, W = cams[0].image_height, cams[0].image_width

    points_num = 0
    for cam in cams:
        depth = cam.flow_depth.squeeze().cuda()
        flow_depth_mask = cam.flow_depth_mask.squeeze().cuda()
        gt_image = cam.original_image.cuda().permute(1, 2, 0).contiguous()
        R = torch.tensor(cam.R).cuda()
        T = torch.tensor(cam.T).cuda()

        if cam.alpha_mask is not None:
            flow_depth_mask = torch.logical_and(flow_depth_mask, cam.alpha_mask.cuda().squeeze().bool())

        valid_num = int(flow_depth_mask.sum())
        mask = flow_depth_mask.view(-1)

        zs.append(depth.view(-1, 1)[mask])
        colors.append(gt_image.view(-1, 3)[mask])
        K = torch.tensor(cam.K).cuda().float()
        Rs.append(R.inverse().view(1, 3, 3).repeat(valid_num, 1, 1))
        Ts.append(T.view(1, 1, 3).repeat(valid_num, 1, 1))
        points_num += valid_num

        y, x = torch.meshgrid(torch.arange(0, H).cuda(), torch.arange(0, W).cuda(), indexing='ij')
        y, x = y.unsqueeze(2), x.unsqueeze(2)
        coords.append(torch.stack([x, y, torch.ones_like(x)], dim=-1).float().view(H * W, 3)[mask])

    zs = torch.cat(zs, dim=0).view(-1, 1).float().cuda()
    colors = torch.cat(colors, dim=0).cuda()
    R_invs = torch.cat(Rs, dim=0).cuda().float()
    Ts = torch.cat(Ts, dim=0).cuda().float()
    coords = torch.cat(coords, dim=0).cuda().float()
    K_invs = K.inverse().view(1, 3, 3).repeat(points_num, 1, 1).float()

    points = (coords * zs).view(-1, 1, 3)
    points = torch.bmm(points, K_invs)
    points = torch.bmm(points - Ts, R_invs).view(points_num, 3)

    points = points.cpu().numpy()
    colors = colors.cpu().numpy()

    # subsample to at most ~300k points
    jump = max(1, len(points) // 300000)
    points = points[::jump]
    colors = colors[::jump]
    print("Number of points at initialisation : ", len(points))
    return BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points))
