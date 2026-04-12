import numpy as np
import torch
from utils.graphics_utils import BasicPointCloud
import open3d as o3d

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
        
def compute_depth_by_flow(cams, valid_dis_threshold, n):
    for i, cam_s in enumerate(cams):
        H, W = cam_s.image_height, cam_s.image_width
        R_s = torch.tensor(cam_s.R.transpose()).cuda().float()
        T_s = torch.tensor(cam_s.T.transpose()).cuda().float().view(3, 1)
        K_s = torch.tensor(cam_s.K.transpose()).cuda().float()
        #print(R_s, T_s, K_s)
        #print(cam_s.image_name)
        image_s = cam_s.original_image.cuda()


        cams_near = []
        for j, flow in cam_s.flow:
            cam_t = cams[j]
            R_t = torch.tensor(cam_t.R.transpose()).cuda().float()
            T_t = torch.tensor(cam_t.T.transpose()).cuda().float().view(3, 1)
            cam_dis = ((R_s.inverse() @ T_s - R_t.inverse() @ T_t) ** 2).sum(0)
            cam_dis = torch.sqrt(cam_dis)
            cams_near.append((j, float(cam_dis), flow))
        cams_near = sorted(cams_near, key=lambda x:x[1])
    

        id_dis = 100000
        final_confidence = None
        final_depth = None
        final_mask = None

        for j, _, flow in cams_near[:min(n, len(cams_near))]:
            # if abs(i - j) > 1:
            #     continue
            flow = torch.tensor(flow).cuda()
            cam_t = cams[j]
            image_t = cam_t.original_image.cuda()

            R_t = torch.tensor(cam_t.R.transpose()).cuda().float()
            T_t = torch.tensor(cam_t.T.transpose()).cuda().float().view(3, 1)
            K_t = torch.tensor(cam_t.K.transpose()).cuda().float()

            coords_y = torch.arange(0, H).cuda()
            coords_x = torch.arange(0, W).cuda()
            y, x = torch.meshgrid(coords_y, coords_x)
            y, x = y.unsqueeze(2), x.unsqueeze(2)
            xyz_s = torch.cat([x, y, torch.ones_like(x).cuda().float()], dim=0).view(3, -1)
            

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

            u = (x.contiguous().view(H, W, 1) + flow[:, :, :1]).contiguous().view(1, -1)
            v = (y.contiguous().view(H, W, 1) + flow[:, :, 1:2]).contiguous().view(1, -1)

            x_t_near = (b * b * u - a * b * v - a * c) / (a ** 2 + b ** 2)
            y_t_near = (a * a * v - a * b * u - b * c) / (a ** 2 + b ** 2)
            xyz_t_near = torch.cat([x_t_near, y_t_near, torch.ones_like(x_t_near).cuda()], dim=0)

            dis = torch.abs((a * u + b * v + c) / torch.sqrt(a ** 2 + b ** 2))
            dis_max = dis.max()
            mask = (dis < valid_dis_threshold)

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
            #depth = depth.view(H, W, 1)

            CD_hat = R.inverse() @ K_t.inverse() @ torch.cat([u, v, torch.ones_like(u).cuda()], dim=0)
            DD_hat = CD_hat - CD
            AD_hat = CD_hat - CA

            x = torch.round(x_t_near).long()
            y = torch.round(y_t_near).long()
            valid = torch.logical_and(x >= 0, x <= W-1)
            valid = torch.logical_and(valid, torch.logical_and(y >=0, y <= H-1))
            color_dis = (image_s.view(3, -1) - image_t.view(3, -1)) ** 2
            color_dis = torch.sqrt(color_dis.sum(0))

            cam_dis = ((R_s.inverse() @ T_s - R_t.inverse() @ T_t) ** 2).sum(0, keepdim=True)
            cam_dis = torch.sqrt(cam_dis).repeat(1, H * W)
            

            len_CA = torch.sqrt((CA ** 2).sum(0, keepdim=True))
            len_AB = torch.sqrt((AB ** 2).sum(0, keepdim=True))
            len_CD = torch.sqrt((CD ** 2).sum(0, keepdim=True))
            len_CB = torch.sqrt((CB ** 2).sum(0, keepdim=True))
            len_AD = torch.sqrt((AD ** 2).sum(0, keepdim=True))
            len_CD_hat = torch.sqrt((CD_hat ** 2).sum(0, keepdim=True))
            len_DD_hat = torch.sqrt((DD_hat ** 2).sum(0, keepdim=True))
            len_AD_hat = torch.sqrt((AD_hat ** 2).sum(0, keepdim=True))
            cos_alpha = (len_CD**2 + len_CA**2 - len_AD**2) / (2 * len_CA * len_CD)
            cos_beta = (len_AB**2 + len_CA**2 - len_CB**2) / (2 * len_CA * len_AB)
            sin_alpha = torch.sqrt(1 - cos_alpha**2)
            sin_beta = torch.sqrt(1 - cos_beta**2)
            confidence = len_CA * sin_beta / ((sin_beta * cos_alpha + sin_alpha * cos_beta) ** 2)

            Ost = - R_t @ R_s.inverse() @ T_s + T_t
            Ost = K_t @ Ost
            AF = R.inverse() @ K_t.inverse() @ (Ost / Ost[2:3, :])
            CF = CA + AF
            len_CF = torch.sqrt((CF ** 2).sum(0, keepdim=True))
            DF = AF - AD
            len_DF = torch.sqrt((DF ** 2).sum(0, keepdim=True))
            cos_theta = (len_CF**2 + len_DF**2 - len_CD**2) / (2 * len_CF * len_DF)
            sin_theta = torch.sqrt(1 - cos_theta**2)
            confidence *= ((sin_alpha * cos_theta + cos_alpha * sin_theta) ** 2) / (len_CF * sin_theta)
            confidence = torch.abs(confidence)


            l =  len_CA * sin_alpha / (sin_beta * cos_alpha + sin_alpha * cos_beta)
            #ll = len_CA / (sin_beta * (cos_alpha/sin_alpha + cos_beta / sin_beta))
            #confidence = dis
            cos_alpha_hat = (len_CD_hat**2 + len_CA**2 - len_AD_hat**2) / (2 * len_CD_hat * len_CA)
            sin_alpha_hat = torch.sqrt(1 - cos_alpha_hat**2)
            #confidence = confidence * torch.acos(cos_alpha_hat)
            #confidence = torch.abs(len_CA * sin_beta / (sin_beta * cos_alpha_hat + sin_alpha_hat * cos_beta))

            d = torch.abs(l / len_AB)


            if final_confidence is None:
                final_confidence = confidence
                final_depth = depth
                final_mask = mask
            else:
                final_confidence = torch.where(torch.logical_and(final_mask == False, mask == True), confidence, final_confidence)
                final_depth = torch.where(torch.logical_and(final_mask == False, mask == True), depth, final_depth)
                final_mask = torch.where(torch.logical_and(final_mask == False, mask == True), mask, final_mask)

                final_confidence = torch.where(confidence < final_confidence, confidence, final_confidence)
                final_depth = torch.where(confidence < final_confidence, depth, final_depth)
                final_mask = torch.where(confidence < final_confidence, mask, final_mask)

        cams[i].set_flow_depth(final_depth.view(H, W, 1), final_mask.view(H, W, 1))

def construct_pcd(cams):
    zs = []
    colors = []
    full_proj_transform_invs = []
    Rs = []
    Ts = []
    pcd_idx = []
    pcd_origin_points = []
    masks = []
    coords = []
    num, H, W = len(cams), cams[0].image_height, cams[0].image_width

    points_num = 0
    for i,cam in enumerate(cams):
        R = cam.R
        T = cam.T
        K = cam.K


        depth_reg = cam.flow_depth.squeeze().cuda()
        flow_depth_mask = cam.flow_depth_mask.squeeze().cuda()
        gt_image = cam.original_image.cuda().permute(1,2,0).contiguous()
        # full_proj_transform_inv = cam.full_proj_transform.inverse()
        full_proj_transform_inv = cam.full_proj_transform
        R = torch.tensor(R).cuda()
        T = torch.tensor(T).cuda()

        if cam.alpha_mask is not None:
            alpha_mask = cam.alpha_mask.cuda().squeeze().bool()
            flow_depth_mask = torch.logical_and(flow_depth_mask, alpha_mask)
        
        vaild_num = int(flow_depth_mask.sum())
        mask = flow_depth_mask.view(-1)
        
        zs.append(depth_reg.view(-1, 1)[mask])
        colors.append(gt_image.view(-1, 3)[mask])
        full_proj_transform_invs.append(full_proj_transform_inv.view(1,4,4).repeat(vaild_num, 1, 1))
        K = torch.tensor(cam.K).cuda().float()
        Rs.append(R.inverse().view(1,3,3).repeat(vaild_num, 1, 1))
        Ts.append(T.view(1,1,3).repeat(vaild_num, 1, 1))
        # masks.append(flow_depth_mask.view(-1, 1)[mask])
        points_num += int(flow_depth_mask.sum())

        coords_y = torch.arange(0, H).cuda()
        coords_x = torch.arange(0, W).cuda()
        y, x = torch.meshgrid(coords_y, coords_x)
        y, x = y.unsqueeze(2), x.unsqueeze(2)
        coord = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().view(H*W, 3)[mask]
        coords.append(coord)
    
    zs = torch.cat(zs, dim=0).view(-1, 1).float().cuda()
    colors = torch.cat(colors, dim=0).cuda()
    Rs = torch.cat(Rs, dim=0).cuda()
    Ts = torch.cat(Ts, dim=0).cuda()
    full_proj_transform_invs = torch.cat(full_proj_transform_invs, dim=0).cuda()
    coords_y = torch.arange(0, H).cuda()
    coords_x = torch.arange(0, W).cuda()
    y, x = torch.meshgrid(coords_y, coords_x)
    y, x = y.unsqueeze(2), x.unsqueeze(2)
    coords = torch.cat(coords, dim=0).cuda().float()
    full_proj_transform_invs = full_proj_transform_invs.float()
    R_invs = Rs.float()
    Ts = Ts.float()
    K_invs = K.inverse().view(1, 3, 3).repeat(points_num, 1, 1).float()


    print("Number of points at initialisation : ", points_num)

    points = (coords * zs).view(-1, 1, 3)
    points = torch.bmm(points, K_invs)
    points = torch.bmm(points - Ts, R_invs).view(points_num, 3)

    points = points.cpu().numpy()
    colors = colors.cpu().numpy()
    
    target_num = 300000
    #target_num = 
    jump = max(1, len(points) // target_num)
    points = points[::jump]
    colors = colors[::jump]
    print("Number of points at initialisation : ", len(points))
    
        
    save_pointcloud(points, colors, path="initial_pcd.ply")
    return BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points))

def save_pointcloud(points, colors, path="pc.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(path, pcd)