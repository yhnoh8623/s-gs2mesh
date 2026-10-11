#
# Vis-MVSNet feature extractor f_phi (weights: utils/vismvsnet.pt) and the multi-view feature
# consistency used for the adaptive baseline score (Eq. 4) and L_stereo (Eq. 7).
#
import os
from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ListModule(nn.Module):
    def __init__(self, modules: Union[List, OrderedDict]):
        super(ListModule, self).__init__()
        if isinstance(modules, OrderedDict):
            iterable = modules.items()
        elif isinstance(modules, list):
            iterable = enumerate(modules)
        else:
            raise TypeError('modules should be OrderedDict or List.')
        for name, module in iterable:
            if not isinstance(module, nn.Module):
                module = ListModule(module)
            if not isinstance(name, str):
                name = str(name)
            self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dim=2):
        super(BasicBlock, self).__init__()

        self.conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        self.bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = self.bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = self.bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return self.conv_fn(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return self.conv_fn(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _make_layer(inplanes, block, planes, blocks, stride=1, dim=2):
    downsample = None
    conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
    bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv_fn(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            bn_fn(planes * block.expansion)
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, dim=dim))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, dim=dim))

    return nn.Sequential(*layers)


class UNet(nn.Module):

    def __init__(self, inplanes: int, enc: int, dec: int, initial_scale: int,
                 bottom_filters: List[int], filters: List[int], head_filters: List[int],
                 prefix: str, dim: int=2):
        super(UNet, self).__init__()

        conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
        deconv_fn = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
        current_scale = initial_scale
        idx = 0
        prev_f = inplanes

        self.bottom_blocks = OrderedDict()
        for f in bottom_filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx==0 else 2, dim=dim)
            self.bottom_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.bottom_blocks = ListModule(self.bottom_blocks)

        self.enc_blocks = OrderedDict()
        for f in filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx == 0 else 2, dim=dim)
            self.enc_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.enc_blocks = ListModule(self.enc_blocks)

        self.dec_blocks = OrderedDict()
        for f in filters[-2::-1]:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False),
                conv_fn(2*f, f, 3, 1, 1, bias=False),
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            self.dec_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.dec_blocks = ListModule(self.dec_blocks)

        self.head_blocks = OrderedDict()
        for f in head_filters:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False)
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            block = nn.Sequential(*block)
            self.head_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.head_blocks = ListModule(self.head_blocks)

    def forward(self, x, multi_scale=1):
        for b in self.bottom_blocks:
            x = b(x)
        enc_out = []
        for b in self.enc_blocks:
            x = b(x)
            enc_out.append(x)
        dec_out = [x]
        for i, b in enumerate(self.dec_blocks):
            if len(b) == 3: deconv, post_concat, res = b
            elif len(b) == 2: deconv, post_concat = b
            x = deconv(x)
            x = torch.cat([x, enc_out[-2-i]], 1)
            x = post_concat(x)
            if len(b) == 3: x = res(x)
            dec_out.append(x)
        for b in self.head_blocks:
            x = b(x)
            dec_out.append(x)
        if multi_scale == 1: return x
        else: return dec_out[-multi_scale:]


class FeatExt(nn.Module):

    def __init__(self):
        super(FeatExt, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.unet = UNet(16, 2, 1, 2, [], [32, 64, 128], [], '2d', 2)
        self.final_conv_1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vismvsnet.pt')
        state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        self.load_state_dict({k[16:]: v for k, v in state_dict.items() if k.startswith('module.feat_ext')})

    def forward(self, x):
        out = self.init_conv(x)
        out1, out2, out3 = self.unet(out, multi_scale=3)
        return self.final_conv_1(out1), self.final_conv_2(out2), self.final_conv_3(out3)


FEAT_STRIDES = [2, 4]  # cam.feat[0]: Vis-MVSNet features of the full image (1/2), cam.feat[1]: of the 1/2 image (1/4)


def depth_to_world_points(cam, depth):
    """
    Unproject a (possibly downscaled) depth map of `cam` to world points (Eq. 2).
    depth: [h, w] z-depth. Returns [h*w, 3].
    """
    h, w = depth.shape[-2:]
    c2w = (cam.world_view_transform.T).inverse()
    K = cam.intrinsic[:3, :3].clone()
    K[0] *= w / cam.image_width
    K[1] *= h / cam.image_height
    grid_x, grid_y = torch.meshgrid(torch.arange(w, device=depth.device).float(),
                                    torch.arange(h, device=depth.device).float(), indexing='xy')
    pix = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = pix @ K.inverse().T @ c2w[:3, :3].T
    return depth.reshape(-1, 1) * rays_d + c2w[:3, 3]


def project_points(pts, cam):
    """World points [N, 3] -> full-resolution pixel coords [N, 2] and z-depth [N] (Eq. 3)."""
    w2c = cam.world_view_transform.T
    p_cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = p_cam[:, 2]
    uv = p_cam @ cam.intrinsic[:3, :3].T
    uv = uv[:, :2] / z.clamp(min=1e-6)[:, None]
    return uv, z


def zbuffer_visibility(pts, uv, cam):
    """Keep the closest point per (rounded) source pixel - occlusion mask M_occ^{ref-src}."""
    order = torch.argsort(torch.norm(pts - cam.camera_center[None], dim=-1))
    uv_r = uv[order].round().long()
    key = uv_r[:, 1] * 100000 + uv_r[:, 0]
    _, inv = torch.unique(key, return_inverse=True)
    pos = torch.arange(key.numel(), device=key.device)
    first = torch.full((int(inv.max()) + 1,), key.numel(), device=key.device, dtype=torch.long)
    first = first.scatter_reduce(0, inv, pos, reduce='amin')
    vis_sorted = torch.zeros(key.numel(), dtype=torch.bool, device=key.device)
    vis_sorted[first] = True
    vis = torch.zeros_like(vis_sorted)
    vis[order] = vis_sorted
    return vis


def sample_feat(feat, uv, stride):
    """Bilinearly sample feat [C, h, w] at full-resolution pixel coords uv [N, 2]."""
    C, h, w = feat.shape
    g = uv / stride
    g = torch.stack([g[:, 0] / w * 2 - 1, g[:, 1] / h * 2 - 1], dim=-1)
    out = F.grid_sample(feat[None], g[None, :, None, :], mode='bilinear', padding_mode='zeros', align_corners=False)
    return out[0, :, :, 0].T  # [N, C]


def multiview_feature_consistency(pts, ref_cam, src_cams, valid, reduction='mean'):
    """
    Cosine similarity between Vis-MVSNet features of ref pixels and their reprojections in the source views.

    pts: [N, 3] world points of the ref depth map (gradients flow through pts into the depth).
    valid: [N] bool mask of usable ref pixels.
    reduction: 'score' -> S = sum cos (Eq. 4)
               'mean'  -> mean (1 - cos) over the visible pairs (Eq. 7)
    """
    if valid.sum() == 0:
        return torch.tensor(-float('inf') if reduction == 'score' else 0.0, device=pts.device)
    pts = pts[valid]
    ref_uv, _ = project_points(pts, ref_cam)

    total = 0.0
    count = 0.0
    for level, stride in enumerate(FEAT_STRIDES):
        f_ref = sample_feat(ref_cam.feat[level], ref_uv.detach(), stride)
        for src in src_cams:
            uv, z = project_points(pts, src)
            h, w = src.feat[level].shape[-2:]
            in_range = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w * stride) & (uv[:, 1] < h * stride)
            with torch.no_grad():
                vis = zbuffer_visibility(pts.detach(), uv.detach(), src) & in_range
            if vis.sum() == 0:
                continue
            f_src = sample_feat(src.feat[level], uv[vis], stride)
            cos = F.cosine_similarity(f_ref[vis], f_src, dim=-1, eps=1e-9)
            if reduction == 'score':
                total = total + cos.sum()
            else:
                total = total + (1 - cos).sum()
            count = count + cos.numel()
    if reduction == 'score':
        return torch.as_tensor(total, device=pts.device).float()
    count = torch.tensor(float(count), device=pts.device)
    return torch.as_tensor(total, device=pts.device) / count.clamp(min=1)
