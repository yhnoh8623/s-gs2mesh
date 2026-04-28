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

import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l2_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l2_loss(network_output, gt)
    else:
        (((network_output - gt) * mask) ** 2).sum() / mask.sum()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)





def patchify(img, patch_size):
    img = img.unsqueeze(0)
    img = F.unfold(img, patch_size, stride=patch_size)
    img = img.transpose(2, 1).contiguous()
    return img.view(-1, patch_size, patch_size)

def patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    length = (surf_depth_patches.shape[1]) // 2 * 2
    rand_indices = torch.randperm(length)
    surf_depth_patches_rand = surf_depth_patches[:, rand_indices]
    mono_depth_patches_rand = mono_depth_patches[:, rand_indices]

    patch_rank_loss = torch.max(
        torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * \
            (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin,
        torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    ).mean()

    return patch_rank_loss

def get_depth_ranking_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0

    for transform in [transform1, transform2]:
        surf_depth_crop = transform(surf_depth)
        mono_depth_crop = transform(mono_depth.unsqueeze(0))

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = transform(object_mask)
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)

    return depth_rank_loss