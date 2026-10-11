#
# Stereo matching network f_psi (DEFOM-Stereo) with left-right consistency check.
#
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

DEFOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'DEFOM-Stereo'))
sys.path.append(DEFOM_DIR)
from core.defom_stereo import DEFOMStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


class Stereo:
    def __init__(self, args, device='cuda'):
        """Frozen DEFOM-Stereo (ViT-L). Gradients can still flow through it into the right (stereo-view) image."""
        self.args = args
        self.device = device
        self.valid_iters = args.stereo_valid_iters
        self.train_iters = args.stereo_train_iters
        self.scale_iters = args.stereo_scale_iters

        ckpt = args.stereo_ckpt or os.path.join(DEFOM_DIR, 'checkpoints', 'defomstereo_vitl_sceneflow.pth')
        model_args = Namespace(
            dinov2_encoder='vitl',
            idepth_scale=0.5,
            hidden_dims=[128] * 3,
            corr_implementation='reg',
            shared_backbone=False,
            corr_levels=2,
            corr_radius=4,
            scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            scale_corr_radius=2,
            n_downsample=2,
            context_norm="batch",
            n_gru_layers=3,
            mixed_precision=False,
        )
        self.model = DEFOMStereo(model_args)
        checkpoint = torch.load(ckpt, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        checkpoint = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _disparity(self, image1, image2, iters):
        return self.model(image1, image2, iters=iters, scale_iters=min(self.scale_iters, iters), test_mode=True)

    def run(self, image1, image2, baseline_percentage, fx, radius, with_grad=False):
        """
        Left (train) view depth from a rectified stereo pair.

        image1, image2: [1, 3, H, W] in [0, 255] (left: train image, right: rendered stereo view)
        baseline_percentage: baseline in % of the scene radius
        fx: focal length in pixels at the input resolution
        with_grad: keep the graph of the left->right disparity so that gradients reach image2

        Returns depth [H, W] and the left-right consistency mask [H, W] (True = consistent / not occluded).
        """
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.set_grad_enabled(with_grad):
            disp_LR = self._disparity(image1, image2, self.train_iters if with_grad else self.valid_iters)
        disp_LR = padder.unpad(disp_LR).squeeze()

        with torch.no_grad():
            disp_RL = self._disparity(torch.flip(image2, dims=[3]), torch.flip(image1, dims=[3]), self.valid_iters)
            disp_RL = padder.unpad(torch.flip(disp_RL, dims=[3])).squeeze()

        consistent = self.left_right_consistency(disp_LR.detach().cpu().numpy(), disp_RL.cpu().numpy(),
                                                 self.args.stereo_occlusion_threshold)

        baseline = radius * baseline_percentage / 100
        depth = (fx * baseline) / disp_LR.clamp(min=1e-3)
        consistent = torch.from_numpy(consistent).to(depth.device) & (disp_LR.detach() > 1e-3)
        return depth, consistent

    @staticmethod
    def left_right_consistency(L2R_disparity, R2L_disparity, threshold):
        """
        Warp every left pixel to the right view with the L->R disparity and back with the R->L disparity.
        Pixels that do not return within `threshold` pixels (or leave the image) are marked as occluded (False).
        """
        height, width = L2R_disparity.shape
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

        x_projected = (x_grid - L2R_disparity).astype(np.int32)
        x_projected_clipped = np.clip(x_projected, 0, width - 1)

        x_reprojected = x_projected_clipped + R2L_disparity[y_grid, x_projected_clipped]
        x_reprojected_clipped = np.clip(x_reprojected, 0, width - 1)

        disparity_difference = np.abs(x_grid - x_reprojected_clipped)
        occluded = disparity_difference > threshold
        occluded[(x_projected < 0) | (x_projected >= width)] = True
        return ~occluded


def resize_image(image, scale):
    """[C, H, W] -> [C, h, w] bilinear resize (no-op for scale == 1)."""
    if scale == 1:
        return image
    return F.interpolate(image[None], scale_factor=scale, mode='bilinear', align_corners=False)[0]
