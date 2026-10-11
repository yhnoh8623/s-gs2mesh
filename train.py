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
# Sparse-GS2Mesh training
#   Sec 3.1  3DGS / 2DGS warm-up from the epipolar depth initialization
#   Sec 3.2  adaptive baseline selection from multi-view feature scores
#   Sec 3.3  fine-tuning 3DGS through the stereo matching network (L_stereo)
#   Sec 3.4  2D/3D co-regularized fine-tuning (L_co-img, L_co-depth, L_depth)
#

import os
import sys
import json
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, render_2dgs
from scene import Scene, GaussianModel, SurfelModel
from scene.cameras import StereoCamera
from utils.feat_utils import depth_to_world_points, multiview_feature_consistency
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, l2_loss_mask, ssim
from utils.stereo_utils import Stereo, resize_image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# =============================================================================
#  Helpers
# =============================================================================

def is_dtu(args):
    return args.dataset_type.lower() == 'dtu'


def dtu_bg_mask(gt_image, args):
    """Dark DTU background (NexusGS heuristic)."""
    threshold = 15 / 255 if 'scan110' in args.source_path else 30 / 255
    bg_mask = (gt_image.max(0, keepdim=True).values < threshold)
    bg_mask_clone = bg_mask.clone()
    for i in range(1, 50):
        bg_mask[:, i:] *= bg_mask_clone[:, :-i]
    return bg_mask


def prepare_train_cameras(scene, args):
    """Black out the DTU background of the GT images and cache the object masks (full and stereo resolution)."""
    for cam in scene.getTrainCameras():
        cam.bg_mask = None
        if is_dtu(args):
            cam.bg_mask = dtu_bg_mask(cam.original_image, args)
            cam.original_image[cam.bg_mask.repeat(3, 1, 1)] = 0.
        H, W = cam.image_height, cam.image_width
        h, w = int(H * args.stereo_resolution), int(W * args.stereo_resolution)
        obj = cam.alpha_mask.float().cuda() if cam.alpha_mask is not None else torch.ones((H, W), device='cuda')
        cam.object_mask = obj > 0.5
        cam.stereo_object_mask = F.interpolate(obj[None, None], size=(h, w), mode='nearest')[0, 0] > 0.5


def suppress_background(cam, gaussians, renderer, pipe, background, alpha_key):
    """Push the opacity of the black DTU background to zero (NexusGS)."""
    if cam.bg_mask is None:
        return
    render_pkg = renderer(cam, gaussians, pipe, background)
    (render_pkg[alpha_key][cam.bg_mask] ** 2).mean().backward()
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)


def rgb_loss(image, gt_image, opt):
    Ll1 = l1_loss(image, gt_image)
    return Ll1, (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


def stereo_depth(stereo, cam, stereo_image, baseline, args, with_grad):
    """D^tr = f_psi(I^tr, I^st) at args.stereo_resolution (Eq. 6), with its left-right consistency mask."""
    left = resize_image(cam.original_image, args.stereo_resolution)[None] * 255
    right = resize_image(stereo_image.clamp(0, 1), args.stereo_resolution)[None] * 255
    fx = fov2focal(cam.FoVx, cam.image_width) * left.shape[-1] / cam.image_width
    return stereo.run(left, right, baseline, fx, cam.radius, with_grad=with_grad)


def source_cameras(cam, cameras):
    return [c for c in cameras if c is not cam]


@torch.no_grad()
def prune_invisible(scene, gaussians, pipe, background):
    """Remove Gaussians that are not visible in any train view."""
    visible = None
    for cam in scene.getTrainCameras():
        visibility_filter = render(cam, gaussians, pipe, background)["visibility_filter"]
        visible = visibility_filter if visible is None else visible + visibility_filter
    gaussians.prune_points(~visible)


def prepare_output_and_logger(args):
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


@torch.no_grad()
def training_report(tb_writer, iteration, scene, gaussians, pipe, background):
    torch.cuda.empty_cache()
    cameras = scene.getTrainCameras()
    l1_test, psnr_test = 0.0, 0.0
    for cam in cameras:
        image = torch.clamp(render(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
        gt_image = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
    l1_test, psnr_test = l1_test / len(cameras), psnr_test / len(cameras)
    print("\n[ITER {}] Evaluating train: L1 {} PSNR {}".format(iteration, l1_test, psnr_test))
    if tb_writer:
        tb_writer.add_scalar('train/l1_loss', l1_test, iteration)
        tb_writer.add_scalar('train/psnr', psnr_test, iteration)
        tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()


# =============================================================================
#  Sec 3.1 : 3DGS warm-up (photometric loss)
# =============================================================================

def training_3d(dataset, opt, pipe, args):
    tb_writer = prepare_output_and_logger(args)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    prepare_train_cameras(scene, args)
    torch.cuda.empty_cache()

    # invisible Gaussians are removed at the first iteration and at the test iterations (no densification there)
    clean_iterations = args.test_iterations + [1]

    progress_bar = tqdm(range(opt.warmup_iterations), desc="3DGS warm-up")
    ema_loss_for_log = 0.0
    viewpoint_stack = None

    for iteration in range(1, opt.warmup_iterations + 1):
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        if is_dtu(args):
            suppress_background(viewpoint_cam, gaussians, render, pipe, background, "alpha")

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        Ll1, loss = rgb_loss(image, gt_image, opt)
        loss.backward()

        with torch.no_grad():
            if not loss.isnan():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Points": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.warmup_iterations:
                progress_bar.close()

            if is_dtu(args) and iteration in clean_iterations:
                prune_invisible(scene, gaussians, pipe, background)

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
            if iteration in args.test_iterations:
                training_report(tb_writer, iteration, scene, gaussians, pipe, background)

            if iteration == opt.warmup_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

            # densification
            if iteration < opt.densify_until_iter and iteration not in clean_iterations:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent,
                                                opt.size_threshold, opt.split_num)

            if iteration < opt.warmup_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            gaussians.update_learning_rate(iteration)
            if iteration > opt.opacity_reset_from_iter and \
                    (iteration - opt.opacity_reset_from_iter - 1) % opt.opacity_reset_interval == 0:
                gaussians.reset_opacity()


# =============================================================================
#  Sec 3.1 : 2DGS warm-up (photometric + depth distortion + normal consistency)
# =============================================================================

def training_2d(dataset, opt, pipe, args):
    tb_writer = prepare_output_and_logger(args)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    gaussians = SurfelModel(dataset.sh_degree)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    prepare_train_cameras(scene, args)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(opt.warmup_iterations), desc="2DGS warm-up")
    for iteration in range(1, opt.warmup_iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        if is_dtu(args):
            suppress_background(viewpoint_cam, gaussians, render_2dgs, pipe, background, "rend_alpha")

        render_pkg = render_2dgs(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        Ll1, loss = rgb_loss(image, gt_image, opt)

        # regularization (2DGS schedule)
        lambda_normal = opt.lambda_normal if iteration > 2500 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 1000 else 0.0
        normal_error = (1 - (render_pkg['rend_normal'] * render_pkg['surf_normal']).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * render_pkg["rend_dist"].mean()

        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}",
                                          "distort": f"{ema_dist_for_log:.{5}f}",
                                          "normal": f"{ema_normal_for_log:.{5}f}",
                                          "Points": f"{len(gaussians.get_xyz)}"})
                progress_bar.update(10)
            if iteration == opt.warmup_iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            if iteration == opt.warmup_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.warmup_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # the 2DGS checkpoint holds the Gaussians after the last densification step
            if iteration == opt.warmup_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))


# =============================================================================
#  Sec 3.2 : adaptive baseline selection from multi-view feature scores
# =============================================================================

@torch.no_grad()
def adaptive_baseline_selection(args, scene, gaussians, stereo, pipe, background, out_dir):
    """
    For every train view and candidate baseline B_i: render the stereo view with the warmed-up 3DGS,
    estimate D_i = f_psi(I^tr, I_i^st) and score it with S_i = sum cos(f(I_r, p_r), f(I_s, p_s)) (Eq. 4).
    B* = argmax_i S_i (Eq. 5).
    """
    cameras = scene.getTrainCameras()
    selection = {}
    log_lines = []
    for cam in cameras:
        candidates = []
        for stereo_cam in cam.stereo_cam_list:
            stereo_image = render(stereo_cam, gaussians, pipe, background)["render"]
            depth, consistent = stereo_depth(stereo, cam, stereo_image, stereo_cam.baseline, args, with_grad=False)

            pts = depth_to_world_points(cam, depth)
            valid = cam.stereo_object_mask.view(-1) & (depth.view(-1) > 0)
            score = float(multiview_feature_consistency(pts, cam, source_cameras(cam, cameras), valid, reduction='score'))
            ratio = float((cam.stereo_object_mask & consistent).sum() / cam.stereo_object_mask.sum().clamp(min=1))
            candidates.append((stereo_cam.baseline, score))

            line = f"{cam.image_name} baseline {stereo_cam.baseline:>5} : score {score:.2f}  lr-consistent ratio {ratio:.3f}"
            print(line)
            log_lines.append(line)

        selection[cam.image_name] = max(candidates, key=lambda c: c[1])[0]
        print(f"[Baseline selection] {cam.image_name} -> B* = {selection[cam.image_name]} % of radius")

    with open(os.path.join(out_dir, 'baseline_selection.log'), 'w') as f:
        f.write('\n'.join(log_lines) + '\n')
    with open(os.path.join(out_dir, 'baselines.json'), 'w') as f:
        json.dump(selection, f, indent=2)
    return selection


# =============================================================================
#  Sec 3.3 - 3.5 : fine-tuning with the stereo network + 2D/3D co-regularization
# =============================================================================

def training_finetune(dataset, opt, pipe, args, checkpoint_3d, checkpoint_2d):
    tb_writer = prepare_output_and_logger(args)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    gaussians_3d = GaussianModel(dataset.sh_degree)
    scene = Scene(args, gaussians_3d, shuffle=False)
    (model_params, warmup_iter) = torch.load(checkpoint_3d, weights_only=False)
    gaussians_3d.restore(model_params, opt)

    gaussians_2d = SurfelModel(dataset.sh_degree)
    (model_params, _) = torch.load(checkpoint_2d, weights_only=False)
    gaussians_2d.restore(model_params, opt)

    prepare_train_cameras(scene, args)
    cameras = scene.getTrainCameras()
    stereo = Stereo(args, device='cuda')

    baselines = adaptive_baseline_selection(args, scene, gaussians_3d, stereo, pipe, background, args.model_path)
    torch.cuda.empty_cache()
    stereo_cams = {cam.image_name: StereoCamera(cam, baselines[cam.image_name], scene.radius) for cam in cameras}

    first_iter = warmup_iter + 1
    last_iter = warmup_iter + opt.finetune_iterations
    progress_bar = tqdm(range(opt.finetune_iterations), desc="Fine-tuning")
    ema = {}
    viewpoint_stack = None
    stereo_cache = {}

    for iteration in range(first_iter, last_iter + 1):
        step = iteration - warmup_iter
        gaussians_3d.update_learning_rate(iteration)
        gaussians_2d.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        stereo_cam = stereo_cams[viewpoint_cam.image_name]
        gt_image = viewpoint_cam.original_image.cuda()

        if is_dtu(args):
            suppress_background(viewpoint_cam, gaussians_3d, render, pipe, background, "alpha")
            suppress_background(viewpoint_cam, gaussians_2d, render_2dgs, pipe, background, "rend_alpha")

        # train view and stereo view renderings of both branches
        pkg_3d = render(viewpoint_cam, gaussians_3d, pipe, background)
        pkg_3d_st = render(stereo_cam, gaussians_3d, pipe, background)
        pkg_2d = render_2dgs(viewpoint_cam, gaussians_2d, pipe, background)
        pkg_2d_st = render_2dgs(stereo_cam, gaussians_2d, pipe, background)

        # L_rgb
        _, rgb_loss_3d = rgb_loss(pkg_3d["render"], gt_image, opt)
        _, rgb_loss_2d = rgb_loss(pkg_2d["render"], gt_image, opt)

        # L_2d-reg: normal consistency + depth distortion (2DGS)
        normal_error = (1 - (pkg_2d['rend_normal'] * pkg_2d['surf_normal']).sum(dim=0))[None]
        reg_2d_loss = opt.lambda_normal * normal_error.mean() + opt.lambda_dist * pkg_2d["rend_dist"].mean()

        # L_co-img (Eq. 8): gradients to both branches
        I3D_st, I2D_st = pkg_3d_st["render"], pkg_2d_st["render"]
        co_img_loss = (1.0 - opt.lambda_dssim) * l1_loss(I3D_st, I2D_st) + opt.lambda_dssim * (1.0 - ssim(I3D_st, I2D_st))

        # L_co-depth (Eq. 9): L2 between the stereo-view depths, gradients only to 3DGS
        alpha3D_st = pkg_3d_st["alpha"]
        D3D_st = pkg_3d_st["depth"] / alpha3D_st.clamp(min=1e-6)
        D2D_st = pkg_2d_st["surf_depth"].detach()
        co_mask = ((pkg_2d_st["rend_alpha"] > 0.5) & (alpha3D_st > 0.5)).detach().float()
        co_depth_loss = l2_loss_mask(D3D_st, D2D_st, co_mask)
        co_loss = opt.beta_co_img * co_img_loss + opt.beta_co_depth * co_depth_loss

        # L_stereo (Eq. 6, 7): gradients through the frozen stereo network into the 3DGS stereo rendering.
        # The stereo network is the bottleneck, so it runs once every `stereo_interval` passes over the views;
        # in between, L_depth uses the cached stereo depth of the view.
        name = viewpoint_cam.image_name
        if name not in stereo_cache or (step - 1) // len(cameras) % opt.stereo_interval == 0:
            D_tr, consistent = stereo_depth(stereo, viewpoint_cam, I3D_st, stereo_cam.baseline, args, with_grad=True)
            valid = viewpoint_cam.stereo_object_mask & consistent & (D_tr.detach() > 0)
            pts = depth_to_world_points(viewpoint_cam, D_tr)
            stereo_loss = multiview_feature_consistency(pts, viewpoint_cam, source_cameras(viewpoint_cam, cameras),
                                                        valid.view(-1), reduction='mean')
            stereo_cache[name] = (D_tr.detach(), valid)
        else:
            D_tr, valid = stereo_cache[name]
            stereo_loss = torch.zeros((), device="cuda")

        # L_depth (Eq. 10): 2DGS train-view depth vs stereo depth, gradients only to 2DGS
        D2D_tr = F.interpolate(pkg_2d["surf_depth"][None], size=D_tr.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        depth_loss = l2_loss_mask(D2D_tr, D_tr.detach(), valid.float())

        # Eq. 11 (the branches have disjoint parameters, so one backward of the sum gives both gradients)
        loss_3dgs = rgb_loss_3d + co_loss + opt.beta_stereo * stereo_loss
        loss_2dgs = rgb_loss_2d + opt.beta_2dreg * reg_2d_loss + opt.beta_depth * depth_loss
        (loss_3dgs + loss_2dgs).backward()

        with torch.no_grad():
            logs = {"rgb3d": rgb_loss_3d, "rgb2d": rgb_loss_2d, "stereo": stereo_loss, "co_img": co_img_loss,
                    "co_depth": co_depth_loss, "depth": depth_loss, "reg2d": reg_2d_loss}
            for k, v in logs.items():
                ema[k] = 0.4 * float(v) + 0.6 * ema.get(k, float(v))
            if step % 10 == 0:
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in ema.items()})
                progress_bar.update(10)
            if tb_writer is not None:
                for k, v in logs.items():
                    tb_writer.add_scalar(f'finetune/{k}', float(v), iteration)

            gaussians_3d.optimizer.step()
            gaussians_3d.optimizer.zero_grad(set_to_none=True)
            gaussians_2d.optimizer.step()
            gaussians_2d.optimizer.zero_grad(set_to_none=True)
    progress_bar.close()

    gaussians_3d.save_ply(os.path.join(args.model_path, "point_cloud_3d", "iteration_{}".format(last_iter), "point_cloud.ply"))
    gaussians_2d.save_ply(os.path.join(args.model_path, "point_cloud_2d", "iteration_{}".format(last_iter), "point_cloud.ply"))
    torch.save((gaussians_3d.capture(), last_iter), os.path.join(args.model_path, "chkpnt_3d.pth"))
    torch.save((gaussians_2d.capture(), last_iter), os.path.join(args.model_path, "chkpnt_2d.pth"))
    print(f"\n[Fine-tuning] saved to {args.model_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Sparse-GS2Mesh training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_warmup_3d", action="store_true", help="reuse <model_path>/3d/chkpnt<warmup_iterations>.pth")
    parser.add_argument("--skip_warmup_2d", action="store_true", help="reuse <model_path>/2d/chkpnt<warmup_iterations>.pth")
    parser.add_argument("--skip_finetune", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    root = args.model_path
    checkpoint_3d = os.path.join(root, "3d", f"chkpnt{args.warmup_iterations}.pth")
    checkpoint_2d = os.path.join(root, "2d", f"chkpnt{args.warmup_iterations}.pth")

    if not (args.skip_warmup_3d and os.path.exists(checkpoint_3d)):
        args.model_path = os.path.join(root, "3d")
        training_3d(lp.extract(args), op.extract(args), pp.extract(args), args)
        torch.cuda.empty_cache()

    if not (args.skip_warmup_2d and os.path.exists(checkpoint_2d)):
        args.model_path = os.path.join(root, "2d")
        training_2d(lp.extract(args), op.extract(args), pp.extract(args), args)
        torch.cuda.empty_cache()

    if not args.skip_finetune:
        args.model_path = os.path.join(root, "final")
        training_finetune(lp.extract(args), op.extract(args), pp.extract(args), args, checkpoint_3d, checkpoint_2d)

    print("\nTraining complete.")
