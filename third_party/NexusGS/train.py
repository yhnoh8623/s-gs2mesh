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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import cv2
import torch
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, ssim, get_depth_ranking_loss
from utils.general_utils import get_shading
#from gaussian_renderer import render, network_gui
import img_gaussian_renderer
import geo_gaussian_renderer
import sys
#from scene import Scene, GaussianModel
import img_scene
import geo_scene
#from img_scene import cameras
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from utils.sh_utils import eval_sh
from utils.transformation_utils import get_shading
from utils.point_utils import depths_to_points, depth_to_normal
from utils.stereo_utils import Stereo
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.feat_utils import get_feat_loss
import torch.nn.functional as F

@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = img_gaussian_renderer.render(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts, 0)

def adaptive_baseline_selection(args, img_gaussians, im_scene, img_gaussian_renderer, stereo, pipe, background):
    scan = args.source_path.split('/')[-1]
    baseline_save = {}
    for cam in im_scene.getTrainCameras():
        cam.half_alpha_mask =(F.interpolate(cam.alpha_mask.float()[None][None], scale_factor=0.5, mode='bilinear', align_corners=False)[0][0] > 0.5).to('cuda')
        
        gt_image = cam.original_image*255
        baseline_save[cam.image_name] = {'baseline' : -1, 'feat_loss': -1}
        for stereo_cam in cam.stereo_cam_list:
            #print(f"Baseline for {cam.image_name} : {stereo_cam.baseline}")
            view = img_scene.Camera(0, stereo_cam.R, stereo_cam.T, None, cam.FoVx, cam.FoVy, torch.rand(3,cam.image_height,cam.image_width), None, "abcd", 0)
            #view = img_scene.Camera(0, cam.R, cam.T, None, cam.FoVx, cam.FoVy, torch.rand(3,cam.image_height,cam.image_width), None, "abcd", 0)
            
            render_pkg = img_gaussian_renderer.render(view, img_gaussians, pipe, background)
            stereo_image = render_pkg["render"]*255

            gt_half = F.interpolate(gt_image[None], scale_factor=0.5, mode='bilinear', align_corners=False)[0]
            stereo_half = F.interpolate(stereo_image[None], scale_factor=0.5, mode='bilinear', align_corners=False)[0]
            depth, occlusion_mask, depth_right, depth_occlusion_mask = stereo.run_single(gt_half[None], stereo_half[None], baseline_percentage = stereo_cam.baseline, \
                            fx = fov2focal(cam.FoVx, cam.image_width)/2, radius = cam.radius) # target shape : torch.Size([1, 3, 1162, 1554])
            #cv2.imwrite(f"baseline_test/{scan}/{cam.image_name}_{stereo_cam.baseline}.png", cv2.cvtColor((stereo_half.permute(1, 2, 0)).detach().cpu().numpy(), cv2.COLOR_BGR2RGB))
            #cv2.imwrite(f"baseline_test/{scan}/depth_{cam.image_name}_{stereo_cam.baseline}.png", get_shading(depth.detach().cpu().numpy(), 1e-4))
            #cv2.imwrite(f"baseline_test/{scan}/occlusion_{cam.image_name}_{stereo_cam.baseline}.png", (occlusion_mask*255).detach().cpu().numpy())
            
            surf_points = depths_to_points(cam, depth, resolution = 2)
            mask = (depth.view(-1) > 0)
            mask = mask & (cam.half_alpha_mask.view(-1) > 0.5)
            feat_loss = get_feat_loss(surf_points, cam, im_scene.getTrainCameras(), mask, resolution=1)
            
            occlusion_alpha_mask = cam.half_alpha_mask & occlusion_mask
            ratio = occlusion_alpha_mask.sum() / cam.half_alpha_mask.sum()
            
            print(f"{stereo_cam.baseline}  -->  {feat_loss}, {ratio}")
            
            if baseline_save[cam.image_name]['baseline'] == -1:
                baseline_save[cam.image_name]['baseline'] = stereo_cam.baseline
                baseline_save[cam.image_name]['feat_loss'] = feat_loss
            elif baseline_save[cam.image_name]['feat_loss'] >= feat_loss and ratio > 0.7:
                baseline_save[cam.image_name]['baseline'] = stereo_cam.baseline
                baseline_save[cam.image_name]['feat_loss'] = feat_loss
    
    
    for cam in im_scene.getTrainCameras():
        with open(f"baseline_test/{scan}/baseline.txt", 'a') as f:
            f.write(cam.image_name + ' ' + str(baseline_save[cam.image_name]['baseline']) + '\n')
    return baseline_save
            
            

def training_3d(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    viewpoint_stack, pseudo_stack = None, None

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0

    gaussians = img_scene.GaussianModel(args)

    scene = img_scene.Scene(args, gaussians, shuffle=False)

    gaussians.training_setup(opt)
    torch.cuda.empty_cache()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    ema_loss_for_log = 0.0
    first_iter += 1
    viewpoint_stack = None

    scan = args.source_path.split('/')[-1]
    print(f"Scan : {scan}")
    for cam in scene.getTrainCameras():
        flow_depth = cam.flow_depth
        os.makedirs(f'flow_depth/{scan}', exist_ok=True)
        cv2.imwrite(os.path.join(f'flow_depth/{scan}', cam.image_name + '.png'), get_shading(flow_depth.cpu().numpy(), 1e-4))
        np.save(os.path.join(f'flow_depth/{scan}', cam.image_name + '.npy'), flow_depth.cpu().numpy())

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = False

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        if args.dataset_type == 'dtu':
            if 'scan110' not in args.source_path :
                bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)

            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            gt_image[bg_mask.repeat(3,1,1)] = 0.

            render_pkg = img_gaussian_renderer.render(viewpoint_cam, gaussians, pipe, background)
            (render_pkg["alpha"][bg_mask]**2).mean().backward()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
        elif args.dataset_type == 'blender':
            bg_mask = (gt_image.min(0, keepdim=True).values > 254/255)

        render_pkg = img_gaussian_renderer.render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))


        # rendered_depth = render_pkg["depth"]
        # flow_depth = viewpoint_cam.flow_depth.cuda().unsqueeze(0)


        # if rendered_depth.shape[0] != 0 and iteration > 0 and opt.depth_weight > 0:
        #     midas_depth = viewpoint_cam.midas_depth.cuda().squeeze().unsqueeze(0)
        #     if args.dataset_type == 'dtu':

        #         flow_depth[bg_mask] = flow_depth[~bg_mask].mean()
        #         rendered_depth[bg_mask] = rendered_depth[~bg_mask].mean().detach()
        #     elif args.dataset_type == 'blender':
        #         flow_depth[bg_mask] = 0
        #         midas_depth[bg_mask] = 0

            #flow_depth = flow_depth.view(-1,1)
            #rendered_depth = rendered_depth.view(-1,1)

            #depth_loss = l1_loss_mask(flow_depth, rendered_depth)
            #loss += opt.depth_weight * depth_loss
        


        loss.backward(retain_graph=True)
        with torch.no_grad():
            # Progress bar
            if not loss.isnan():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Points": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            clean_iterations = testing_iterations + [first_iter]
            
            if args.dataset_type == 'dtu' or args.dataset_type == 'blender':
               clean_views(iteration, clean_iterations, scene, gaussians, pipe, background)
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, img_gaussian_renderer.render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter and iteration not in clean_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = opt.size_threshold
                    
                    if args.dataset_type == 'blender':
                        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                        dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                        color = torch.clamp_min(sh2rgb + 0.5, 0.0)
                        white_mask = color.min(-1, keepdim=True).values > 253/255
                        gaussians.xyz_gradient_accum[white_mask] = 0
                        gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(gaussians.opacity_activation(gaussians._opacity[white_mask]) * 0.1)

                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration, opt.dis_prune, opt.split_num)

                    if args.dataset_type == 'blender':
                        if 'ship' in args.source_path: 
                            gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.5, 0)
                        if 'hotdog' in args.source_path: 
                            gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.2, 0)      

                if iteration > opt.densify_from_iter and iteration % opt.prune_interval == 0 and opt.prune_interval != -1:
                    gaussians.prune(args, viewpoint_cam, rendered_depth, opt.prune_depth_threshold)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def training_2d(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = geo_scene.GaussianModel(dataset.sh_degree)
    scene = geo_scene.Scene(dataset, gaussians, shuffle = False)
    gaussians.training_setup(opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    scan = dataset.source_path.split('/')[-1]
    os.makedirs(f'2d_test/{scan}', exist_ok=True)
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = geo_gaussian_renderer.render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 2500 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 1000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()
        
        #print(render_pkg['surf_depth'].shape)
        if iteration%100 == 0:
            cv2.imwrite(os.path.join(f"2d_test/{scan}/{iteration}_depth.png"), get_shading(render_pkg['surf_depth'][0].detach().cpu().numpy(), 1e-4))

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

           # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, geo_gaussian_renderer.render, (pipe, background))
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, img_gaussian_renderer.render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training_mutual(dataset, opt, pipe, args, img_checkpoint, geo_checkpoint):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    viewpoint_stack, pseudo_stack = None, None
    
    stereo = Stereo(args, device="cuda")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0

    img_gaussians = img_scene.GaussianModel(args)
    im_scene = img_scene.Scene(args, img_gaussians, shuffle=False)
    
    if img_checkpoint:
        (model_params, first_iter) = torch.load(img_checkpoint)
        img_gaussians.restore(model_params, opt)
        
    geo_gaussians = geo_scene.GaussianModel(dataset.sh_degree)
    ge_scene = geo_scene.Scene(dataset, geo_gaussians, shuffle = False)
    
    if geo_checkpont:
        (model_params, first_iter) = torch.load(geo_checkpoint)
        geo_gaussians.restore(model_params, opt)
    
    img_gaussians.training_setup(opt)
    geo_gaussians.training_setup(opt)
    
        
    scan = args.source_path.split('/')[-1]
    os.makedirs(f"baseline_test/{scan}", exist_ok = True)
    stereo = Stereo(args, device='cuda')
     
    baseline_save = adaptive_baseline_selection(args, img_gaussians, im_scene, img_gaussian_renderer, stereo, pipe, background)
   
    for cam in im_scene.getTrainCameras():
        cam.gt_half = F.interpolate(cam.original_image[None], scale_factor=0.5, mode='bilinear', align_corners=False)[0]
        #cam.half_alpha_mask는 adaptive baseline selectioin에서 진행함.
        
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        #gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = im_scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        stereo_viewpoint_cam = viewpoint_cam.stereo_cam
        
        img_render_pkg = img_gaussian_renderer.render(viewpoint_cam, img_gaussians, pipe, background)
        img_image, img_depth, img_viewspace_point_tensor, img_visibility_filter, img_radii = img_render_pkg["render"], img_render_pkg["depth"], img_render_pkg["viewspace_points"], img_render_pkg["visibility_filter"], img_render_pkg["radii"]
        
        stereo_img_render_pkg = img_gaussian_renderer.render(stereo_viewpoint_cam, img_gaussians, pipe, background)
        stereo_img_image, stereo_img_depth = stereo_img_render_pkg["render"], stereo_img_render_pkg["depth"]
        stereo_img_image_half = F.interpolate(stereo_img_image[None], scale_factor=0.5, mode='bilinear', align_corners=False)[0]
    
        geo_render_pkg = geo_gaussian_renderer.render(viewpoint_cam, geo_gaussians, pipe, background)
        geo_image, geo_depth, geo_viewspace_point_tensor, geo_visibility_filter, geo_radii = geo_render_pkg["render"], geo_render_pkg["surf_depth"], geo_render_pkg["viewspace_points"], geo_render_pkg["visibility_filter"], geo_render_pkg["radii"]
        
        stereo_geo_render_pkg = geo_gaussian_renderer.render(stereo_viewpoint_cam, geo_gaussians, pipe, background)
        stereo_geo_image, stereo_geo_depth = stereo_geo_render_pkg["render"], stereo_geo_render_pkg["surf_depth"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        img_Ll1 = l1_loss(img_image, gt_image)
        img_loss = (1.0 - opt.lambda_dssim) * img_Ll1 + opt.lambda_dssim * (1.0 - ssim(img_image, gt_image))
        
        geo_Ll1 = l1_loss(geo_image, gt_image)
        geo_loss = (1.0 - opt.lambda_dssim) * gep_Ll1 + opt.lambda_dssim * (1.0 - ssim(geo_image, gt_image))
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = opt.lambda_normal * (normal_error).mean()
        dist_loss = opt.lambda_dist * (rend_dist).mean()
        geo_loss += (dist_loss + normal_loss)
        

        stereo_depth, occlusion_mask, stereo_depth_right, right_occlusion_mask = stereo.run_single_w_gradient((viewpoint_cam.gt_half*255)[None], stereo_img_image_half[None],
                            baseline_percentage = stereo_viewpoint_cam.baseline, \
                            fx = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width)/2, radius = viewpoint_cam.radius)
        
        total_loss.backward()

        iter_end.record()
        
        #print(render_pkg['surf_depth'].shape)
        if iteration%100 == 0:
            cv2.imwrite(os.path.join(f"2d_test/{scan}/{iteration}_depth.png"), get_shading(render_pkg['surf_depth'][0].detach().cpu().numpy(), 1e-4))

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

           # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, geo_gaussian_renderer.render, (pipe, background))
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, img_gaussian_renderer.render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    


           

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth = render_results["depth"]
                    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            #tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50_00, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.warmup_iterations)
    args.checkpoint_iterations.append(args.warmup_iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    args.iterations = args.warmup_iterations

    model_path_save = args.model_path
    
  
    args.model_path = os.path.join(model_path_save, "3d")
    #training_3d(lp.extract(args), op.extract(args), pp.extract(args), args)
    
    args.model_path = os.path.join(model_path_save, "2d")
    training_2d(lp.extract(args), op.extract(args), pp.extract(args), args)
    
    args.model_path = os.path.join(model_path_save, "final")
    img_checkpoint = os.path.join(model_path_save, "3d/chkpnt" + str(args.warmup_iterations) + ".pth")
    geo_checkpoint = os.path.join(model_path_save, "2d/chkpnt" + str(args.warmup_iterations) + ".pth")
    
    #print(img_checkpoint)
    training_mutual(lp.extract(args), op.extract(args), pp.extract(args), args, img_checkpoint, geo_checkpoint)

    # All done
    print("\nTraining complete.")