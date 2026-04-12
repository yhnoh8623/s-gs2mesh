# =============================================================================
#  Imports
# =============================================================================

import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
from PIL import Image
from argparse import Namespace
import matplotlib.pyplot as plt
from utils.transformation_utils import get_shading

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..', 'DEFOM-Stereo')))
from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder
import glob
import shutil

from pathlib import Path
# =============================================================================
#  Class for stereo matching model
# =============================================================================

class Stereo:
    def __init__(self, args, device='cuda'):
        """
        Initialize the Stereo class.

        Parameters:
        base_dir (str): Base directory of the repository.
        renderer (Renderer): Renderer class object.
        args (ArgParser): Program arguments.
        device (str): Device to run the model on.
        """
        self.dataset_name= args.dataset_type
        if self.dataset_name == 'blendedMVS':
            self.image_list = args.image_list
        
        #self.renderer = renderer
        self.args = args
        self.model_name = "Defom"
        self.device = device
        self.model_args = Namespace(valid_iters=32,
                                    scale_iters = 8)
       
        args.restore_ckpt = '../../third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth'
        args.dinov2_encoder = 'vitl' #vits
        args.idepth_scale = 0.5
        args.hidden_dims = [128]*3
        args.corr_implementation = 'reg'
        args.corr_levels = 2
        args.corr_radius = 4
        args.scale_list = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        args.scale_corr_radius = 2
        args.n_downsample = 2
        args.context_norm = "batch"
        args.n_gru_layers = 3
        args.mixed_precision = False
        self.model = DEFOMStereo(args)
        checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
        if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self.model = torch.nn.DataParallel(self.model)


    def load_image(self, imfile):
        """
        Load an image and prepare it for the stereo model.

        Parameters:
        imfile (str): Path to the image file.

        Returns:
        torch.Tensor: Image tensor prepared for the model.
        """
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)
    
    def run_single(self, image1, image2, start=0, visualize=False, resolution = 1):
        disparities = {'LR': None, 'RL': None}
        padder = InputPadder(image1.shape, divis_by=32)
        
        for direction in ['LR', 'RL']:
        #print('shape')
            #print(image1.shape)
            if direction == 'LR':
                image1, image2 = padder.pad(image1, image2)

            if direction == 'LR':
                image1_to_model = image1
                image2_to_model = image2
            elif direction == 'RL':
                image1_to_model = torch.flip(image2, dims=[3])
                image2_to_model = torch.flip(image1, dims=[3])
            
            disp_pr = self.model(image1_to_model, image2_to_model, iters = self.model_args.valid_iters, scale_iters = self.model_args.scale_iters, test_mode=True)
            #prev_flow, flow_up = self.model(image1_to_model, image2_to_model, iters=self.model_args.valid_iters, test_mode=True, flow_init=prev_flows[direction] if self.args.stereo_warm else None)
            if direction == 'RL':
                disp_pr = torch.flip(disp_pr, dims=[3])
                
            disparities[direction] = padder.unpad(disp_pr).detach().cpu().squeeze().numpy()
            #disparities[direction] = padder.unpad(disp_pr).squeeze()

            
            output_directory = os.path.join(output_dir, f'out_{self.model_name}')
            os.makedirs(output_directory, exist_ok=True)
            
            np.save(os.path.join(output_directory, f"disparity_{direction}.npy"), disparities[direction])
            plt.imsave(os.path.join(output_directory, f"disparity_{direction}.png"), disparities[direction], cmap='jet')
        
        occlusion_mask, disparity_difference = self.get_occlusion_mask(disparities['LR'], disparities['RL'], self.args.stereo_occlusion_threshold, baseline_percentage = left_camera['baseline_percentage'])
        right_occlusion_mask = self.get_right_occlusion_mask(disparities['LR'], disparities['RL'], self.args.stereo_occlusion_threshold)
        depth = (left_camera['fx'] * baseline) / (disparities['LR'])
        depth_right = (left_camera['fx'] * baseline) / (disparities['RL'])
        
        return depth, occlusion_mask
   
    def run(self, start=0, visualize=False, resolution = 1, use_GT_for_stereo_input = True):
        """
        Run the stereo model: render a pair of images and compute the disparity and depth using the stereo model.

        Parameters:
        start (int): The view number from which to start. Default is 0.
        visualize (bool): Flag to visualize the results for debugging.

        Returns:
        None
        """
        #print('run????')
        with torch.no_grad():
            prev_flows = {'LR': None, 'RL': None}
            disparity_txt = ""
            #test_mask = np.load('test_mask.npy')
            for camera_number, left_camera in enumerate(tqdm(self.renderer.left_cameras)):
                if camera_number < start:
                    continue
                baseline = left_camera['baseline']
                baseline_percentage = left_camera['baseline_percentage']
                self.args.stereo_occlusion_threshold = 0.5 * baseline_percentage
                output_dir = self.renderer.render_folder_name(left_camera['camera_number'])
                self.renderer.render_image_pair(camera_number, visualize=False)
            #     scan_num = output_dir.split('/')[-2]
            #     image_num = int(output_dir.split('/')[-1][:3])
            #     print(f'outputdir : {output_dir}')
            #     try:                
            #         if self.dataset_name == 'DTU':
            #             original_image_path = f"data/DTU/{scan_num}/images/{image_num:04d}.png"
            #             original_image = self.load_image(original_image_path)[: , :3, : , :]
            #         elif self.dataset_name == 'blendedMVS':
            #             image_num_bmvs = self.image_list[scan_num][image_num]
            #             original_image_path = f"data/blendedMVS/{scan_num}/image/{image_num_bmvs:03d}.png"
            #             original_image = self.load_image(original_image_path)[: , :3, : , :]
            #         elif self.dataset_name == 'custom':
            #             image_name = left_camera['image_name']
            #             original_image_path = f"data/custom/{scan_num}/images/{image_name}"
            #             # shutil.copy(original_image_path, os.path.join(output_dir, 'left.png'))
            #             # continue
            #             original_image = self.load_image(original_image_path)[: , :3, : , :]
                       
            #     except Exception as e:
            #         pass
           
            #     image1 = self.load_image(os.path.join(output_dir, 'left.png'))
            #     image2 = self.load_image(os.path.join(output_dir, 'right.png'))

            #     # ==========================================================
            #     # for GT Image
            #     # ==========================================================
            #     print(f"use_GT_for_stereo_input : {use_GT_for_stereo_input}")
            #     if use_GT_for_stereo_input:
            #         try:
            #             print('using original image')
            #             image1 = original_image
            #         except:
            #             print('using rendered image')
            #             pass
            #     # ==========================================================
            #     disparities = {'LR': None, 'RL': None}
            #     padder = InputPadder(image1.shape, divis_by=32)
               
            #     for direction in ['LR', 'RL']:
            #     #print('shape')
            #         #print(image1.shape)
            #         if direction == 'LR':
            #             image1, image2 = padder.pad(image1, image2)

            #         if direction == 'LR':
            #             image1_to_model = image1
            #             image2_to_model = image2
            #         elif direction == 'RL':
            #             image1_to_model = torch.flip(image2, dims=[3])
            #             image2_to_model = torch.flip(image1, dims=[3])
                   
            #         disp_pr = self.model(image1_to_model, image2_to_model, iters = self.model_args.valid_iters, scale_iters = self.model_args.scale_iters, test_mode=True)
            #         #prev_flow, flow_up = self.model(image1_to_model, image2_to_model, iters=self.model_args.valid_iters, test_mode=True, flow_init=prev_flows[direction] if self.args.stereo_warm else None)
            #         if direction == 'RL':
            #             disp_pr = torch.flip(disp_pr, dims=[3])
                       
            #         disparities[direction] = padder.unpad(disp_pr).detach().cpu().squeeze().numpy()
            #         #disparities[direction] = padder.unpad(disp_pr).squeeze()

                   
            #         output_directory = os.path.join(output_dir, f'out_{self.model_name}')
            #         os.makedirs(output_directory, exist_ok=True)
                   
            #         np.save(os.path.join(output_directory, f"disparity_{direction}.npy"), disparities[direction])
            #         plt.imsave(os.path.join(output_directory, f"disparity_{direction}.png"), disparities[direction], cmap='jet')
               
            #     occlusion_mask, disparity_difference = self.get_occlusion_mask(disparities['LR'], disparities['RL'], self.args.stereo_occlusion_threshold, baseline_percentage = left_camera['baseline_percentage'])
            #     right_occlusion_mask = self.get_right_occlusion_mask(disparities['LR'], disparities['RL'], self.args.stereo_occlusion_threshold)
            #     depth = (left_camera['fx'] * baseline) / (disparities['LR'])
            #     depth_right = (left_camera['fx'] * baseline) / (disparities['RL'])
            #     masked_depth = depth * occlusion_mask
            #     #for depth ensemble pixel by pixel
            #     #disparity_difference[~occlusion_mask] = float('inf') # for depth ensemble pixel by pixel
            #     try:
            #         # for depth ensemble baseline by baseline
            #         object_mask = np.load(os.path.join(output_dir, 'left_mask.npy'))
            #         final_mask = occlusion_mask.astype(bool) & object_mask.astype(bool)
            #         disparity_difference = disparity_difference * final_mask
            #         disparity_LR = disparities['LR'].copy()
            #         disparity_difference = disparity_difference / disparity_LR
            #         #disparity_difference = disparity_difference / baseline

            #         #disparity_error = np.mean(disparity_difference) / left_camera['baseline']
            #         disparity_error = np.sum(disparity_difference) / np.count_nonzero(disparity_difference)
            #         ratio = np.count_nonzero(disparity_difference) / np.sum(object_mask) * 100
            #         print(f"ratio : {ratio}")

            #         # test by small occlusion amsk?
            #         # object_mask = np.load(os.path.join(output_dir, 'left_mask.npy'))
            #         # #final_mask = test_mask.astype(bool) & object_mask.astype(bool)
            #         # final_mask = occlusion_mask.astype(bool) & object_mask.astype(bool)
            #         # disparity_difference = disparity_difference * final_mask
            #         # #disparity_LR = disparities['LR'].copy()
            #         # #disparity_difference = disparity_difference / disparities['LR']
            #         # #disparity_error = np.mean(disparity_difference)
            #         # disparity_error = np.mean(disparity_difference) / left_camera['baseline']

            #         print(f"{left_camera['camera_number']} {left_camera['baseline_percentage']} : {disparity_error*1000}")
            #         print(np.count_nonzero(disparity_difference))
            #         disparity_txt += f"{left_camera['camera_number']} {left_camera['baseline_percentage']} {disparity_error*1000} {ratio}\n"
            #     except:
            #         pass


            #     #print(f'disparity_difference shape : {disparity_difference.shape}') #(1162, 1154)
            #     #print(disparity_difference)
            #     #print('depth')
            #     #print(np.max(depth))
            #     #print(f"mean = {np.mean(depth)}")
            #     np.save(os.path.join(output_directory, "occlusion_mask.npy"), occlusion_mask)
            #     plt.imsave(os.path.join(output_directory, "occlusion_mask.png"), occlusion_mask)
            #     np.save(os.path.join(output_directory, "right_occlusion_mask.npy"), right_occlusion_mask)
            #     plt.imsave(os.path.join(output_directory, "right_occlusion_mask.png"), right_occlusion_mask)
               
            #     #np.save(os.path.join(output_directory, "disparity_difference.npy"), disparity_difference)
            #     np.save(os.path.join(output_directory, "depth.npy"), depth)
            #     cv2.imwrite(os.path.join(output_directory, 'depth.png'), depth)            
            #     shading = get_shading(depth, self.args.stereo_shading_eps)
            #     shading_right = get_shading(depth_right, self.args.stereo_shading_eps)
            #     #masked_shading = get_shading(masked_depth, self.args.stereo_shading_eps)
            #     cv2.imwrite(os.path.join(output_directory, 'shading.png'), shading)
            #     #cv2.imwrite(os.path.join(output_directory, 'shading_masked.png'), masked_shading)

            #     cv2.imwrite(os.path.join(output_directory, 'shading_right.png'), shading_right)
               
            #     if visualize:
            #         print(f"baseline: {baseline}")
            #         print(f"minimal depth: {depth.min()}, maximal depth: {depth.max()}")
            #         self.view_results_single(camera_number)

            #     torch.cuda.empty_cache()
            # with open('disparity.txt', 'w', encoding='utf-8') as f:
            #     f.write(disparity_txt)
               
    def get_occlusion_mask(self, L2R_disparity, R2L_disparity, occlusion_threshold, baseline_percentage = 5):
        """
        Calculate the occlusion mask given a pair of disparities.

        Parameters:
        L2R_disparity (np.ndarray): Left-to-right disparity map.
        R2L_disparity (np.ndarray): Right-to-left disparity map.
        occlusion_threshold (int): Threshold on the reprojection error.

        Returns:
        np.ndarray: Binary occlusion mask where 0 indicates occluded pixels and 1 indicates visible pixels.
        """

        #print(f"occlusion_threshold : {occlusion_threshold}")
       
        height, width = L2R_disparity.shape
   
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
       
        x_projected = (x_grid - L2R_disparity).astype(np.int32)
        x_projected_clipped = np.clip(x_projected, 0, width - 1)
       
        x_reprojected = x_projected_clipped + R2L_disparity[y_grid, x_projected_clipped]
        x_reprojected_clipped = np.clip(x_reprojected, 0, width - 1)
       
        disparity_difference = np.abs(x_grid - x_reprojected_clipped)
   
        #new_occlusion_threshold = (occlusion_threshold / 2 * baseline_percentage)
        print(f"{baseline_percentage} new occlusion threshold : {occlusion_threshold}")
        occlusion_mask = (disparity_difference > occlusion_threshold).astype(np.uint8)
       
        occlusion_mask[(x_projected < 0) | (x_projected >= width)] = 1

        occlusion_mask = occlusion_mask > 0.5
       
        return ~occlusion_mask, disparity_difference

    def get_right_occlusion_mask(self, L2R_disparity, R2L_disparity, occlusion_threshold):
        """
        Calculate the occlusion mask given a pair of disparities.

        Parameters:
        L2R_disparity (np.ndarray): Left-to-right disparity map.
        R2L_disparity (np.ndarray): Right-to-left disparity map.
        occlusion_threshold (int): Threshold on the reprojection error.

        Returns:
        np.ndarray: Binary occlusion mask where 0 indicates occluded pixels and 1 indicates visible pixels.
        """
        height, width = L2R_disparity.shape
   
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
       
        x_projected = (x_grid + R2L_disparity).astype(np.int32)
        x_projected_clipped = np.clip(x_projected, 0, width - 1)
       
        x_reprojected = x_projected_clipped - L2R_disparity[y_grid, x_projected_clipped]
        x_reprojected_clipped = np.clip(x_reprojected, 0, width - 1)
       
        disparity_difference = np.abs(x_grid - x_reprojected_clipped)
   
        occlusion_mask = (disparity_difference > occlusion_threshold).astype(np.uint8)
       
        occlusion_mask[(x_projected < 0) | (x_projected >= width)] = 1

        occlusion_mask = occlusion_mask > 0.5
       
        return ~occlusion_mask



    def view_results_single(self, camera_number):
        """
        Visualize the rendering and stereo results for a single view.

        Parameters:
        camera_number (int): View number to visualize.

        Returns:
        None
        """
        output_dir = self.renderer.render_folder_name(camera_number)

        paths = {'left_img': 'left.png',
                 'right_img': 'right.png',
                 'object_mask': 'left_mask.png',
                 'occlusion_mask': f'out_{self.model_name}/occlusion_mask.png',
                 'disparity': f'out_{self.model_name}/disparity_LR.png',
                 'shading': f'out_{self.model_name}/shading.png'
                }

        images = {path_name: None for path_name in paths.keys()}

        for path_name, filename in paths.items():
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                images[path_name] = Image.open(path)
            else:
                images[path_name] = Image.fromarray(np.random.randint(0, 255, (self.renderer.left_cameras[0]['height'], self.renderer.left_cameras[0]['width'], 3), dtype=np.uint8))

        images['lr_img'] = Image.blend(images['left_img'], images['right_img'], alpha=0.5)
       
        images_list = [images[key] for key in ['lr_img', 'object_mask', 'disparity', 'occlusion_mask', 'shading']]
       
        widths, heights = zip(*(i.size for i in images_list))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        disparity_min, disparity_max = None, None
        for i, im in enumerate(images_list):
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
            if i == 2:
                disparity_min = np.array(im).min()
                disparity_max = np.array(im).max()
        concatenated_image = new_im
       
        factor = 0.25
        smaller_image = concatenated_image.resize((int(total_width * factor), int(max_height * factor)))

        plt.figure(figsize=(20, 10))
        plt.imshow(smaller_image)
        plt.title(f'{camera_number:03}')
        plt.axis('off')
        plt.show()
        print(f"minimal disparity: {disparity_min}, maximal disparity: {disparity_max}")

    def view_results(self):
        """
        Visualize the rendering and stereo results for all views.

        Returns:
        None
        """
        for camera_number in range(len(self.renderer)):
            self.view_results_single(camera_number)
