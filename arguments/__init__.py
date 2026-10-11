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

import os
from argparse import ArgumentParser


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = key.startswith("_")
            if shorthand:
                key = key[1:]
            flags = ["--" + key] + (["-" + key[0:1]] if shorthand else [])
            t = type(value)
            if t == bool:
                group.add_argument(*flags, default=value, action="store_true")
            elif t == list:
                group.add_argument(*flags, default=value, nargs="+", type=type(value[0]) if len(value) > 0 else str)
            else:
                group.add_argument(*flags, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for key, value in vars(args).items():
            if key in vars(self) or ("_" + key) in vars(self):
                setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.dataset_type = "dtu"
        self.n_views = 3

        # epipolar depth initialization (NexusGS)
        self.valid_dis_threshold = 1.0
        self.near_n = 2
        self.drop_rate = 1.0

        # stereo matching network f_psi (DEFOM-Stereo)
        self.stereo_ckpt = ""                   # default: third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth
        self.stereo_resolution = 0.5            # input scale of the stereo network during training
        self.stereo_occlusion_threshold = 3.0   # left-right consistency threshold (pixels, at stereo resolution)
        self.stereo_valid_iters = 32            # GRU iterations without gradients
        self.stereo_train_iters = 32            # GRU iterations when backpropagating through the stereo network
        self.stereo_scale_iters = 8
        super().__init__(parser, "Loading Parameters")

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        self.depth_ratio = 1.0                  # 2DGS surface depth: 1 = median depth (bounded scenes), 0 = expected depth
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # warm-up (Sec 3.1)
        self.warmup_iterations = 7000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.03
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 10_000
        self.densify_grad_threshold = 0.0005
        self.opacity_reset_interval = 3000
        self.opacity_reset_from_iter = 2000     # 3DGS: opacity is reset at iterations opacity_reset_from_iter + k * interval + 1
        self.prune_threshold = 0.005            # 3DGS opacity pruning
        self.size_threshold = -1                # 3DGS screen-size pruning (-1: off)
        self.split_num = 4                      # 3DGS: number of Gaussians a split produces
        self.opacity_cull = 0.05                # 2DGS opacity pruning
        self.lambda_dist = 1000.0               # 2DGS depth distortion
        self.lambda_normal = 0.05               # 2DGS normal consistency

        # adaptive baseline selection (Sec 3.2): candidates in % of the scene radius
        self.renderer_baseline_candidate = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # fine-tuning (Sec 3.3 - 3.5)
        self.finetune_iterations = 1000
        self.stereo_interval = 4                # run the stereo network (L_stereo) once every N passes over the views
        self.beta_stereo = 1.0                  # beta1: L_stereo   (3DGS)
        self.beta_2dreg = 1.0                   # beta2: L_2d-reg   (2DGS)
        self.beta_depth = 1.0                   # beta3: L_depth    (2DGS)
        self.beta_co_img = 1.0                  # beta4: L_co-img   (3DGS and 2DGS)
        self.beta_co_depth = 1.0                # beta5: L_co-depth (3DGS)
        super().__init__(parser, "Optimization Parameters")
