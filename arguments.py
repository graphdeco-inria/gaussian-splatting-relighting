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

from argparse import ArgumentParser, Namespace
import sys
import os
import datetime
from relighting.light_directions import *

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, _value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(_value)
            value = _value if not fill_none else None 
            if shorthand:
                if t == bool:
                    if _value:
                        group.add_argument("--no_" + key, ("-" + key[0:1]), default=value, action="store_false", dest=key)
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+", type=type(_value[0]))
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if _value:
                        group.add_argument("--no_" + key, default=value, action="store_false", dest=key)
                    else:
                        group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs="+", type=type(_value[0]))
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = [1536, 1024]
        self._white_background = False
        self.eval = False
        self.freeze_geometry = False
        self.label = ""
        self.halfres = False

        self.train_dirs = BACKWARD_DIR_IDS # backward dirs only
        self.camera_ids = [-1] # -1 alone means train on all views

        self.preview_dirs = [LEFT_DIR_ID, RIGHT_DIR_ID, TOP_DIR_ID, BACK_DIR_ID]

        self.light_vector_size = 64
        self.light_vector_lr = 0.001
        self.n_neurons = 128

        self.skip_images = 0
        self.max_images = 999999999

        self.num_encodings_view = 4
        self.num_encodings_light = 4 

        self.train_after_resume = False
        self.densify_after_resume = False

        self.num_warmup_iters = 5000
        self.num_feat_per_gaussian_channel = 16 

        self.use_key_views = True
        self.key_view_every_k_step = 3

        self.znear_pruning = True # if enabled and znear_fraction !=, prune the gaussians that move in front of any camera's near clipping plane
        self.znear_quantile = 0.01
        self.znear_scale = 0.9

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)

        g.preview_dirs = [ x for x in g.preview_dirs if x in g.train_dirs ]
        if g.preview_dirs == []:
            g.preview_dirs = g.train_dirs

        if g.halfres:
            g.resolution = [768, 512]
        
        return g
    
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.rand_background = False 

        self.load_iter = 0 # which itertion to load from, 0 means disabled
        
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000 

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01 
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.mlp_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_lpips = 0.0 # 0.1, 0.05 seem to converge
        self.densify_from_iter = 500
        self.densify_grad_threshold = 0.0002

        self.posititon_lr_max_steps = 30_000
        self.densification_interval = 100 
        self.opacity_reset_interval = 3000 
        self.densify_until_iter = 15_000 

        self.viewer = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
