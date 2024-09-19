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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F
import math 
import shutil
from relighting.light_directions import *

def save_image(image, *args, **kwargs):
    return torch.utils.save_image(F.avg_pool2d(image, 2), *args, **kwargs, padding=0)

def imsave(*args, **kwargs):
    return torchvision.utils.save_image(*args, **kwargs, padding=0)

def render_set(args, model_path, name, iteration, views, gaussians, pipeline, background):
    futures = []

    with ThreadPoolExecutor() as executor:
        if args.sweep:
            render_path = os.path.join(model_path, name, "sweep_frames")
            
            if os.path.exists(render_path):
                shutil.rmtree(render_path)
            makedirs(render_path)

            def write_images(idx, rendering):
                imsave(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            i = 0
            for j in range(100):
                t = j / (100 - 1)

                from relighting.light_directions import polar_angles_by_id
                theta_1 = polar_angles_by_id[6]["theta"]
                phi_1 = polar_angles_by_id[6]["phi"]
                theta_2 = polar_angles_by_id[7]["theta"]
                phi_2 = polar_angles_by_id[7]["phi"]

                theta = theta_1 

                phi = (1.0 - t) * phi_1 + t * phi_2
                
                x_coord = math.sin(theta) * math.cos(phi)
                y_coord = math.sin(theta) * math.sin(phi)
                z_coord = math.cos(theta)
                light_vec = torch.tensor([x_coord, y_coord, z_coord], device="cuda")

                preds1 = render(views[0], gaussians, pipeline, background, light_vec=light_vec, override_view_id="mean")
                preds2 = render(views[-1], gaussians, pipeline, background,  light_vec=light_vec, override_view_id="mean")
                rendering = torch.cat([preds1["render"], preds2["render"]], dim=-2)
                executor.submit(write_images, i, rendering)
                i += 1
        else:
            render_path = os.path.join(model_path, name, "eval_frames" if not args.video else "video_frames")
            if os.path.exists(render_path):
                shutil.rmtree(render_path)
            makedirs(render_path)

            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if not args.video and (idx % 10 != 0):
                    continue 

                def write_images(idx, view, rendering, light_id=None):
                    imsave(rendering, os.path.join(render_path, '{0:05d}_dir_{1:02d}'.format(idx, light_id) + ".png"))
            
                for k in BACKWARD_DIR_IDS if not args.video else [LEFT_DIR_ID, RIGHT_DIR_ID, TOP_DIR_ID, BACK_DIR_ID]:
                    preds = render(view, gaussians, pipeline, background, light_id=k, override_view_id="mean", reference_camera_pose=views[0])
                    futures.append(
                        executor.submit(write_images, idx, view, preds["render"].cpu(), light_id=k))
                    
            for future in tqdm(as_completed(futures), "Saving renders to disk", total=len(views)):
                future.result()

def render_sets(args, modelParams: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(modelParams, pipeline)

        bg_color = [1,1,1] if modelParams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        scene = Scene(modelParams, pipeline, gaussians, load_iteration=iteration, shuffle=False, adjust_znear=False, split="test")
        cameras = scene.getTrainCameras() # note that this is a retrofitted hack; these are in fact test cameras
        
        render_set(args, modelParams.model_path, "test", scene.loaded_iter, cameras, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--video", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    args.train_dirs = args.preview_dirs

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    model_args = model.extract(args)
    model_args.max_images = None
    model_args.resume = False

    model_args.skip_loading_relit_images = True

    model_args.train_dirs = BACKWARD_DIR_IDS
    model_args.preview_dirs = BACKWARD_DIR_IDS
    
    render_sets(args, model_args, args.iteration, pipeline.extract(args))

    