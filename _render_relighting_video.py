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
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import math 
import shutil
import numpy as np
from relighting.gray_ball_renderer import GrayBallRenderer
from relighting.light_directions import polar_angles_by_id

def imsave(*args, **kwargs):
    return torchvision.utils.save_image(*args, **kwargs, padding=0)

def render_set(args, model_path, name, iteration, views, gaussians, pipeline, background, scene):
    gray_ball_renderer = GrayBallRenderer(150)

    render_path = os.path.join(model_path, name, f"renders")
    if os.path.exists(render_path):
        shutil.rmtree(render_path)
    makedirs(render_path)

    video_config = eval(open(args.source_path + "/train/video_config.py", "r").read())
    config = video_config["views"][args.view_number]
    view = [x for x in views if int(x.image_name) == config["camera_id"]][0]

    camera_centers = [v.camera_center.cpu().numpy() for v in views]
    centroid = np.mean(camera_centers, axis=0)
    scene_radius = np.max(np.sqrt(np.sum((camera_centers - centroid) ** 2, axis=1))) / 7 # get a very rough estimate of the scene size, divided by 7 for no good reason

    c2w_rot_colmap = torch.from_numpy(view.R).cuda().float()
    c2w_rot_blender = -c2w_rot_colmap
    c2w_rot_blender[:, 0] = -c2w_rot_blender[:, 0]

    def get_rotmatrix(rot_xyz):
        rot_mat = np.eye(3)
        rot_mat = rot_mat @ np.array([[1, 0, 0], [0, np.cos(rot_xyz[0]), -np.sin(rot_xyz[0])], [0, np.sin(rot_xyz[0]), np.cos(rot_xyz[0])]])
        rot_mat = rot_mat @ np.array([[np.cos(rot_xyz[1]), 0, np.sin(rot_xyz[1])], [0, 1, 0], [-np.sin(rot_xyz[1]), 0, np.cos(rot_xyz[1])]])
        rot_mat = rot_mat @ np.array([[np.cos(rot_xyz[2]), -np.sin(rot_xyz[2]), 0], [np.sin(rot_xyz[2]), np.cos(rot_xyz[2]), 0], [0, 0, 1]])
        return rot_mat

    if args.camera_mode != "static":
        view.T += np.array(config["init_translation"])
        rot_xyz = np.array(config["init_rotation"])
        
        c2w_rot_blender = c2w_rot_blender @ torch.from_numpy(get_rotmatrix(rot_xyz)).cuda().float()
        rotated_c2w_colmap = c2w_rot_blender.clone()
        rotated_c2w_colmap[:, 0] = -rotated_c2w_colmap[:, 0]
        rotated_c2w_colmap = -rotated_c2w_colmap
        view.R = rotated_c2w_colmap.cpu().numpy()
        
    if args.camera_mode in ["ken_burns"]:
        total_translation = np.array(config["ken_burns"]) * scene_radius 
        view.T -= total_translation / 2
    view.update()

    Ts = []
    c2ws = []
    init_c2w = c2w_rot_blender.cpu().clone()
    init_o = view.camera_center.cpu().clone()

    scene.autoadjust_znear()
    view.znear *= 0.7 # in case things get too close due to camera motion

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in tqdm(range(args.num_frames)):
            if args.camera_mode == "orbit":
                orbit_distance = config["orbit"]["distance"]
                center_of_focus = init_o - orbit_distance * init_c2w[:, 2] 
                def rotate_camera_around_point(c2w_rot, o, p, theta, axis):
                    # Axis-angle to rotation matrix
                    axis = axis / torch.linalg.norm(axis)
                    a = torch.cos(theta / 2)
                    b, c, d = -axis * torch.sin(theta / 2)
                    aa, bb, cc, dd = a * a, b * b, c * c, d * d
                    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
                    R = torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
                    # Translate camera so rotation point is at origin, apply rotation, translate back
                    rotated_c2w_rot = R @ c2w_rot
                    rotated_o = R @ (o - p) + p
                    return rotated_c2w_rot, rotated_o
                y_axis = init_c2w[:, {"x":0, "y":1}[config["orbit"]["axis"]]]
                # t = math.sin(math.pi * torch.tensor(i / (args.num_frames-1) * 2))
                t = math.sin(6 * math.pi * i / (args.num_frames-1)) / 2 + 0.5
                theta = torch.lerp(torch.tensor(config["orbit"]["radians"]/2), torch.tensor(-config["orbit"]["radians"]/2), t)                
                rotated_c2w, rotated_o = rotate_camera_around_point(init_c2w, init_o, center_of_focus, theta, y_axis.cpu())
                rotated_c2w_colmap = rotated_c2w.clone()
                rotated_c2w_colmap[:, 0] = -rotated_c2w_colmap[:, 0]
                rotated_c2w_colmap = -rotated_c2w_colmap
                view.R = rotated_c2w_colmap.cpu().numpy()
                view.T = -view.R.T @ rotated_o.cpu().numpy() #+ total_translation * i / (args.num_frames - 1)
                c2ws.append(rotated_c2w_colmap)
                Ts.append(rotated_o.cpu().numpy())
                view.update()
            
            if args.camera_mode in ["ken_burns"]:
                view.T += total_translation / args.num_frames
                view.update()
            
            if args.static_light:
                theta = polar_angles_by_id[args.static_light_id]["theta"]
                phi = polar_angles_by_id[args.static_light_id]["phi"]

            elif args.camera_mode == "orbit":
                if i < args.num_frames // 3:
                    light_id = 23
                elif i < 2 * args.num_frames // 3:
                    light_id = 18 
                else:
                    light_id = 14

                theta = polar_angles_by_id[light_id]["theta"]
                phi = polar_angles_by_id[light_id]["phi"]
            else:
                if i < args.num_frames // 2:
                    # left-right
                    t = (i % (args.num_frames//2)) / (args.num_frames//2 - 1)
                    w = math.sin(math.pi * 2 * t) / 2 + 0.5
                    phi = -math.pi * (1.0 - w) + 0 * w
                    theta = math.pi/2
                else:
                    # circles
                    t = (i % (args.num_frames//2)) / (args.num_frames//2 - 1)
                    t = (0.25 + t) % 1.0
                    w = math.cos(math.pi * 2 * (1.0 - t)) / 2 + 0.5
                    phi = 0 * (1.0 - w) + -math.pi * w
                    w = -math.cos(math.pi * 2 * t*2) / 2 + 0.5
                    theta = 0 * (1.0 - w) + math.pi/2 * w

                theta_range = (0.4, 1.35)
                phi_range = (-2.3, -0.3)

                theta_init_range = (0.0, math.pi/2)
                theta_scale = (theta_range[1] - theta_range[0]) / (theta_init_range[1] - theta_init_range[0])
                theta_shift = theta_range[0]
                theta = theta_scale * (theta - theta_init_range[0]) + theta_shift

                phi_init_range = (-math.pi, 0)
                phi_scale = (phi_range[1] - phi_range[0]) / (phi_init_range[1] - phi_init_range[0])
                phi_shift = phi_range[0]
                phi = phi_scale * (phi - phi_init_range[0]) + phi_shift
            
            x_coord = math.sin(theta) * math.cos(phi)
            y_coord = math.sin(theta) * math.sin(phi)
            z_coord = math.cos(theta)
        
            light_vec = c2w_rot_blender @ torch.tensor([x_coord, z_coord, -y_coord], device="cuda")
            preds = render(view, gaussians, pipeline, background, light_vec=light_vec, override_view_id="mean")
            frame = gray_ball_renderer.render_onto(preds["render"][None].cpu(), torch.tensor([x_coord, z_coord, -y_coord]))[0]
            executor.submit(imsave, *(frame**video_config["gamma"]*video_config["exposure"], os.path.join(render_path, '{0:05d}'.format(i) + ".png")))
        print("Flushing images to disk...")

def render_sets(args, modelParams: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(modelParams, pipeline)
        
        bg_color = [1,1,1] if modelParams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            scene = Scene(modelParams, pipeline, gaussians, load_iteration=iteration, shuffle=False, adjust_znear=True)
            cameras = scene.getTrainCameras()
            print("Rendering train...")
            render_set(args, modelParams.model_path, "train", scene.loaded_iter, cameras, gaussians, pipeline, background, scene)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--legacy", action="store_true")

    parser.add_argument("--static_light", action="store_true")
    parser.add_argument("--secondary_view", action="store_true")

    parser.add_argument("--camera_mode", type=str, choices=["static", "ken_burns", "orbit"], required=True)

    parser.add_argument("--relighting", action="store_true")
    parser.add_argument("--move_while_relighting", action="store_true")
    parser.add_argument("--arc", action="store_true")
    parser.add_argument("--view_number", default=0, type=int)
    parser.add_argument("--static_light_id", default=23, type=int)
    parser.add_argument("--num_frames", default=270, type=int)
    
    args = get_combined_args(parser)
    args.skip_test = True 
    print("Rendering " + args.model_path)

    args.train_dirs = args.preview_dirs
    # Initialize system state (RNG)
    safe_state(args.quiet)

    modelParams = model.extract(args)
    # modelParams.max_images = None
    modelParams.resume = False
    modelParams.train_dirs = list(range(25))
    modelParams.preview_dirs = list(range(25))
    
    args.skip_loading_relit_images = True
    modelParams.skip_loading_relit_images = True

    if args.camera_mode == "orbit":
        args.num_frames = int(args.num_frames*1.5)

    render_sets(args, modelParams, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    