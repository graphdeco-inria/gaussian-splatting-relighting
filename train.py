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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import math 
from kornia.color.lab import rgb_to_lab, lab_to_rgb

import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import traceback
import json

def training(modelParams: ModelParams, optParams: OptimizationParams, pipeParams: PipelineParams, testing_iterations, saving_iterations, resume=False):
    if not args.resume:
        tb_writer = prepare_output_and_logger(modelParams) 
    gaussians = GaussianModel(modelParams, pipeParams)
    scene = Scene(modelParams, pipeParams, gaussians, load_iteration=-1 if resume else None)

    if args.resume:
        tb_writer = prepare_output_and_logger(modelParams)
        scene.model_path = modelParams.model_path
    gaussians.training_setup(optParams)

    bg_color = [1, 1, 1] if modelParams.white_background else [0, 0, 0]

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(optParams.iterations), desc="Training progress")

    cameras_by_id = { int(k.split("_")[-1]):v for k,v in [(cam.image_name, cam) for cam in scene.getTrainCameras()] }

    if modelParams.use_key_views:
        key_views = json.load(open(modelParams.source_path + "/train/key_views.json"))

    for iteration in range(1, optParams.iterations + 1):     
        if pipeParams.rand_background:
            bg_color = list(torch.rand(3))
        else:
            bg_color = list(torch.zeros(3))
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        while True:
            if optParams.viewer:
                if network_gui.conn == None:
                    network_gui.try_connect()
                    
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipeParams.convert_SHs_python, pipeParams.compute_cov3D_python, keep_alive, scaling_modifer, theta, phi, renderMode, net_view_id, net_light_id = network_gui.receive()
                        if custom_cam != None:
                            x_coord = math.sin(theta) * math.cos(phi)
                            y_coord = math.sin(theta) * math.sin(phi)
                            z_coord = math.cos(theta)
                            light_vec_local = torch.tensor([x_coord, z_coord, -y_coord], device="cuda")
                
                            R_c2w_colmap = custom_cam.world_view_transform[:3, :3].cuda().float()
                            R_c2w_blender = -R_c2w_colmap
                            R_c2w_blender[:, 0] = -R_c2w_blender[:, 0]
                            light_vec = R_c2w_blender @ light_vec_local

                            with torch.no_grad():
                                net_image = render(custom_cam, gaussians, pipeParams, background, scaling_modifer, light_vec=light_vec, override_view_id="mean")["render"]
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                            iteration += 1
                        network_gui.send(net_image_bytes, modelParams.source_path + "\\train")
                        if args.resume and not args.train_after_resume:
                            do_training = False
                        if do_training and ((iteration < int(optParams.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        print("failed to send or render")
                        traceback.print_exc()
                        network_gui.conn = None
            if not (args.resume and not args.train_after_resume):
                break
                

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            if args.camera_ids != [-1]:
                viewpoint_stack = [cameras_by_id[cam_id] for cam_id in args.camera_ids]
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
        if args.camera_ids != [-1] and iteration == modelParams.num_warmup_iters:
            viewpoint_stack = [cameras_by_id[cam_id] for cam_id in args.camera_ids]
        
        if iteration > modelParams.num_warmup_iters and modelParams.use_key_views and iteration % modelParams.key_view_every_k_step == 0:
            viewpoint_cam = cameras_by_id[key_views[(iteration // modelParams.key_view_every_k_step) % len(key_views)]]
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        losses = {}
        
        # Render
        if iteration > modelParams.num_warmup_iters:
            k = random.choice(args.train_dirs)
            render_pkg = render(viewpoint_cam, gaussians, pipeParams, background, light_id=k)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if k == -1:
                target = viewpoint_cam.original_image.cuda()
            else:
                target = viewpoint_cam.relit_images[k].cuda() / 255.0

            image = F.interpolate(image[None], target.shape[1:], mode="bilinear", antialias=True)[0]
            
            Ll1 = l1_loss(image, target)
            losses["l1"] = (1.0 - optParams.lambda_dssim) * Ll1 
            losses["ssim"] = optParams.lambda_dssim * (1.0 - ssim(image, target)) 
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipeParams, background, warmup=modelParams.num_warmup_iters != -1)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            target = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, target)
            losses["l1"] = (1.0 - optParams.lambda_dssim) * Ll1 
            losses["ssim"] =  optParams.lambda_dssim * (1.0 - ssim(image, target)) 

        loss = sum(losses.values())
        loss.backward() 
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == optParams.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipeParams, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians ({} total points)".format(iteration, len(scene.gaussians._xyz)))
                scene.save(iteration)

            # Optimizer step
            if iteration < optParams.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)
            
            # Densification
            if (not args.resume or args.densify_after_resume) and iteration < optParams.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) #?? need to account for both grads?
                if iteration > optParams.densify_from_iter and iteration % optParams.densification_interval == 0:
                    size_threshold = 20 if iteration > optParams.opacity_reset_interval else None
                    gaussians.densify_and_prune(optParams.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                elif modelParams.znear_pruning:
                    with torch.no_grad():
                        scene.gaussians.prune_points(~render_pkg["mask"])
            
                if iteration % optParams.opacity_reset_interval == 0 or (modelParams.white_background and iteration == optParams.densify_from_iter):
                    gaussians.reset_opacity()

def prepare_output_and_logger(args):    
    dir_name = os.path.join(".", "output", *os.path.relpath(args.source_path, ".").split("/")[1:])
    if args.label == "":
        os.makedirs(dir_name, exist_ok=True)
        args.model_path = dir_name + "/" + f"{len(os.listdir(dir_name)):02d}"
    else:
        args.model_path = dir_name + "/" + args.label
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    return SummaryWriter(args.model_path)

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, render, renderArgs):
    if tb_writer:
        for key, value in losses.items():
            tb_writer.add_scalar(f'train_losses/{key}', value.item(), iteration)
        tb_writer.add_scalar('train_losses/total_loss', sum(losses.values()).item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        if args.camera_ids == [-1]:
            all_train_cams = scene.getTrainCameras()
        else:
            all_train_cams = [x for x in scene.getTrainCameras() if int(x.image_name) in args.camera_ids]
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [all_train_cams[idx % len(all_train_cams)] for idx in range(5, 30, 5)]})
        
        if args.rand_background:
            bg_color = list(torch.rand(3))
        else:
            bg_color = list(torch.zeros(3))
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                
                avg_l1_test = torch.tensor(0.0, device="cuda")
                avg_psnr_test = torch.tensor(0.0, device="cuda")

                for idx, viewpoint in enumerate(config['cameras']):
                    for k in scene.gaussians.modelParams.preview_dirs:
                        image = torch.clamp(render(viewpoint, scene.gaussians, *renderArgs, light_id=k)["render"], 0.0, 1.0)
                        if k == -1:
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        else:
                            gt_image = torch.clamp(viewpoint.relit_images[k].to("cuda") / 255.0, 0.0, 1.0)
                        
                        images = torch.cat((images, image.unsqueeze(0)), dim=0)
                        gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
                        
                        if tb_writer and (idx < 5): 
                            save_image(torch.stack([image, gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view_{idx}_dir_{k:02d}_iter_{iteration:09}.png", padding=0)

                        avg_l1_test += l1_loss(images, gts)
                        avg_psnr_test += psnr(images, gts).mean()            

                avg_l1_test /= len(scene.gaussians.modelParams.preview_dirs) * len(config['cameras'])
                avg_psnr_test /= len(scene.gaussians.modelParams.preview_dirs) * len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], avg_l1_test, avg_psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', avg_l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', avg_psnr_test, iteration)
                    with open(f"{tb_writer.log_dir}/metrics_{config['name']}.txt", "a") as file:
                        print(f"{iteration}: {avg_psnr_test}", file=file)
                
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    modelParamsParser = ModelParams(parser)
    optimParamsParser = OptimizationParams(parser)
    pipelineParamsParser = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1_000, 5_000, 10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1_000, 5_000, 10_000, 20_000, 30_000])

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    args.save_iterations.append(args.iterations)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    modelParams = modelParamsParser.extract(args)

    modelParams.resume = args.resume
    training(modelParams, optimParamsParser.extract(args), pipelineParamsParser.extract(args), args.test_iterations, args.save_iterations, args.resume)

    # All done
    print("\nTraining complete.")