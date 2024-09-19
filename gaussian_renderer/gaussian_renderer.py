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
import math
import os 
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from arguments import PipelineParams
import random 
import os 
from relighting.sh_encoding import sh_encoding

import torch.nn.functional as F 


def render(viewpoint_camera, pc : GaussianModel, pipe: PipelineParams, bg_color : torch.Tensor=None, scaling_modifier = 1.0, override_color = None, light_id: int = None, light_vec=None, fd_dropout=0.0, fake_point=None, override_view_id=None, reso_factor=1.0, warmup=False, reference_camera_pose=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    if bg_color is None:
        bg_color = torch.tensor([0.0, 0.0, 0.0], device="cuda")
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=not pc.modelParams.freeze_geometry, device="cuda") + 0
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=not pc.freeze_geometry, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height * reso_factor),
        image_width=int(viewpoint_camera.image_width * reso_factor),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False # you may need to remove this depdending on your version of the rasterizer
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if pc.modelParams.znear_pruning:
        camera = viewpoint_camera
        if isinstance(camera.R, torch.Tensor):
            R = camera.R.cuda().float()
        else:
            R = torch.from_numpy(camera.R).cuda().float()
        if isinstance(camera.camera_center, torch.Tensor):
            T = camera.camera_center
        else:
            T = torch.from_numpy(camera.camera_center)
        points_world = pc.get_xyz
        R_c2w_blender = -R 
        R_c2w_blender[:, 0] = -R_c2w_blender[:, 0]
        points_local = (R_c2w_blender.T @ (points_world - T).T).T
        
        x_size = math.tan(camera.FoVx / 2)
        y_size = math.tan(camera.FoVy / 2)
        
        x = points_local[:, 0]
        y = points_local[:, 1]
        z = points_local[:, 2]
        in_frustrum_cone = (x / z > -x_size) & (x / z < x_size) & (y / z > -y_size) & (y / z < y_size)
        mask = ~in_frustrum_cone | (in_frustrum_cone & (-z > viewpoint_camera.znear))
        
        scales = scales[mask]
        rotations = rotations[mask]
        means3D = means3D[mask]
        means2D = means2D[mask]
        opacity = opacity[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
    else:
        mask = torch.ones(means3D.shape[0], dtype=torch.bool, device=means3D.device)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        dir_pp = pc.get_xyz[mask] - (fake_point if fake_point is not None else viewpoint_camera).camera_center.repeat(pc.get_features[mask].shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        view_pe = sh_encoding(dir_pp_normalized, pc.modelParams.num_encodings_view)

        if not warmup:
            assert light_id is not None or light_vec is not None

            if light_vec is None:
                if light_id == -1:
                    local_dir_blender = torch.tensor([0.0, 0.0, 0.0], device="cuda")
                else:
                    from relighting.light_directions import polar_angles_by_id
                    theta = polar_angles_by_id[light_id]["theta"]
                    phi = polar_angles_by_id[light_id]["phi"]

                    x_phys_cam_conv = math.sin(theta) * math.cos(phi)
                    y_phys_cam_conv = math.sin(theta) * math.sin(phi)
                    z_phys_cam_conv = math.cos(theta)

                    local_dir_blender = torch.tensor([x_phys_cam_conv, z_phys_cam_conv, -y_phys_cam_conv], device="cuda")
                
                R_c2w_colmap = torch.from_numpy((viewpoint_camera if reference_camera_pose is None else reference_camera_pose).R).cuda().float()
                R_c2w_blender = -R_c2w_colmap
                R_c2w_blender[:, 0] = -R_c2w_blender[:, 0]

                light_vec = R_c2w_blender @ local_dir_blender

            flash_pe = sh_encoding(light_vec[None], pc.modelParams.num_encodings_light)

            # Include the per-view vector
            view_id = int(viewpoint_camera.image_name.split("_")[-1]) if override_view_id is None else override_view_id
            if view_id == "mean":
                residual = pc.light_vectors[1:51].mean(dim=0, keepdim=True)
            else:
                residual = pc.light_vectors[view_id][None]
            flash_pe = torch.cat([flash_pe, residual], dim=1)

            net_in = torch.cat([view_pe, pc.get_features[mask].flatten(1, 2), flash_pe.repeat(pc.get_features[mask].shape[0], 1)], dim=1)

            colors_precomp = pc.mlp(net_in)
        else:
            net_in = torch.cat([view_pe, pc.get_features[mask].flatten(1, 2)], dim=1)
            if pc.modelParams.num_warmup_iters != -1:
                net_in = torch.cat([net_in, torch.zeros(net_in.shape[0], pc.mlp.in_feat - net_in.shape[1]).cuda()], dim=-1)
                colors_precomp = pc.mlp(net_in)[:, :3]
            else:
                colors_precomp = pc.mlp(net_in)

    else:
        colors_precomp = override_color
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if colors_precomp is not None:
        colors_precomp = colors_precomp.float()

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    visibility_filter = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
    visibility_filter[mask] = radii > 0
    radii2 = torch.torch.zeros_like(mask, dtype=torch.int, device=mask.device)
    radii2[mask] = radii
    result = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : visibility_filter,
        "radii": radii2,
        "mask": mask
    }

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return result


