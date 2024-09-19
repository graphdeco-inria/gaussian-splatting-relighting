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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, PILtoTorchU8
from utils.graphics_utils import fov2focal
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    # if args.resolution in [1, 2, 4, 8]:
    #     resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    # else:  # should be a type that converts to float
    #     if args.resolution == -1:
    #         if orig_w > 1600:
    #             global WARNED
    #             if not WARNED:
    #                 print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
    #                     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
    #                 WARNED = True
    #             global_down = orig_w / 1600
    #         else:
    #             global_down = 1
    #     else:
    #         global_down = orig_w / args.resolution

    #     scale = float(global_down) * float(resolution_scale)
    #     resolution = (int(orig_w / scale), int(orig_h / scale))

    if args.resolution == [-1, -1]:
        # if orig_w > 1600:
        #     global WARNED
        #     if not WARNED:
        #         print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
        #             "If this is not desired, please explicitly specify '--resolution/-r' as 1")
        #         WARNED = True
        #     global_down = orig_w / 1600
        # else:
        #     global_down = 1
        resolution = cam_info.image.size
    else:
        resolution = args.resolution

    if cam_info.relit_images is not None:
        relit_images = [
            PILtoTorchU8(image, resolution)[:3, ...] if image is not None else None for image in cam_info.relit_images
        ]
    else:
        relit_images = None
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, relit_images=relit_images, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    with ThreadPoolExecutor(max_workers=9) as executor:
        for result in tqdm(executor.map(lambda x: loadCam(args, x[0], x[1], resolution_scale), enumerate(cam_infos))):
            camera_list.append(result)

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
