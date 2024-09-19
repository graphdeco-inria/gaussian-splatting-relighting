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
import random
import json
import torch 
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np 
import math 

class Scene:

    gaussians : GaussianModel

    def __init__(self, modelParams : ModelParams, pipeParams: PipelineParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], reload_cameras=False, adjust_znear=True, split="train"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.modelParams = modelParams
        self.pipeParams = pipeParams
        
        self.model_path = modelParams.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = pipeParams.load_iter or searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        print("Source path is:", modelParams.source_path)
        if os.path.exists(os.path.join(modelParams.source_path, split, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](modelParams, modelParams.source_path + "/" + split, modelParams.images, modelParams.eval)
        elif os.path.exists(os.path.join(modelParams.source_path, f"transforms_{split}.json")):
            print(f"Found transforms_{split}.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](modelParams, modelParams.source_path, modelParams.white_background, modelParams.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info = scene_info
        
        if not self.loaded_iter and not reload_cameras:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            if not reload_cameras:
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        print("Total of", len(scene_info.train_cameras), "cameras")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, modelParams)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, modelParams)


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"),
                                              og_number_points=len(scene_info.point_cloud.points),
                                              spatial_lr_scale=self.cameras_extent)
            
            self.gaussians.mlp = torch.load(
                os.path.join(self.model_path,
                                "point_cloud",
                                "iteration_" + str(self.loaded_iter),
                                "mlp.pt")
            )
            self.gaussians.light_vectors = torch.load(
                os.path.join(self.model_path,
                                "point_cloud",
                                "iteration_" + str(self.loaded_iter),
                                "light_vectors.pt")
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        if modelParams.znear_pruning and adjust_znear:
            self.autoadjust_znear()
                

    def autoadjust_znear(self):
        for camera in self.train_cameras[1.0]:
            R = torch.from_numpy(camera.R).cuda().float()
            T = camera.camera_center
            points_world = torch.from_numpy(self.scene_info.point_cloud.points).cuda().float()
            R_c2w_blender = -R 
            R_c2w_blender[:, 0] = -R_c2w_blender[:, 0]
            points_local = (R_c2w_blender.T @ (points_world - T).T).T
            
            x_size = math.tan(camera.FoVx / 2)
            y_size = math.tan(camera.FoVy / 2)
            
            x = points_local[:, 0]
            y = points_local[:, 1]
            z = points_local[:, 2]
            frustrum_mask = (-z > 0) & (x / z > -x_size) & (x / z < x_size) & (y / z > -y_size) & (y / z < y_size)
            points_in_frustrum = points_local[frustrum_mask]
            camera.znear = (-points_in_frustrum[:, 2]).quantile(self.modelParams.znear_quantile) * self.modelParams.znear_scale
            camera.update()

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        torch.save(self.gaussians.mlp, os.path.join(point_cloud_path, "mlp.pt"))
        torch.save(self.gaussians.light_vectors, os.path.join(point_cloud_path, "light_vectors.pt"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]