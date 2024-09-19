import glob 
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import tyro
import os
from PIL import Image 
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_tensor
from dataclasses import dataclass
from tqdm import tqdm 
import torch
from cleanfid import fid
from typing import List, Literal
import json
from kornia.color.lab import rgb_to_lab, lab_to_rgb

#
import numpy as np
from skimage.exposure import match_histograms

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Conf:
    ground_truth_pattern: str = "gts/{scene}/relit_for_eval_dir_{dir:02d}_{i:04d}.png"
    predictions_pattern = {
        # "outcast" : "outcast/{scene}/color_{dir:02d}{i:02d}_dir_{dir:02d}.png",
        # "r3dg" : "r3dg/{scene}/color_{i:04d}_dir_{dir:02d}.png",
        "ours" : "ours/{scene}/{i:05d}_dir_{dir:02d}.png",
        # "tensoir" : "tensoir_png/{scene}/{i:03d}_dir_{dir:02d}_0000.png",
    }
    directions_to_eval: List[int] = tuple([i for i in range(25) if i not in [2,3,19,20,21,22,24]]) #todo 
    matching: Literal["none", "histogram", "meanstd"] = "none"
    normalize: bool = False
    max_id: int = 100

conf = tyro.cli(Conf)

metrics = dict(
    psnr=PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(device), 
    lpips=LearnedPerceptualImagePatchSimilarity(normalize=True).to(device),
    ssim=StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device), 
    # kid=None
)
metrics_arrow = { "psnr": "up", "ssim": "up", "lpips": "down"}
methods = ["ours", "outcast", "r3dg", "tensoir" ]
scenes = ["easy_bedroom" , "easy_kitchen", "easy_livingroom", "easy_office", "hard_bedroom", "hard_kitchen", "hard_livingroom", "hard_office"]
disable_computing = False 

latex_table_src = ""
latex_table_src += "\\begin{table}[]\n"
latex_table_src += "\\resizebox{\\textwidth}{!}{\n"
latex_table_src += "\setlength{\\tabcolsep}{1.5pt}\n"
latex_table_src += "\\begin{tabular}{|c"+"|l"*(len(metrics)*len(methods))+"|}\n"
latex_table_src += "\\hline\n"
latex_table_src += "Method $\\rightarrow$ & "
#Make a header with method name using multiple columns, as many as number of metrics
for method in methods:
    latex_table_src += "\\multicolumn{"+str(len(metrics))+"}{c|}{"+method.replace("_"," ").title()+"} "
    if method != methods[-1]:
        latex_table_src += " & "
latex_table_src += " \\\\\n"
latex_table_src += "\\hline\n"
latex_table_src += "Scene $\downarrow$ / Metrics & "
for method in methods:
    latex_table_src += " & ".join([f"\\footnotesize{{{label.upper()}}} $\{metrics_arrow[label]}arrow$" for label in metrics.keys()])
    if method != methods[-1]:
        latex_table_src += " & "
        
latex_table_src += " \\\\\n"
latex_table_src += "\\hline\n"

base_path = os.path.dirname(os.path.abspath(__file__))

scores_by_scene = {}

for scene in scenes:
    #Replace _ and captialize
    latex_table_src += scene.replace("_", " ").title() 
    scores = {}
    for method in methods:
        scores[method] = { key: 0.0 for key in metrics.keys() }
        
        n_samples=0
        if disable_computing:
            continue
        for i in tqdm(range(0, conf.max_id, 10)):
            for dir in conf.directions_to_eval:
                gt_path = base_path + "/" + conf.ground_truth_pattern.format(scene=scene,i=i, dir=dir)
                pred_path = base_path + "/" + conf.predictions_pattern[method].format(scene=scene,i=i % conf.max_id, dir=dir)
                recolored_path = base_path + f"/recolored_{conf.matching}_" + conf.predictions_pattern[method].format(scene=scene,i=i % conf.max_id, dir=dir)

                gt = np.array(Image.open(gt_path))
                pred = np.array(Image.open(pred_path))
                
                if conf.matching == "histogram":
                    pred = match_histograms(pred,gt,channel_axis=0) 
                    Image.fromarray(pred).save(recolored_path)

                pred = to_tensor(pred)
                gt = to_tensor(gt)

                if conf.matching == "meanstd":
                    def match_color(reference, image):
                        a_lab = rgb_to_lab(reference)
                        b_lab = rgb_to_lab(image)

                        def match_statistics(image, reference):
                            return (image - image.mean()) / image.std() * reference.std() + reference.mean()

                        c_lab_L = match_statistics(b_lab[0:1], a_lab[0:1])
                        c_lab_A = match_statistics(b_lab[1:2], a_lab[1:2])
                        c_lab_B = match_statistics(b_lab[2:3], a_lab[2:3])
                        c_lab = torch.cat([c_lab_L, c_lab_A, c_lab_B])

                        return lab_to_rgb(c_lab) 

                    pred = match_color(gt, pred)
                    os.makedirs(os.path.dirname(recolored_path), exist_ok=True)
                    Image.fromarray((pred * 255).moveaxis(0, -1).cpu().numpy().astype(np.uint8)).save(recolored_path)

                for metric, metric_fn in metrics.items():
                    if metric_fn is not None:
                        scores[method][metric] += metric_fn(pred[None].to(device), gt[None].to(device)).item()
                n_samples += 1

        for metric, metric_fn in metrics.items():
            if metric is not None and n_samples != 0:
                scores[method][metric] /= n_samples
        

    scores_by_scene[scene] = scores
    json.dump(scores_by_scene, open(base_path+f"/scores.json", "w"))

    for method in methods:
        for metric,value in scores[method].items():

            latex_table_src += " & "

            if value == 0.0:
                latex_table_src += "N/A"
                continue

            sorted_method_for_metric = sorted(scores.keys(), key=lambda x: scores[x][metric],reverse=(metrics_arrow[metric]=="down"))

            if method == sorted_method_for_metric[-1]:
                latex_table_src += "\\cellcolor{red!25}"
            if len(methods)>1 and method == sorted_method_for_metric[-2]:
                latex_table_src += "\\cellcolor{orange!25}"
            if len(methods)>2 and method == sorted_method_for_metric[-3]:
                latex_table_src += "\\cellcolor{yellow!25}"

            if metric == "psnr":
                latex_table_src += f"{value:.2f}"
            else:
                latex_table_src += f"{value:.3f}"

    latex_table_src += " \\\\\n"
    latex_table_src += "\\hline\n"

latex_table_src += "\\end{tabular}\n}\n"
latex_table_src += "\\end{table}\n"

#Save the table
with open(base_path+"/latex_table.txt", "w") as f:
    f.write(latex_table_src)