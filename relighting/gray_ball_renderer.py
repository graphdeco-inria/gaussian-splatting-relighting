import torch 
import numpy as np 
from dataclasses import dataclass
import torch.nn.functional as F 
from torchvision.utils import save_image
import math
from PIL import Image
import torchvision.transforms.functional as TF

@dataclass
class GrayBallRenderer:
    resolution: int
    albedo: float = 0.15
    ambient: float = 2.0
    fresnel: float = 0.25
    spec_power: float = 3.0
    spec: float = 0.4

    def __post_init__(self):
        mask = torch.zeros((self.resolution*4, self.resolution*4, 3))
        for i in range(self.resolution*4):
            for j in range(self.resolution*4):
                if (i-self.resolution*2)**2 + (j-self.resolution*2)**2 < (self.resolution*2)**2:
                    mask[i, j] = 1.0

        self.mask = F.avg_pool2d(mask.permute(2, 0, 1).float().mean(dim=0).nan_to_num(0.0)[None], 4)[0]
        
        # Create normals
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        x, y = np.meshgrid(x, y)
        normals = np.stack([x, -y, np.sqrt(np.abs(1 - x**2 - y**2))], axis=-1)
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
        self.normals = torch.from_numpy(normals).moveaxis(-1, 0) * self.mask 

    def render(self, light_dir):
        self.normals = self.normals.to(light_dir.device)
        self.mask = self.mask.to(light_dir.device)
        view_dir = torch.tensor([0, 0, 1]).to(light_dir.device)
        diffuse = torch.sum(self.normals[None] * light_dir[:, None, None], dim=1, keepdim=True)
        halfway_vec = (light_dir + view_dir).to(light_dir.device)
        halfway_vec = halfway_vec / (torch.norm(halfway_vec, dim=-1, keepdim=True) + 1e-8)
        specular = self.spec * (self.normals * halfway_vec[:, None, None]).sum(dim=0).clamp(0)**math.exp(self.spec_power)
        modified_schlick = 0.04 + (1.0 - 0.04) * (1.0 - (self.normals * view_dir[:, None, None]).sum(dim=0).clamp(0))**2
        return (self.albedo * (diffuse + self.ambient).clamp(0) + specular + self.fresnel * modified_schlick) * self.mask 
    
    def render_onto(self, bg, light_dir, padding=30):
        assert bg.ndim == 4
        bg[:, :, -self.resolution-padding:-padding, -self.resolution-padding:-padding] *= (1.0 - self.mask)
        bg[:, :, -self.resolution-padding:-padding, -self.resolution-padding:-padding] += self.render(light_dir)
        return bg
    

if __name__ == "__main__":
    renderer = GrayBallRenderer(165)

    light_dir = torch.tensor([1.0, 1.0, 1.0]) / np.sqrt(3)
    bg = TF.to_tensor(Image.open("dirs.png").convert("RGB")).to(light_dir.device)[None]

    save_image(renderer.render_onto(bg, light_dir), "withind.png")
