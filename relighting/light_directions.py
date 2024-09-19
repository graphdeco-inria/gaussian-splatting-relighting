import math 
import torch 
import json 
from relighting.sh_encoding import sh_encoding
import os
from typing import Union


# These directions are used as target directions during training, while the other directions from the multilum dataset are excluded.
BACKWARD_DIR_IDS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23)
                   
# Useful directions ids for visualization
LEFT_DIR_ID = 23
TOP_DIR_ID = 14
RIGHT_DIR_ID = 18
BACK_DIR_ID = 10


def get_light_dir_encoding(ids: Union[torch.Tensor, int]):
    if isinstance(ids, int):
        ids = torch.tensor([ids])
        
    encodings = []

    for id in ids:
        id = id.item()
        
        if polar_angles_by_id[id] == {}:
            raise ValueError(f"Direction with id {id} is not available (only directions with a backward-facing camera flash are supported).")

        assert polar_angles_by_id[id]["direction_id"] == id
        phi = polar_angles_by_id[id]["phi"]
        theta = polar_angles_by_id[id]["theta"]

        x_coord = math.sin(theta) * math.cos(phi)
        y_coord = math.sin(theta) * math.sin(phi)
        z_coord = math.cos(theta)

        encodings.append(sh_encoding(torch.tensor([x_coord, y_coord, z_coord])))

    return torch.stack(encodings)

polar_angles_by_id = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "light_directions.json"), "r"))