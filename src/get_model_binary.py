import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import Model 
from src.models_binary import DINOv2Model 

import torchvision.transforms.functional as TF # added

def get_model(device,backbone,ckpt_path,selected_layers,nproj,proj_dim):
    if backbone=="ViT-L/14":
        model = Model(
            backbone=("ViT-L/14", 1024),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
        )
    else:
        model = DINOv2Model(
            backbone=("vit_base_patch14_dinov2",768),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
            selected_layers=selected_layers,
        )
    #state_dict = torch.load(ckpt_path, map_location=device)
    #for name in state_dict:
    #    exec(
    #        f'model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
    #    )

    state_dict = torch.load(ckpt_path, map_location=device)

    missing_keys = []
    for name, value in state_dict.items():
            try:
                parts = name.split('.')
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                # Wrap in nn.Parameter only if it’s an actual parameter
                if isinstance(value, torch.Tensor):
                    setattr(obj, parts[-1], torch.nn.Parameter(value))
                else:
                    setattr(obj, parts[-1], value)
            except AttributeError:
                missing_keys.append(name)

            if missing_keys:
                print(f"⚠️ Warning: some keys from checkpoint were not found in model ({len(missing_keys)}):")
                for k in missing_keys[:10]:
                    print(f"  - {k}")
                if len(missing_keys) > 10:
                    print("  ...")


    return model
