import torch
import torch.nn as nn
import clip
import timm
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed, grid_h, grid_w): #Interpolates the DINOv2 positional embeddings
    cls_pos = pos_embed[:, 0:1, :]        # [1, 1, C]
    patch_pos = pos_embed[:, 1:, :]       # [1, H*W, C]

    num_patches = patch_pos.shape[1]
    orig_h = orig_w = int(num_patches ** 0.5)

    # reshape to 2D grid
    patch_pos = patch_pos.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)  # [1, C, H, W]

    # interpolate to new grid size
    patch_pos = F.interpolate(patch_pos,size=(grid_h, grid_w),mode="bicubic",align_corners=False)

    # reshape back to tokens
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)

    return torch.cat([cls_pos, patch_pos], dim=1)

class DINOv2Model(nn.Module):
    def __init__(self, backbone, nproj, proj_dim, device, selected_layers=None):
        super().__init__()
        self.device = device

        self.backbone_name, self.embed_dim = backbone
        self.dino = timm.create_model(self.backbone_name, pretrained=True)
        self.dino.to(device)
        self.dino.eval()
        for _, param in self.dino.named_parameters():
            param.requires_grad = False

        self.num_layers = len(self.dino.blocks)
        self.selected_layers = selected_layers or []
        self.saved_patches = {}

        self.alpha = nn.Parameter(torch.randn([1, self.num_layers, proj_dim]))

        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend([
                nn.Linear(self.embed_dim if i == 0 else proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
            ])
        self.proj1 = nn.Sequential(*proj1_layers)

        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend([
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
            ])
        self.proj2 = nn.Sequential(*proj2_layers)

        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, 1) 
        )

    def forward(self, x_input):
        with torch.no_grad():
            B, _, H, W = x_input.shape

            x_patches = self.dino.patch_embed(x_input)   # [B, N, C]
            cls_token = self.dino.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x_patches), dim=1)

            ph, pw = self.dino.patch_embed.patch_size
            grid_h = H // ph
            grid_w = W // pw

            pos_embed = interpolate_pos_embed(self.dino.pos_embed,grid_h, grid_w)
            x = x + pos_embed
            x = self.dino.pos_drop(x) #dropout

            cls_tokens = []
            for i, blk in enumerate(self.dino.blocks):
                x = blk(x)
                if i in self.selected_layers:
                    self.saved_patches[i] = x[:, 1:, :].detach().cpu()
                cls_tokens.append(x[:, 0, :])

            cls_tokens = torch.stack(cls_tokens, dim=1)

        g = self.proj1(cls_tokens.float())
        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)
        logits = self.head(z)

        return logits, z

    def get_patch_tokens(self, layer_idx=None):
        if layer_idx is None:
            return self.saved_patches
        return self.saved_patches.get(layer_idx, None)
