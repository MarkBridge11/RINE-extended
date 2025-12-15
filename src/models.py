import torch
import torch.nn as nn
import clip
import timm

"""
Designed to capture intermediate outputs from specific layers of a neural network during the forward pass, using PyTorch's hook mechanism.
"""
class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)
        # register this hook usable only in the forward pass. Automatically called during the forward pass when encountering the hooked module .
        # hook_fn will give back the input and output tensors of that layer

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


"""
Designed to capture attention weights from a multihead attention module. During model's forward pass with that boolean at true the attention weights are not discarded
and taken with the hook.
This hook is not memory efficient because it re-runs the forward pass, which doubles the computation for that layer, it's only for visualization or inference purpose.
"""
class AttnHook:
    def __init__(self, name, module):
        self.name = name
        self.module = module # should be the attention layer
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = None
        self.input_cache = None

    def hook_fn(self, module, input, output):
        if self.input_cache is None:  # Compute and cache only on first call
            self.input_cache = input
            _, self.output = module(input[0], input[1], input[2], need_weights=True)
            # the three inputs should be QKV, recompute the module and return attention_output and attention_weights
            # attention_weights is a tensor of shape (batch_size,num_heads,seq_len,seq_len)

    def close(self):
        self.hook.remove()

"""
The Model class is a custom PyTorch module that builds on top of a pre-trained CLIP model,
adding trainable projection and classification layers while leveraging CLIP’s frozen features via hooks
"""
class Model(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        self.clip, self.preprocess = clip.load(backbone[0], device=device) #backbone[0] is to load the CLIP model
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
            # Why "ln_2"?: In CLIP’s Vision Transformer, each transformer block typically has two layer normalization layers: ln_1 (before attention) and ln_2 (before the feed-forward network). 
            # Targeting ln_2 suggests the model uses post-feed-forward features.
        ]

        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            # based on the number of times the CLIP's concatenated CLS token are projected. So this means that Q1 and after also Q2 are based on nproj layers composed of ReLU
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    #backbone[1] should be the dimension of the CLIP:ViT-L/14 embedding features dimension. So that enters and the output as the projected space dimension
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 3), 
                #modified to 3 from 1 since now we have three classes. At inference time we have to use softmax instead of sigmoid and categorical CE during training.
            ]
        )

    """
    Each ReLU is followed by dropout, which mitigates overfitting, especially 
    important since CLIP's features are high-dimensional and the trainable parameters (alpha, proj1, proj2, head) are optimized on potentially limited task-specific data.
    """

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :] # 0 means CLS token only, : all the images, : all the hooks, : for each feature_dim of the embedding
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g # here the softmax is considered as weights for TIE
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return p, z

class DINOv2Model(nn.Module):
    def __init__(self, backbone, nproj, proj_dim, device,selected_layers=None):
        super().__init__()
        self.device = device

        self.backbone_name, self.embed_dim = backbone
        self.dino = timm.create_model(self.backbone_name,pretrained=True)
        self.dino.to(device)
        self.dino.eval()
        for name, param in self.dino.named_parameters():
            param.requires_grad = False

        self.num_layers = len(self.dino.blocks)

        self.selected_layers = selected_layers or []
        self.saved_patches = {}

        self.alpha = nn.Parameter(torch.randn([1, self.num_layers, proj_dim]))

        # First projection (from embed_dim → proj_dim)
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
            nn.Linear(proj_dim, 3),  # 3-class output
        )

    def forward(self, x):
        with torch.no_grad():
            B = x.shape[0]

            x = self.dino.patch_embed(x)
            cls_token = self.dino.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.dino.pos_embed
            x = self.dino.pos_drop(x)

            cls_tokens = []
            for i,blk in enumerate(self.dino.blocks):
                x = blk(x)

                if i in self.selected_layers:
                    self.saved_patches[i] = x[:, 1:, :].detach().cpu()

                cls_tokens.append(x[:, 0, :])   # CLS token only

            # Stack -> shape [B, num_layers, embed_dim]
            cls_tokens = torch.stack(cls_tokens, dim=1)

        g = self.proj1(cls_tokens.float())  # shape [B, num_layers, proj_dim]

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)  # weighted sum over layers -> [B, proj_dim]

        z = self.proj2(z)
        p = self.head(z)

        return p, z

    def get_patch_tokens(self, layer_idx=None):
        if layer_idx is None:
            return self.saved_patches
        return self.saved_patches.get(layer_idx, None)