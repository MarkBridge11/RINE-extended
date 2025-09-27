import torch
import torch.nn as nn
import clip


"""
The Hook class is designed to capture intermediate outputs from specific layers of a neural network during the forward pass, using PyTorch’s hook mechanism.

"""

class Hook:
    def __init__(self, name, module):
        self.name = name # stores the layer's name
        self.hook = module.register_forward_hook(self.hook_fn)
        # register this hook usable only in the forward pass. Automatically called during the forward pass when encountering the hooked module .
        # hook_fn will give back the input and output tensors of that layer

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove() #when the hook is no longer needed is removed


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

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device) #backbone[0] is to load the CLIP model
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules() # to iterate over all CLIP's visual transformer
            if "ln_2" in name 
            # Why "ln_2"?: In CLIP’s Vision Transformer, each transformer block typically has two layer normalization layers: ln_1 (before attention) and ln_2 (before the feed-forward network). 
            # Targeting ln_2 suggests the model uses post-feed-forward features.
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim])) # A that has dimension 1,n blocks of transformer, projected dimension of embeddings 
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
        self.proj1 = nn.Sequential(*proj1_layers) # to put everything in sequential 
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj): #same.
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers) #same
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )

    """
    Each ReLU is followed by dropout, which mitigates overfitting, especially 
    important since CLIP’s features are high-dimensional and the trainable parameters (alpha, proj1, proj2, head) are optimized on potentially limited task-specific data.
    """

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)

            for h in self.hooks[:1]:  # just check first block
                print("Hook output:", h.output.shape)      # should be [B, seq_len, D]
                print("CLS slice:", h.output[:, 0, :].shape)  # [B, D]
            
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :] # 0 means the current image being processed, : all the tokens, : all the hooks, : for each feature_dim of the embedding
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g # here the softmax is considered as weights for TIE
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return p, z
