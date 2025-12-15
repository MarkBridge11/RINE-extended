import torch
import torch.nn as nn
import torch.nn.functional as F

class FuseBlock(nn.Module):

    def __init__(self, features=256, use_residual_refine=True):
        super().__init__()
        self.use_residual_refine = use_residual_refine

        # Align channels before fusion
        self.align = nn.Conv2d(features, features, kernel_size=1, bias=False)

        self.refine1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, features),
            nn.ReLU(inplace=True),
        )

        if use_residual_refine:
            self.refine2 = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, features),
                nn.ReLU(inplace=True),
            )

    def forward(self, current, upsampled_prev):
        # Align deep feature to shallow feature space
        upsampled_prev = self.align(upsampled_prev)

        # Additive fusion
        fused = current + upsampled_prev

        refined = self.refine1(fused)

        if self.use_residual_refine:
            refined2 = self.refine2(refined)
            refined = refined + refined2

        return refined


class TamperSegHead(nn.Module):
    def __init__(self, dims=[96, 192, 384, 768], features=256, out_channels=1):
        super().__init__()

        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, dim, kernel_size=1),  # for DINOv2-Base
                nn.ReLU(inplace=True),
            )
            for dim in dims
        ])

        # Scratch layers: normalize all feature levels to 256 channels
        self.scratch_layers = nn.ModuleList([
            nn.Conv2d(dim, features, kernel_size=3, padding=1, bias=False)
            for dim in dims
        ])

        # Three fusion blocks
        self.fuse_blocks = nn.ModuleList([
            FuseBlock(features, use_residual_refine=True) for _ in range(3)
        ])

        # Decoder stage 1 (256 → 128)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, features // 2),
            nn.ReLU(inplace=True)
        )
        self.res1 = nn.Conv2d(features, features // 2, kernel_size=1, bias=False)

        # Decoder stage 2 (128 → 64)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, features // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, features // 4),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Conv2d(features // 2, features // 4, kernel_size=1, bias=False)

        # Final refinement block
        self.final_refine = nn.Sequential(
            nn.Conv2d(features // 4, features // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, features // 4),
            nn.ReLU(inplace=True)
        )

        # Final mask output
        self.mask_out = nn.Conv2d(features // 4, out_channels, kernel_size=1)


    def forward(self, features, patch_h, patch_w):
        outs = []

        for i, f in enumerate(features):
            f = f.permute(0, 2, 1) # reshape 2D feature map
            f = f.reshape(f.shape[0], f.shape[1], patch_h, patch_w)

            f = self.projects[i](f)
            f = self.scratch_layers[i](f)
            f = F.relu(f, inplace=True)

            outs.append(f)

        fused = outs[-1] # deepest feature
        for fuse_block, current in zip(self.fuse_blocks, reversed(outs[:-1])): # progressive fusions (DPT-style)
            upsampled = F.interpolate(
                fused, size=current.shape[-2:],
                mode='bilinear', align_corners=True
            )
            fused = fuse_block(current, upsampled)

        x = self.dec_conv1(fused)
        x = x + self.res1(F.interpolate(fused, size=x.shape[-2:], mode="nearest"))

        x2 = self.dec_conv2(x)
        x = x2 + self.res2(F.interpolate(x, size=x2.shape[-2:], mode="nearest"))

        x = self.final_refine(x)
        mask = self.mask_out(x)
        
        return mask
