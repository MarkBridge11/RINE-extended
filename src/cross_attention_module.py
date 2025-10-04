import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GuidedCrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8): #embed_size should be either the CLIP embedding dimension or Q1+TIE embedding dimension if using guided attention (usually is 1024 too here)
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(embed_size // num_heads) # d_h
        
        # Linear layers for Q, K, V for all heads
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)

        # Output linear layer
        self.out_proj = nn.Linear(embed_size, embed_size)

    def forward(self, cls_feat, patch_tokens):
        """
        cls_feat: [batch_size, embedding_dim]          # aggregated CLS from Q1+TIE, batch_size * embedding_dim (because it was concatenated, otherwise we woudld have also have n for transformer blocks)
        patch_tokens: [batch_size, patches, embedding_dim]   # non-CLS tokens, batch_size * patches * embedding_dim (because it was concatenated, otherwise we woudld have also have n for transformer blocks)
        
        CHECK DIMENSIONS SOMETHING FOR SURE IS WRONG!!!
        """
        batch_size, patches, embedding_dim = patch_tokens.shape
        assert cls_feat.shape[-1] == embedding_dim, "CLS and patch embedding dims must match otherwise something has to change"
        
        # Project queries, keys, values
        q = self.q_proj(cls_feat).view(batch_size, 1, self.num_heads, embedding_dim // self.num_heads) # [B, seq_len, num_heads, d_h]
        k = self.k_proj(patch_tokens).view(batch_size, patches, self.num_heads, embedding_dim // self.num_heads) # [B, seq_len, num_heads, d_h]
        v = self.v_proj(patch_tokens).view(batch_size, patches, self.num_heads, embedding_dim // self.num_heads) # [B, seq_len, num_heads, d_h]

        # Rearrange to [batch_size, num_heads, seq_len, dim] for matmul
        q = q.permute(0, 2, 1, 3)   # [batch_size, h, 1, d_h]
        k = k.permute(0, 2, 1, 3)   # [batch_size, h, patches, d_h]
        v = v.permute(0, 2, 1, 3)   # [batch_size, h, patches, d_h]

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, h, 1, patches]
        attn_weights = F.softmax(attn_scores, dim=-1)                    # [batch_size, h, 1, patches]

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # [batch_size, h, 1, d_h]
        out = out.permute(0, 2, 1, 3).reshape(batch_size, 1, embedding_dim)  # [batch_size, 1, embedding_dim]
        out = self.out_proj(out).squeeze(1)             # [batch_size, embedding_dim]

        # Average heads for a clean patch attention map
        attn_map = attn_weights.mean(1).squeeze(2)      # [batch_size, patches]

        return out, attn_map
