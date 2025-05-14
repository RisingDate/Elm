# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class XTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class XTransformer(nn.Module):
    def __init__(self, input_dim, dim=64, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.transformer = nn.Sequential(*[XTransformerBlock(dim, heads, dropout) for _ in range(depth)])
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        # x: (B, input_dim)
        x = self.input_proj(x).unsqueeze(1)  # -> (B, 1, dim)
        x = self.transformer(x)  # -> (B, 1, dim)
        x = x.squeeze(1)  # -> (B, dim)
        return self.output_head(x)  # -> (B, 1)
