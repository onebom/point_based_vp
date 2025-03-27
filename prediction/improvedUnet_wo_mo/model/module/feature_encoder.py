import torch
import torch.nn as nn
from torchvision import models

from einops import rearrange

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8,):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, temporal_distance = None, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, temporal_dim = None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temporal_dim, dim_out*2)
        ) if exists(temporal_dim) else None
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, temp_emb = None, scale_shift = None):
        
        scale_shift = None
        if exists(self.mlp):
            assert exists(temp_emb), 'temporal distance must be provided for temporal mlp'
            temp_emb = self.mlp(temp_emb)
            temp_emb = rearrange(temp_emb, 'b c -> b c 1 1')
            scale_shift = temp_emb.chunk(2, dim=1)
            
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class FeatureEncoder(nn.Module):
    def __init__(
        self,
        dim,
        temporal_dim = None,
        in_chans=3,
        dim_mults = (1, 2, 4),
    ):
        super().__init__()
        self.init_conv = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)
        
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = (idx >= (len(in_out) - 1))
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, temporal_dim=temporal_dim),
                ResnetBlock(dim_out, dim_out, temporal_dim=temporal_dim),
                Downsample(dim_out, dim_out) if not is_last else nn.Identity()
            ]))

    def forward(self, x, temp_emb = None):
        x = self.init_conv(x)
        
        for blk1, blk2, downsample in self.downs:
            x = blk1(x, temp_emb)
            x = blk2(x, temp_emb)
            x = downsample(x)
            
        return x
            
