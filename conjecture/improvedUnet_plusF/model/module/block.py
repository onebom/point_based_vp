import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.module.normalization import Normalization

def exists(val):
    return val is not None

def Upsample(dim, use_deconv=True, padding_mode="reflect"):
    if use_deconv:
        return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(dim, dim, (1, 3, 3), (1, 1, 1), (0, 1, 1), padding_mode=padding_mode)
        )

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        drop=0.,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# building block modules

class Block(nn.Module):
    def __init__(self, 
                 dim, dim_out,
                 kernel,
                 padding, 
                 groups=8, 
                 motion_dim=None, 
                 dropout_rate=0.0):
        super().__init__()
        spade = True if exists(motion_dim) else False
        
        self.conv = nn.Conv3d(dim, dim_out, kernel, padding = padding)
        self.norm = Normalization(dim_out, cond_dim=motion_dim, norm_type='group', num_groups=groups, spade=spade)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, scale_shift = None, motion_cond = None):       
        x = self.conv(x)
        x = self.norm(x, motion_cond)
        
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, conv_method, 
                 use_res = True, 
                 time_emb_dim = None, 
                 groups=8, 
                 motion_dim=None, 
                 dropout_rate=0.0
                 ):
        super().__init__()
        self.use_res = use_res
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        kernel, padding = ((1,3,3), (0,1,1)) if conv_method == "2d" else ((3,1,1), (1,0,0))

        self.block1 = Block(dim, dim_out, kernel, padding, 
                            groups=groups, motion_dim=motion_dim, dropout_rate=dropout_rate)
        self.block2 = Block(dim_out, dim_out, kernel, padding, 
                            groups=groups, motion_dim=motion_dim, dropout_rate=dropout_rate)
        
        if conv_method=="temporal":
            nn.init.zeros_(self.block2.conv.weight)
            nn.init.zeros_(self.block2.conv.bias)
        
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, motion_cond = None):
        if self.use_res:
            scale_shift = None
            if exists(self.mlp):
                assert exists(time_emb), 'time emb must be passed in'
                time_emb = self.mlp(time_emb)
                time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
                scale_shift = time_emb.chunk(2, dim = 1)  #### ??????

            h = self.block1(x, scale_shift = scale_shift, motion_cond = motion_cond)
            h = self.block2(h, motion_cond = motion_cond)
            
            x =  h + self.res_conv(x)
        
        return x
        