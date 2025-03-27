import torch
import torch.nn as nn
from einops import rearrange

from model.util import EinopsToAndFrom
from model.module.attention import TemporalAttentionLayer,AttentionModule
from model.module.normalization import Normalization

from rotary_embedding_torch import RotaryEmbedding

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

# building block modules
class MotionBlock(nn.Module):
    def __init__(
            self,
            ch,
            fea_ch,
            mc_dim,
            dropout_rate=0.
            ):
        super().__init__()
        self.x_embedding = nn.Sequential(nn.Linear(ch, fea_ch),
                                         nn.LayerNorm(fea_ch),
                                         nn.ReLU(inplace=True), 
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(fea_ch, fea_ch),
                                         nn.LayerNorm(fea_ch),
                                         nn.Dropout(dropout_rate)
                                       )
        
        self.mc_embedding = nn.Sequential(nn.Linear(fea_ch*2, mc_dim),
                                         nn.LayerNorm(mc_dim),
                                         nn.ReLU(inplace=True), 
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(mc_dim, mc_dim),
                                         nn.LayerNorm(mc_dim),
                                         nn.Dropout(dropout_rate)
                                       )
        
    def forward(self, x, x_fea):
        B,C,T,PN = x.shape
        x = rearrange(x, 'b c t pn -> b (t pn) c')
        x_fea = rearrange(x_fea, 'b c t pn -> b (t pn) c')

        x = self.x_embedding(x)
        mc = torch.concat([x, x_fea], dim=-1)

        mc = self.mc_embedding(mc)

        return mc

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

class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, conv_method, 
                 time_emb_dim = None, 
                 groups=8, 
                 motion_dim=None, 
                 dropout_rate=0.0,
                 kernel=None,
                 stride=None,
                 padding=None,
                 ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        if kernel is None or padding is None:
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
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)  #### ??????

        h = self.block1(x, scale_shift = scale_shift, motion_cond = motion_cond)
        h = self.block2(h, motion_cond = motion_cond)
        
        # x =  h + self.res_conv(x)
        res_x = self.res_conv(x)
        if res_x.shape==h.shape:
            h = h + res_x
        
        return h

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        cond_dim = None,
        time_emb_dim = None,
        resnet_groups = None
    ):
        super().__init__()

        self.t_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_in)
        ) if exists(time_emb_dim) else None

        self.block2D = ConvBlock(dim_in, dim_out, 
                                   time_emb_dim = time_emb_dim, 
                                   conv_method="2d", 
                                   groups=resnet_groups)
        
        self.blockTemporal = ConvBlock(dim_out, dim_out, 
                                         conv_method="temporal", 
                                         groups=resnet_groups, 
                                         dropout_rate = 0.1)
        
        dim_in = cond_dim if cond_dim is not None else dim_in
        self.mc_block = nn.Sequential(nn.Linear(dim_in, dim_out),
                                         nn.SiLU(), 
                                         nn.Dropout(0.1),
                                         nn.Linear(dim_out, dim_out),
                                         nn.Dropout(0.1)
                                       )

        
    def forward(self, x, mc, emb):
        """
        vid: (b, c, t, h, w)
        trj: (b,(t, pn), c)
        """
        B,C,T,H,W = x.shape
        
        t = emb["t"]
        
        # 1. vid block 
        x = self.block2D(x, t)
        x = self.blockTemporal(x)
        
        mc = self.mc_block(mc)
        return x, mc

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        use_attn,
        attn_heads = 8,
        attn_dim_head = 32
    ):
        super().__init__()
        self.use_attn = use_attn
        
        if self.use_attn:
            self.attn2D = AttentionModule(dim, shape = "(b t) tk c")
            self.attnST = AttentionModule(dim, shape = "b (tk t) c")
            
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.attnTemporal = EinopsToAndFrom('b c t h w', 'b (h w) t c', 
                                             TemporalAttentionLayer(
                                                 dim, heads = attn_heads,
                                                 dim_head = attn_dim_head,
                                                 rotary_emb = rotary_emb
                                                 )
                                             )
        
    def forward(self, x, m, emb):
        B, C, T, H, W = x.shape
        
        frame_idx = emb["frame_idx"]
        time_rel_pos_bias = emb["time_rel_pos_bias"]
        
        if self.use_attn:
            x = rearrange(x, 'b c t h w -> b c t (h w)')
            m = rearrange(m, 'b (t pn) c -> b c t pn', t=T)
                
            x = self.attn2D(x, m, frame_idx = frame_idx)
            x = self.attnST(x, m, frame_idx = frame_idx)        
                        
            x = rearrange(x, 'b c t (h w) -> b c t h w', h=H, w=W)

        x = self.attnTemporal(x, pos_bias=time_rel_pos_bias)
        
        return x