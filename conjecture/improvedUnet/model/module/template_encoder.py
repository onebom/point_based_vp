import torch
import torch.nn as nn

from einops import rearrange

from timm.models.layers import trunc_normal_

from model.module.feature_encoder import FeatureEncoder
from model.module.attention import SinusoidalPosEmb

from model.util import exists, DropPath


def to_2tuple(x):
    return tuple([x] * 2)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    
class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, frame_size=64, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
            int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
            int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1),
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
            
    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape

    
class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 temporal_dim=None,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        if exists(temporal_dim):
            self.adaLN_modulation_qkv = nn.Sequential(
                nn.SiLU(),
                nn.Linear(temporal_dim, 8 * dim, bias=True)
            )

    def forward(self, query, key, temp_emb=None):
        x = query
        
        if exists(temp_emb):
            shift_msa_q, scale_msa_q, shift_msa_kv, scale_msa_kv, gate_msa, shift_mlp, scale_mlp, gate_mlp =  \
                self.adaLN_modulation_qkv(temp_emb).chunk(8, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.drop_path(self.attn(modulate(self.norm_q(query), shift_msa_q, scale_msa_q), 
                                                                    modulate(self.norm_k(key), shift_msa_kv, scale_msa_kv)))
            x = x + gate_mlp.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        else:
            x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x

class SelfAttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 temporal_dim=None,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        if exists(temporal_dim):
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(temporal_dim, 6 * dim, bias=True)
            )

    def forward(self, x, temp_emb=None):
        if exists(temp_emb):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temp_emb).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.drop_path(self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))
            x = x + gate_mlp.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TemplateAttnBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.,
        temporal_dim=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        
        self.attn = SelfAttnBlock(dim, num_heads, mlp_ratio, temporal_dim=temporal_dim, drop=drop, act_layer=act_layer, norm_layer=norm_layer)
        self.cross_attn = CrossAttnBlock(dim, num_heads, mlp_ratio, temporal_dim=temporal_dim, drop=drop, act_layer=act_layer, norm_layer=norm_layer)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
        # self.norm_template = nn.LayerNorm(dim)
        # self.norm_x = nn.LayerNorm(dim)
        
        
        
    def forward(self, template, x, temp_emb = None):
        # template = self.norm_template(template)
        # x = self.norm_x(x)
        
        template = self.attn(template, temp_emb)
        template = self.cross_attn(template, x, temp_emb)
        
        return template
        
        

        
        
class TemplateEncoder(nn.Module):
    def __init__(
        self,
        dim,
        in_chans=3,
        feature_dim=64,
        frame_size=64,
        num_heads=8,
        depth=8,
        patch_size=4,
        mlp_ratio=4.0,
        pos_embed_type='fourier',
    ):
        super().__init__()
        
        self.patch_H, self.patch_W = int(frame_size // patch_size), int(frame_size // patch_size)
        
        self.embed_dim = dim * 4
        
        self.template_query = nn.Parameter(
            torch.zeros(1, self.patch_H * self.patch_W, self.embed_dim)
        )
        trunc_normal_(self.template_query, std=.02)
        
        self.image_encoder = FeatureEncoder(dim=feature_dim)
        self.temp_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        if pos_embed_type == 'fourier':
            self.pos_emb = self.build_2d_sincos_position_embedding()
        elif pos_embed_type == 'simple':
            self.pos_emb = self.build_simple_position_embedding()
        else:
            raise ValueError(f"Unknown pos_embed type {pos_embed_type}")
        
        
        blocks = []
        for i in range(depth):
            blocks.append(TemplateAttnBlock(self.embed_dim, num_heads, mlp_ratio, temporal_dim=self.embed_dim))
        
        self.blocks = nn.ModuleList(blocks)
        
        mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.displacement_head = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 3)
        )
    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_H*self.patch_W, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed
    
    
    
    def loss(self, x_cond, x_pred, template):
        x_cond = rearrange(x_cond, 'b t c h w -> (b t) c h w') # [B*T, C, H, W]
        x_pred = rearrange(x_pred, 'b t c h w -> (b t) c h w') # [B*T, C, H, W]
        template = rearrange(template, 'b t c h w -> (b t) c h w') # [B*T, 3, H, W]
        
        flow = template[:, :2] # [B*T, 2, H, W]
        occlusion = template[:, 2:] # [B*T, 1, H, W]
        
        deform_x = nn.functional.grid_sample(x_cond, flow.permute(0, 2, 3, 1))
        deform_x = deform_x * occlusion
        
        loss = nn.functional.mse_loss(deform_x, x_pred)
        return loss
    
    def forward(self, x_cond, x_pred, temporal_distance=None):
        
        B, C, T1, H, W = x_cond.shape
        _, _, T2, _, _ = x_pred.shape
        
        x = torch.cat([x_cond, x_pred], dim=2) # [B, C, T1+T2, H, W]
        
        temp_emb = temporal_distance
        if exists(temporal_distance):
            temp_emb = self.temp_mlp(temp_emb)
            temp_emb = temp_emb.repeat(T1, 1)
            
        x = rearrange(x, 'b c t h w -> (b t) c h w') # [B*T, 3, 64, 64]
        x_feat = self.image_encoder(x) # [B*T, C=256, H=16, W=16]
        x_feat = rearrange(x_feat, '(b t) c h w -> b t c h w', t=T1+T2)
        x_feat_cond, x_feat_pred = x_feat[:,:T1], x_feat[:,T1:] # [B, T1, C, H, W], [B, T2, C, H, W]
        
        x_feat = rearrange(x_feat_cond, 'b t c h w -> (b t) (h w) c')
        x_feat = x_feat + self.pos_emb
        
        
        template = self.template_query.repeat(B*T1, 1, 1) #[B*T, HW, C]
        template = template + self.pos_emb
        
        for blk_idx, blk in enumerate(self.blocks):
            template = blk(template, x_feat, temp_emb)
        
        template = self.displacement_head(template) # [B*T, H*W, 3]
        template = rearrange(template, '(b t) (h w) c -> b t c h w', b=B, t=T1, h=self.patch_H, w=self.patch_W) # [B, T, 3, H, W]
        
        loss = self.loss(x_feat_cond, x_feat_pred, template)
        
        return loss, template