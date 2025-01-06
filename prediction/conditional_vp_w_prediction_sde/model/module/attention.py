import math
import copy
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial, reduce, lru_cache
from operator import mul

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from rotary_embedding_torch import RotaryEmbedding

from einops_exts import rearrange_many
from model.util import exists, default
from model.module.normalization import Normalization
from model.module.block import Mlp

def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000.):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def get_window_size(x_size, window_size, shift_size):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class TemporalCondition(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.,
        spatial=False,
        temporal=False,
    ):
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal
        
        if (self.spatial or self.temporal) == False:
            raise ValueError('At least one of spatial or temporal must be True')
        
        hidden_dim = int(dim * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Linear(dim+1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x, temporal_distance=None):
        if temporal_distance == None:
            return x
        B, C, T, H, W = x.shape
        
        if self.spatial:
            x = rearrange(x, 'B C T H W -> (B T) (H W) C')
            temporal_distance = repeat(temporal_distance, 'B -> (B T) (H W) 1', T=T, H=H, W=W)
        
            x = torch.cat([x, temporal_distance], dim=-1) # (BT, HW, C+1)
            x = self.fc(x) # (BT, HW, C)
        
            x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, H=H, W=W)
        if self.temporal:
            x = rearrange(x, 'B C T H W -> (B H W) T C')
            temporal_distance = repeat(temporal_distance, 'B -> (B H W) T 1', T=T, H=H, W=W)
            
            x = torch.cat([x, temporal_distance], dim=-1) # (BHW, T, C+1)
            x = self.fc(x)
            x = rearrange(x, '(B H W) T C -> B C T H W', B=B, H=H, W=W)
        
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device, frame_idx = None):
        if frame_idx is not None:
            q_pos = frame_idx.unsqueeze(2)
            k_pos = frame_idx.unsqueeze(1)
            rel_pos = k_pos - q_pos
            rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
            values = self.relative_attention_bias(rp_bucket)
            return values.permute(0, 3, 1, 2) # [B, H, N, N]
        else:
            q_pos = torch.arange(n, dtype = torch.long, device = device)
            k_pos = torch.arange(n, dtype = torch.long, device = device)
            rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
            rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
            values = self.relative_attention_bias(rp_bucket)
            return rearrange(values, 'i j h -> h i j')
        

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, dim_head, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rotary_emb=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = qk_scale or dim_head ** -0.5
        self.rotary_emb = rotary_emb
        hidden_dim = dim_head * num_heads

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class STWAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(2,4,4),
        shift_size=(0,0,0),
        heads=8,
        dim_head=32,
        rotary_emb=None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.attn = WindowAttention3D(dim, window_size=window_size, num_heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        mask_matrix = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(self.dim_head*self.heads,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x
    
# class SpatialAttentionLayer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         heads = 4,
#         dim_head = 32,
#         rotary_emb = None,
#         drop = 0.,
#         drop_path = 0.,
#         temporal_cond = False,
#         mlp_ratio = 4.
#     ):
#         super().__init__()
#         self.temporal_cond = temporal_cond
#         hidden_dim = int(dim * mlp_ratio)
        
#         self.norm = Normalization(dim, norm_type='layer')
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
#         self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)
        
#         if self.temporal_cond:
#             self.fc = Mlp(in_features=dim+1, hidden_features=hidden_dim, out_features=dim, drop=drop)
            
#         self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, drop=drop)

#     def forward(self, x, temporal_distance=None):
#         """_summary_
#         Args:
#             x : [BT, HW, 2C]
#         """
#         B, C, T, H, W = x.shape
#         x = rearrange(x, 'b c t h w -> (b t) (h w) c')
#         if exists(temporal_distance):
#             temporal_distance = repeat(temporal_distance, 'b -> (b t) (h w) 1', t=T, h=H, w=W)
#             x = torch.cat([x, temporal_distance], dim=-1)
#             x = self.fc(x)
#         x = x + self.drop_path(self.norm(x))
#         x = x + self.drop_path(self.mlp(x))
        
#         x = rearrange(x, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)

#         return x
    
class SpatialAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        temp_dim = 256,
        heads = 4,
        dim_head = 32,
        rotary_emb = None,
        drop = 0.,
        drop_path = 0.,
        temporal_cond = False,
        mlp_ratio = 4.,
        use_attn=True
    ):
        super().__init__()
        self.temporal_cond = temporal_cond
        self.use_attn = use_attn
        hidden_dim = int(dim * mlp_ratio)
        
        self.norm1 = Normalization(dim, norm_type='layer')
        self.norm2 = Normalization(dim, norm_type='layer')
        
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)
        
        self.temp_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temp_dim, dim * 2)
        ) if exists(temp_dim) else None
        self.act = nn.SiLU()
        
        if self.temporal_cond:
            self.fc = Mlp(in_features=dim+1, hidden_features=hidden_dim, out_features=dim, drop=drop)
            
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, drop=drop)

    def forward(self, x, temporal_distance=None):
        """_summary_
        Args:
            x : [B 2C T H W]
        """
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        
        if exists(temporal_distance):
            temporal_distance = repeat(temporal_distance, 'b -> (b t) (h w) 1', t=T, h=H, w=W)
            x = torch.cat([x, temporal_distance], dim=-1)
            x = self.fc(x)
            
        if self.use_attn:
            x = x + self.attn(self.norm1(x))
    
        x = x + self.mlp(self.norm2(x))
        
        x = rearrange(x, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)

        return x
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=32, dropout=0., mlp_ratio=4.):
        super().__init__()
        
        self.norm_q = Normalization(query_dim, norm_type='layer')
        if exists(context_dim):
            self.norm_k = Normalization(context_dim, norm_type='layer')
        self.norm_mlp = Normalization(query_dim, norm_type='layer')
        
        self.attn = CrossAttention(query_dim, context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = Mlp(in_features=query_dim, hidden_features=int(mlp_ratio*query_dim))
        
    def forward(self, x, context=None):
        x = x + self.attn(self.norm_q(x), context=self.norm_k(context) if exists(context) else None)
        x = x + self.mlp(self.norm_mlp(x))
        return x
        


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        qkv_fuse = False,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        else:
            self.to_q = nn.Linear(dim, hidden_dim, bias = False)
            self.to_k = nn.Linear(dim, hidden_dim, bias = False)
            self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        query,
        key = None,
        value = None,
        pos_bias = None,
    ):
        B, N, C, device = *query.shape, query.device

        if self.qkv_fuse:
            assert key is None and value is None
            qkv = self.to_qkv(query).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            if key is None:
                key = query
            if value is None:
                value = key
            q = rearrange(self.to_q(query), 'b n (h c) -> b h n c', h = self.heads)
            k = rearrange(self.to_k(key), 'b n (h c) -> b h n c', h = self.heads)
            v = rearrange(self.to_v(value), 'b n (h c) -> b h n c', h = self.heads)

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum('b h n d, b h m d -> b h n m', q, k) * self.scale

        # relative positional bias
        if exists(pos_bias):
            if pos_bias.dim() == 3:
                pos_bias = pos_bias.unsqueeze(0)
            mul = sim.shape[0] // pos_bias.shape[0]
            sim = sim + pos_bias.repeat(mul, 1, 1, 1)

        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class AttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=32,
        rotary_emb=None,
        mlp_ratio=4.,
        is_cross=False
    ):
        super().__init__()
        self.norm_q = Normalization(dim, norm_type='layer')
        if is_cross:
            self.norm_k = Normalization(dim, norm_type='layer')
        self.norm_mlp = Normalization(dim, norm_type='layer')
        
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim))
    
    def forward(self, query, key=None, value=None, pos_bias=None):
        out = query + self.attn(self.norm_q(query), 
                                  key=self.norm_k(key) if key is not None else None,
                                  value=self.norm_k(value) if value is not None else None,
                                  pos_bias=pos_bias)
        out = out + self.mlp(self.norm_mlp(out))
        return out
        
        
        
    
class TemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.hidden_dim = hidden_dim

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
    ):
        b, n, device = x.shape[0], x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        
        # split out heads
        q, k, v = rearrange_many(qkv, 'b m n (h d) -> (b m) h n d', h = self.heads)

        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias
        if exists(pos_bias):  
            mul = sim.shape[0] // pos_bias.shape[0]
            sim = sim + pos_bias.repeat(mul, 1, 1, 1)

        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '(b m) h n d -> b m n (h d)', b=b)
        return self.to_out(out)

class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None,
        norm_layer=nn.LayerNorm,
        drop=0.,
        act_layer=nn.GELU,
        drop_path=0.,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = TemporalAttention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)

    def forward(self, x, pos_bias=None):
        r = x
        x = self.norm(x)
        x = self.attn(x, pos_bias)
        x = r + x
        return x
    
class STAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_t=None,
        heads=8,
        dim_head=32,
        mlp_ratio=4.,
        norm_type='layer',
        is_cross=False,
    ):
        super().__init__()
        
        if dim_t is None:
            dim_t = dim
            
        rotary_emb = RotaryEmbedding(min(32, dim_head))
        
        self.attn_s = AttentionLayer(dim, heads=heads, dim_head=dim_head, is_cross=is_cross)
        self.attn_t = AttentionLayer(dim_t, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb, is_cross=is_cross)
        
    def forward(self, x, context=None, pos_bias=None):
        """_summary_
        Args:
            x : [B, C, T1, H, W]
            context : [B, C, T2, H, W]
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        context = rearrange(context, 'b c t h w -> (b t) (h w) c') if exists(context) else None
        x = self.attn_s(query=x,
                        key=context if exists(context) else None)
        x = rearrange(x, '(b t) (h w) c -> b c t h w', h=H, w=W, t=T)
        context = rearrange(context, '(b t) (h w) c -> b c t h w', h=H, w=W, t=T) if exists(context) else None
        
        x = rearrange(x, 'b c t h w -> (b h w) t c')
        context = rearrange(context, 'b c t h w -> (b h w) t c') if exists(context) else None
        x = self.attn_t(query=x, 
                        key=context if exists(context) else None,
                        pos_bias = None)
        x = rearrange(x, '(b h w) t c -> b c t h w', h=H, w=W)
        context = rearrange(context, '(b h w) t c -> b c t h w', h=H, w=W) if exists(context) else None
        
        return x
    

class CrossFrameAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout=0.,
    ):
        super().__init__()
        
        self.to_q = nn.Linear(dim, dim_head * heads)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x, context=None):
        B, N, C = x.shape
            
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        
        return self.to_out(out)
    

class CrossFrameAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        frame_dim=256,
        heads=8,
        dim_head=32,
        dropout=0.,
        mlp_ratio=4.,
        cond_frames=4,
        pred_frames=4,
        use_attn=False
    ):
        super().__init__()
        self.cf = cond_frames
        self.pf = pred_frames
        self.use_attn = use_attn
        
        self.norm_q = Normalization(dim, norm_type='layer')
        # self.norm_kv = Normalization(dim, norm_type='layer')
        self.norm_mlp = Normalization(dim, norm_type='layer')
        
        self.cross_attn = CrossFrameAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # self.frame_emb_mlp = nn.Sequential(
        #     nn.Linear(frame_dim, dim),
        #     nn.GELU(),
        #     nn.Linear(dim, dim)
        # )
        # self.action_emb_mlp = nn.Sequential(
        #     nn.Linear(frame_dim, dim),
        #     nn.GELU(),
        #     nn.Linear(dim, dim)
        # )
        self.fusion = Mlp(in_features=dim+1, hidden_features=int(mlp_ratio*dim), out_features=dim)
        
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        
    def forward(self, x, frame_idx=None, action_class=None):
        """_Args_
            x : [B, C, T, H, W]
            frame_idx : [B, T]
            action_class : [B, C]
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        
        frame_idx = frame_idx.repeat(B, 1)
        frame_emb = repeat(frame_idx, 'b t -> (b t) (h w) 1', t=T, h=H, w=W)
        
        x_shape = x.shape
        frame_emb_shape = frame_emb.shape
        
        x = torch.cat([x, frame_emb], dim=2) # [BT, HW, C+2]
        x = self.fusion(x) # [BT, HW, C]
        x = rearrange(x, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)
        
        # if self.use_attn:
        #     cond, pred = x[:, :, :self.cf], x[:, :, self.cf:]
            
        #     cond = rearrange(cond, 'b c t h w -> b (h w t) c')
        #     pred = rearrange(pred, 'b c t h w -> b (h w t) c')
        #     pred = pred + self.cross_attn(self.norm_q(pred), context=self.norm_kv(cond))
        #     pred = pred + self.mlp(self.norm_mlp(pred))
        #     cond = rearrange(cond, 'b (h w t) c -> b c t h w', h=H, w=W)
        #     pred = rearrange(pred, 'b (h w t) c -> b c t h w', h=H, w=W)
            
        #     x = torch.cat([cond, pred], dim=2)
            
        if self.use_attn:
            x = rearrange(x, 'b c t h w -> b (h w t) c')
            x = x + self.cross_attn(self.norm_q(x))
            x = x + self.mlp(self.norm_mlp(x))
            x = rearrange(x, 'b (h w t) c -> b c t h w', h=H, w=W)
        
        return x
        
        
class CrossCondAttention(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        heads = 8,
        dim_head = 32,
        dropout=0.,
    ):
        super().__init__()
                
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, cond):
        B, N, C = x.shape
            
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        
        return self.to_out(out)        
        
class CrossCondAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        heads=8,
        dim_head=32,
        dropout=0.,
        mlp_ratio=4.,
        use_attn=False

    ):
        super().__init__()
        self.use_attn = use_attn

        self.norm_q = Normalization(dim, norm_type='layer')
        self.norm_kv = Normalization(cond_dim, norm_type='layer')
        self.norm_mlp = Normalization(dim, norm_type='layer')
        
        self.cross_attn = CrossCondAttention(dim, cond_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        
    def forward(self, x, motion_cond):
        """_Args_
            x : [B, C, T, H, W]
            motion_cond : [B, T, PN, C] 
        """
        B, C, T, H, W = x.shape

        if self.use_attn:
            x = rearrange(x, 'b c t h w -> (b t) (h w) c')
            mc = rearrange(motion_cond, 'b t pn c -> (b t) pn c')

            x = x + self.cross_attn(self.norm_q(x), self.norm_kv(mc))
            x = x + self.mlp(self.norm_mlp(x))
            x = rearrange(x, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)
            
        return x                

class CondAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout=0.,
    ):
        super().__init__()
                
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, q, kv):
        B, N, C = q.shape 
            
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        
        return self.to_out(out)   
        
class CondAttentionLayer(nn.Module):
    def __init__(
        self,
        tc_dim,
        pn_prime,
        heads=8,
        dim_head=32,
        dropout=0.,
        mlp_ratio=4.,
        last_attn=False
    ):
        super().__init__()
        self.last_attn = last_attn
        
        self.pn_prime = pn_prime
        self.tc_dim = tc_dim
        self.query = nn.Parameter(torch.randn(self.pn_prime, tc_dim))

        self.norm_q = Normalization(tc_dim, norm_type='layer')
        self.norm_kv = Normalization(tc_dim, norm_type='layer')
        self.norm_mlp = Normalization(tc_dim, norm_type='layer')
        
        self.attn = CondAttention(tc_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = Mlp(in_features=tc_dim, hidden_features=int(mlp_ratio*tc_dim), out_features=tc_dim)
        
    def forward(self, x, q=None):
        """_Args_
            param
                x : [B, C, T, PN] 
            return
                out:[B, T, PN, C]
        """
        B, C, T, N = x.shape
        assert self.tc_dim == T*C

        if q is not None:
            q = rearrange(q, 'b c t pn -> b pn (t c)')
        else:
            q = self.query.unsqueeze(0).repeat(B, 1, 1) # [b, pn', tc]
            
        kv = rearrange(x, 'b c t pn -> b pn (t c)') 

 
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv)) # [b, pn', tc]
        q = q + self.mlp(self.norm_mlp(q))
        if not self.last_attn:
            q = rearrange(q, 'b pn2 (t c) -> b c t pn2', t=T, c=C)
        else:
            q = rearrange(q, 'b pn2 (t c) -> b t pn2 c', t=T, c=C)
            
        return q