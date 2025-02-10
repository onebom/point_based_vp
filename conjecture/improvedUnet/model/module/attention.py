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
        
# class TemporalAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         heads = 4,
#         dim_head = 32,
#         rotary_emb = None
#     ):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.hidden_dim = hidden_dim

#         self.rotary_emb = rotary_emb
#         self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
#         self.to_out = nn.Linear(hidden_dim, dim, bias = False)

#     def forward(
#         self,
#         x,
#         pos_bias = None,
#     ):
#         b, n, device = x.shape[0], x.shape[-2], x.device

#         qkv = self.to_qkv(x).chunk(3, dim = -1)
        
#         # split out heads
#         q, k, v = rearrange_many(qkv, 'b m n (h d) -> (b m) h n d', h = self.heads)

#         # scale
#         q = q * self.scale

#         # rotate positions into queries and keys for time attention
#         if exists(self.rotary_emb):
#             q = self.rotary_emb.rotate_queries_or_keys(q)
#             k = self.rotary_emb.rotate_queries_or_keys(k)

#         # similarity
#         sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

#         # relative positional bias
#         if exists(pos_bias):  
#             mul = sim.shape[0] // pos_bias.shape[0]
#             sim = sim + pos_bias.repeat(mul, 1, 1, 1)

#         # numerical stability
#         sim = sim - sim.amax(dim = -1, keepdim = True).detach()
#         attn = sim.softmax(dim = -1)

#         # aggregate values

#         out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
#         out = rearrange(out, '(b m) h n d -> b m n (h d)', b=b)
#         return self.to_out(out)


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
        
        x = torch.cat([x, frame_emb], dim=2) # [BT, HW, C+1]
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

class CondAttentionTemporalModule(nn.Module):
    def __init__(
        self,
        dim,
        heads= 4,
        dim_head = 32,
        attn_dim_head = 32,
        use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        
        self.attn = TemporalAttention(dim, 
                                      heads=heads,
                                      dim_head=dim_head, 
                                      rotary_emb=rotary_emb)
        
        self.attn_cross = TemporalAttention(dim,
                                            cond_dim=dim,
                                            heads=heads,
                                            dim_head=dim_head, 
                                            rotary_emb=rotary_emb)
        
    def forward(self, x, motion_map, pos_bias=None):
        b, c, t, h, w = x.shape

        if self.use_attn:  
            x = rearrange(x, 'b c t h w -> b (h w) t c')
            
            # self temporal attention
            x = x + self.attn(x, pos_bias=pos_bias)
            
            # cross temporal attention
            if motion_map is not None:   
                b, c2, t, h2, w2 = motion_map.shape
                assert h==h2 and w==w2
                         
                mc = rearrange(motion_map, 'b c t h w -> b (h w) t c')
                x = x + self.attn_cross(x, mc, pos_bias = pos_bias)
            
            x = rearrange(x, 'b (h w) t c -> b c t h w', h= h, w=w)
        return x

class AttentionSTModule(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio = 4.,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        use_attn=False
        ):
        super().__init__()
        
        self.use_attn = use_attn
        
        self.fusion = Mlp(in_features=dim+1, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.self_attn = SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
    
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.norm = Normalization(dim, norm_type='layer')
        
    
    def forward(self, x, frame_idx=None):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        
        frame_idx = frame_idx.repeat(B, 1)
        frame_emb = repeat(frame_idx, 'b t -> (b t) (h w) 1', t=T, h=H, w=W)
        
        x = torch.cat([x, frame_emb], dim=2) # [BT, HW, C+2]
        x = self.fusion(x) # [BT, HW, C]
        x = rearrange(x, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)
            
        if self.use_attn:
            x = rearrange(x, 'b c t h w -> b (h w t) c')
            x = x + self.self_attn(x)
            x = x + self.mlp(self.norm(x))
            x = rearrange(x, 'b (h w t) c -> b c t h w', h=H, w=W)
        
        return x

# =================================================================
class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.attn = TemporalAttention(dim, 
                                      heads=heads, 
                                      dim_head=dim_head, 
                                      rotary_emb=rotary_emb)

    def forward(self, x, pos_bias=None):
        x = x + self.attn(x, pos_bias = pos_bias)
        return x

class CondAttention2DModule(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        mlp_ratio = 4.,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        use_attn=False
        ):
        super().__init__()
        self.use_attn = use_attn
        
        self.fusion = Mlp(in_features=dim+1, hidden_features=int(mlp_ratio*dim), out_features=dim)
        
        self.self_attn = SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.norm = Normalization(dim, norm_type='layer')

        cond_dim = cond_dim if cond_dim else dim
        self.cross_attn = SpatialAttention(dim, cond_dim=cond_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.cross_mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.cross_norm = Normalization(dim, norm_type='layer')
                    
    def forward(self, x, motion_cond=None, frame_idx=None):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        
        frame_idx = frame_idx.repeat(B, 1)
        frame_emb = repeat(frame_idx, 'b t -> (b t) (h w) 1', t=T, h=H, w=W)
        
        x = torch.cat([x, frame_emb], dim=2)
        x = self.fusion(x)

        if self.use_attn: #[32, 16]
            # 1. self-attention
            x = x + self.self_attn(x)
            x = x + self.mlp(self.norm(x))
            
            if motion_cond is not None:
                B, C_p, T, PN = motion_cond.shape
                # 2. cross-attention
                mc = rearrange(motion_cond, 'b c t pn -> (b t) pn c')
                x = x + self.cross_attn(x, mc)
                x = x + self.cross_mlp(self.cross_norm(x))
        
        x = rearrange(x, '(b t) (h w) c -> b c t h w', t=T, h=H, w=W)

        return x

class CondAttentionSTModule(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        mlp_ratio = 4.,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        use_attn=False
        ):
        super().__init__()
        self.use_attn = use_attn
        
        self.fusion = Mlp(in_features=dim+1, hidden_features=int(mlp_ratio*dim), out_features=dim)
        
        self.self_attn = SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.norm = Normalization(dim, norm_type='layer')
        
        cond_dim = cond_dim if cond_dim else dim
        self.cross_attn = SpatialAttention(dim, cond_dim=cond_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.cross_mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.cross_norm = Normalization(dim, norm_type='layer')
        
    def forward(self, x, motion_cond=None, frame_idx=None):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        
        frame_idx = frame_idx.repeat(B, 1)
        frame_emb = repeat(frame_idx, 'b t -> (b t) (h w) 1', t=T, h=H, w=W)
        
        x = torch.cat([x, frame_emb], dim=2) # [BT, HW, C+2]
        x = self.fusion(x) # [BT, HW, C]
        x = rearrange(x, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)
            
        if self.use_attn:
            x = rearrange(x, 'b c t h w -> b (h w t) c')
            x = x + self.self_attn(x)
            x = x + self.mlp(self.norm(x))
            
            if motion_cond is not None:
                B, C_p, T, PN = motion_cond.shape
                mc = rearrange(motion_cond, 'b c t pn -> b (pn t) c')
                x = x + self.cross_attn(x, mc)
                x = x + self.cross_mlp(self.cross_norm(x))
            
            x = rearrange(x, 'b (h w t) c -> b c t h w', h=H, w=W)
        
        return x


class SpatialAttention(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        heads = 8,
        dim_head = 32,
        dropout = 0.
        ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.norm = Normalization(dim, norm_type='layer')
        self.cond_norm = Normalization(cond_dim, norm_type='layer') if cond_dim else None
        
        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        
        kv_dim = cond_dim if cond_dim else dim
        self.to_k = nn.Linear(kv_dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(kv_dim, hidden_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, cond = None):
        B, N, C = x.shape
        
        x = self.norm(x)
        if exists(self.cond_norm):
            cond = self.cond_norm(cond)
        context = default(cond, x) # cond==None: x
        
        q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        
        return self.to_out(out)

class TemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
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
        
        self.norm = Normalization(dim, norm_type='layer')
        self.cond_norm = Normalization(cond_dim, norm_type='layer') if cond_dim else None
        
        self.to_q = nn.Linear(dim, hidden_dim , bias = False)
        
        kv_dim = cond_dim if cond_dim else dim
        self.to_k = nn.Linear(kv_dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(kv_dim, hidden_dim, bias = False)
        
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        cond = None,
        pos_bias = None,
    ):
        b, n, device = x.shape[0], x.shape[-2], x.device

        x = self.norm(x)
        if exists(self.cond_norm):
            cond = self.cond_norm(cond)
        context = default(cond, x)
        
        q,k,v = self.to_q(x), self.to_k(context), self.to_v(context) # b, 32*32, 6, 128 
        
        # split out heads
        q, k, v = map(lambda t: rearrange(t, 'b m n (h d) -> (b m) h n d', h=self.heads), (q, k, v))

        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q) #8192,4,6,32
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias
        if exists(pos_bias):  
            mul = sim.shape[0] // pos_bias.shape[0]
            sim = sim + pos_bias.repeat(mul, 1, 1, 1) #8192,4,6,6

        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '(b m) h n d -> b m n (h d)', b=b)
        return self.to_out(out)
    
class MotionQEncoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        pn_prime,
        track_dim,
        lq_attn_num=4,
        heads=8,
        dim_head=32,
        dropout=0.,
        mlp_ratio=4.0,
        use_attn=False
    ):
        super().__init__()
        self.use_attn = use_attn        
        self.query = nn.Parameter(torch.randn(pn_prime, dim))
        
        # 1. Learnable motion query attention 
        self.cond_attn = nn.ModuleList([])
        for _ in range(lq_attn_num):
            self.cond_attn.append(nn.ModuleList([
                SpatialAttention(dim, cond_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Normalization(dim, norm_type='layer'), # norm_mlp
                Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
            ]))
        
        # 2. motion encoder
        self.motion_encoder = MotionEncoder(in_dim=track_dim, out_dim=dim_out)
    
    def forward(self, mc):
        B, C, T, N = mc.shape
        encoded_mc=None
        if self.use_attn:         
            q = self.query.unsqueeze(0).repeat(B, 1, 1) # [b, pn', tc]
            kv = rearrange(mc, 'b c t pn -> b pn (t c)')
            
            for attn, norm_mlp, mlp in self.cond_attn:
                q = q + attn(q, kv)
                q = q + mlp(norm_mlp(q))

            mc = rearrange(q, 'b pn2 (t c) -> b c t pn2', t=T, c=C)
            
            encoded_mc = rearrange(q, 'b pn2 (t c) -> b c (t pn2)', t=T, c=C)
            encoded_mc = self.motion_encoder(encoded_mc)
            encoded_mc = rearrange(encoded_mc, 'b c (t pn2) -> b c t pn2', t=T)
            
        return encoded_mc, mc

class MotionEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        layer_num=2
    ):
        super().__init__()
        ch = out_dim//layer_num
        
        layers = []
        layers.append(nn.Conv1d(in_dim, ch, 1))
        layers.append(nn.ReLU())

        while ch < out_dim:
            next_ch = min(ch * 2, out_dim) 
            layers.append(nn.Conv1d(ch, next_ch, 1))
            layers.append(nn.ReLU())
            ch = next_ch
        
        assert ch == out_dim
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, mc):
        out = self.model(mc)
        return out


