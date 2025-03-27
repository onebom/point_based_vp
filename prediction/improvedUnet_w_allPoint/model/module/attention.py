import math
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from model.util import exists, default
from model.module.normalization import Normalization
# from .block import Mlp
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

# =================================================================
class Attention(nn.Module):
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

class AttentionModule(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        shape=None,
        mlp_ratio = 4.,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        ):
        super().__init__()
        self.shape = shape
        
        self.fusion = Mlp(in_features=dim+1, hidden_features=int(mlp_ratio*dim), out_features=dim)

        self.attn = Attention(dim, cond_dim=cond_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.norm = Normalization(dim, norm_type='layer')
        
    def forward(self, x, cond=None, frame_idx=None):
        B, C, T, TK = x.shape
        
        if frame_idx is not None:
            x = rearrange(x, 'b c t tk -> (b t) tk c')
            
            frame_idx = frame_idx.repeat(B, 1)
            frame_emb = repeat(frame_idx, 'b t -> (b t) tk 1', tk=TK)
            
            x = torch.cat([x, frame_emb], dim=2) # [BT, HW, C+2]
            x = self.fusion(x) # [BT, HW, C]
            x = rearrange(x, '(b t) tk c -> b c t tk', b=B)
            
        x = rearrange(x, f'b c t tk -> {self.shape}')
        if cond is not None:
            cond = rearrange(cond, f'b c t tk -> {self.shape}')

        x = x + self.attn(x, cond)
        x = x + self.mlp(self.norm(x))
                
        x = rearrange(x, f'{self.shape} -> b c t tk', tk=TK, t=T)
        
        return x

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

class BiCrossAttentionModule(nn.Module):
    def __init__(
        self,
        dim,
        shape = None,
        mlp_ratio = 4.,
        heads = 8,
        dim_head = 32,
        dropout = 0.
        ):
        super().__init__()
        self.shape = shape
        
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.norm = Normalization(dim, norm_type='layer')
        
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
        self.norm2 = Normalization(dim, norm_type='layer')
        
        ####
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
    
        self.q_norm = Normalization(dim, norm_type='layer')
        self.k_norm = Normalization(dim, norm_type='layer')
        
        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        self.to_qv = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim, hidden_dim, bias = False)
        
        self.to_q_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.to_k_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def attention(self, q, k):
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q,k,qv,kv = self.to_q(q), self.to_k(k), self.to_qv(q), self.to_kv(k)
        
        q, k, qv, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, qv, kv))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        q_attn = sim.softmax(dim=-1)
        k_attn = sim.transpose(1, 2).softmax(dim=-1)
        
        q_out = einsum('b i j, b j d -> b i d', q_attn, kv)
        q_out = rearrange(q_out, '(b h) n d -> b n (h d)', h=self.heads)
        
        k_out = einsum('b i j, b j d -> b i d', k_attn, qv)
        k_out = rearrange(k_out, '(b h) n d -> b n (h d)', h=self.heads)
        
        return self.to_q_out(q_out), self.to_k_out(k_out)
    
    def forward(self, x1, x2):
        B, C, T, TK = x1.shape
        
        x1, x2 = map(lambda x: rearrange(x, f'b c t tk -> {self.shape}'), (x1, x2))
        q_x1, q_x2 = self.attention(x1, x2)
        
        x1 = x1 + q_x1
        x2 = x2 + q_x2

        x1 = x1 + self.mlp(self.norm(x1))
        x2 = x2 + self.mlp2(self.norm2(x2))

        x1, x2 = map(lambda x: rearrange(x, f'{self.shape} -> b c t tk', t=T), (x1, x2))
        
        return x1, x2
