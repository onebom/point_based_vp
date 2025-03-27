import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from model.module.normalization import Normalization
from .normalization import Normalization

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
                 stride,
                 padding, 
                 groups=8, 
                 motion_dim=None, 
                 dropout_rate=0.0):
        super().__init__()
        spade = True if exists(motion_dim) else False
        
        if stride is None:
            self.conv = nn.Conv3d(dim, dim_out, kernel, padding = padding)
        else:
            self.conv = nn.Conv3d(dim, dim_out, kernel, stride=stride, padding = padding)
            
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
                 use_res = True, 
                 time_emb_dim = None, 
                 groups=8, 
                 motion_dim=None, 
                 dropout_rate=0.0,
                 kernel=None,
                 stride=None,
                 padding=None,
                 ):
        super().__init__()
        self.use_res = use_res
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        if kernel is None or padding is None:
            kernel, padding = ((1,3,3), (0,1,1)) if conv_method == "2d" else ((3,1,1), (1,0,0))

        self.block1 = Block(dim, dim_out, kernel, stride, padding, 
                            groups=groups, motion_dim=motion_dim, dropout_rate=dropout_rate)
        self.block2 = Block(dim_out, dim_out, kernel, stride, padding, 
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
            
            res_x = self.res_conv(x)
            if res_x.shape==h.shape:
                h = h + res_x
        
        return h


#-------model---------
from einops import rearrange

# from torch.distributions.multivariate_normal import MultivariateNormal

import torch_cluster
from torch_geometric.nn import knn_graph, GATConv
from rotary_embedding_torch import RotaryEmbedding

# from model.util import exists,EinopsToAndFrom
# from model.module.attention import TemporalAttentionLayer, AttentionModule, CrossAttentionModule
# from model.module.normalization import Normalization

from .util import exists,EinopsToAndFrom
from .attention import TemporalAttentionLayer, AttentionModule, CrossAttentionModule
from .normalization import Normalization

class LinearEmbedding(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dropout_rate=0.
    ):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(dim_in, dim_out),
                                   nn.LayerNorm(dim_out),
                                   nn.ReLU(inplace=True), 
                                   nn.Dropout(dropout_rate),
                                   nn.Linear(dim_out, dim_out),
                                   nn.LayerNorm(dim_out),
                                   nn.Dropout(dropout_rate)
                                       )
        
    def forward(self, x):
        return self.embed(x)
                
class InitialBlock(nn.Module):
    def __init__(
        self,
        vid_ch,
        trj_ch,
        fea_dim,
        dim,
        kernel_size = 7,
        attn_heads = 8,
        attn_dim_head = 32,
    ):
        super().__init__()
        
        # video
        padding = kernel_size // 2
        self.vid_conv = nn.Conv3d(vid_ch, dim, (1, kernel_size, kernel_size), padding = (0, padding, padding))     
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.vid_temp_attn = EinopsToAndFrom('b c f h w', 'b (h w) f c', 
                                             TemporalAttentionLayer(
                                                 dim, heads = attn_heads,
                                                 dim_head = attn_dim_head,
                                                 rotary_emb = rotary_emb
                                                 )
                                             )
        
        # traj
        self.coord_embedding = LinearEmbedding(trj_ch, fea_dim)
        self.trj_embedding = LinearEmbedding(fea_dim*2, dim)

    def forward(self, vid, trj):
        """
        vid: (b,c,t,h,w)
        trj: [(b,c(3:x,y,vis),t,pn), (b,c(128),t,pn)]
        """
        B,C,T,H,W = vid.shape
        # 1. initial video block
        vid = self.vid_conv(vid)
        vid = self.vid_temp_attn(vid)
        
        # 2. initial trj block
        trj_c, trj_f = trj["coord"], trj["track_f"]
        
        trj_c = rearrange(trj_c, 'b c t pn -> b (t pn) c')
        trj_c = self.coord_embedding(trj_c)
        trj_c = rearrange(trj_c, 'b (t pn) c -> b c t pn', t=T)
        trj = torch.concat([trj_c, trj_f], dim=1)
        
        trj = rearrange(trj, 'b c t pn -> b (t pn) c')
        trj = self.trj_embedding(trj)
        trj = rearrange(trj, 'b (t pn) c -> b c t pn', t=T)
        
        return vid, trj

class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        time_emb_dim = None,
        resnet_groups = None
    ):
        super().__init__()

        self.t_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_in)
        ) if exists(time_emb_dim) else None

        self.vid_block2D = ConvBlock(dim_in, dim_out, 
                                     time_emb_dim = time_emb_dim, 
                                     conv_method="2d", 
                                     groups=resnet_groups)
        
        self.vid_blockTemporal = ConvBlock(dim_out, dim_out, 
                                           conv_method="temporal", 
                                           groups=resnet_groups, 
                                           dropout_rate = 0.1)
        
        self.trj_block = LinearEmbedding(dim_in, dim_out, 
                                         dropout_rate= 0.1)

        
    def forward(self, vid, trj, emb):
        """
        vid: (b,c,t,h,w)
        trj: (b,c,t,pn)
        """
        B,C,T,H,W = vid.shape
        
        t = emb["t"]
        
        # 1. vid block 
        vid = self.vid_block2D(vid, t)
        vid = self.vid_blockTemporal(vid)
        
        # 2. trj block
        # b,c,t,pn + b,c
        trj = trj + self.t_mlp(t)[:,:,None,None] # scale shift로 바꾸기
        
        trj = rearrange(trj, 'b c t pn -> b (t pn) c')
        trj = self.trj_block(trj)
        trj = rearrange(trj, 'b (t pn) c -> b c t pn', t=T)
        
        return vid, trj

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        frame_num,
        use_attn,
        attn_heads = 8,
        attn_dim_head = 32
    ):
        super().__init__()
        self.use_attn = use_attn
        
        if self.use_attn:
            self.trj_attn = AttentionModule(dim*frame_num, shape = "b tk (t c)")
            self.vid_attnST = AttentionModule(dim, shape = "b (tk t) c")
            
            self.attnST_cross = CrossAttentionModule(dim, shape = "b (tk t) c")
            
            self.trj_attn2D = AttentionModule(dim, shape = "(b t) tk c")
            self.vid_attn2D = AttentionModule(dim, shape = "(b t) tk c")
            
            self.attn2D_cross = CrossAttentionModule(dim, shape = "(b t) tk c")
            
        
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.vid_attnTemporal = EinopsToAndFrom('b c t h w', 'b (h w) t c', 
                                             TemporalAttentionLayer(
                                                 dim, heads = attn_heads,
                                                 dim_head = attn_dim_head,
                                                 rotary_emb = rotary_emb
                                                 )
                                             )
        self.trj_attnTemporal= EinopsToAndFrom('b c t pn', 'b pn t c', 
                                             TemporalAttentionLayer(
                                                 dim, heads = attn_heads,
                                                 dim_head = attn_dim_head,
                                                 rotary_emb = rotary_emb
                                                 )
                                             )
        
    def forward(self, vid, trj, emb):
        B, C, T, H, W = vid.shape
        
        frame_idx = emb["frame_idx"]
        time_rel_pos_bias = emb["time_rel_pos_bias"]
        
        if self.use_attn:
            vid = rearrange(vid, 'b c t h w -> b c t (h w)')
            x = self.vid_attnST(vid, frame_idx = frame_idx)
            m = self.trj_attn(trj)
            
            vid, trj = self.attnST_cross(x, m)
                
            x = self.vid_attn2D(vid, frame_idx = frame_idx)        
            m = self.trj_attn2D(trj, frame_idx = frame_idx)
            
            vid, trj = self.attn2D_cross(x, m)
            
            vid = rearrange(vid, 'b c t (h w) -> b c t h w', h=H, w=W)

        vid = self.vid_attnTemporal(vid, pos_bias=time_rel_pos_bias)
        trj = self.trj_attnTemporal(trj, pos_bias=time_rel_pos_bias)
        
        return vid, trj

class PointScaling(nn.Module):
    def __init__(
        self,
        dim,
        reduced_ratio = 0.25,
        k = 8,
        heads=2,
        mlp_ratio = 4.,
        isDown = True
        ):
        super().__init__()
        self.k = k
        self.reduced_ratio = reduced_ratio
        
        if isDown:
            self.down_mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
            self.down_norm = Normalization(dim, norm_type='layer')
        else:
            self.up_mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
            self.up_norm = Normalization(dim, norm_type='layer')
            
            self.out_mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), out_features=dim)
            self.out_norm = Normalization(dim, norm_type='layer')
        
        self.gat = GATConv(dim, dim // heads, heads=heads, concat=True)
    
    def grouping(self, points, reduced_pn):
        B,PN,T,C = points.shape
        
        motion_magnitude = (points[:, :, 1:] - points[:, :, :-1]).norm(dim=-1).mean(dim=-1) #[B, PN]
        motion_weights = torch.softmax(motion_magnitude, dim=1)  # [B, PN]
        
        centroid_indices = torch.multinomial(motion_weights, reduced_pn, replacement=False)  # [B, reduced_pn]
        centroids = torch.gather(points, 1, centroid_indices[:,:,None,None].expand(-1, -1, T, C)) # [B, reduced_pn, T, C]
        
        distances = torch.cdist(points.view(B, PN, -1), centroids.view(B, reduced_pn, -1)) # [B, PN, reduced_pn]
        min_indices = torch.argmin(distances, dim=-1) # [B, PN]
        
        soft_group_weights = F.one_hot(min_indices, num_classes=reduced_pn).float()  # [B, PN, reduced_pn]
        
        return soft_group_weights
    
    def downsample(self, x):
        B,C,T,PN = x.shape
        x = rearrange(x, 'b c t pn -> b pn (t c)')
        x = self.down_mlp(self.down_norm(x))
        
        # KNN to find nearest neighbors in batch mode
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(PN)
        node = x.view(-1, T*C) #[B*PN, T*C]
        edge_index = knn_graph(node, k=self.k, batch=batch_idx) #[2, B*PN*k]
        
        # GAT to update point features based on neighbors with motion-aware attention
        # src, dst = edge_index
        # motion_sim = torch.exp(-torch.norm(node[src] - node[dst], dim=1))
        updated_node = self.gat(node, edge_index) #[B*PN, T*C]
        
        updated_node = updated_node.view(B, PN, T, C)
        
        reduced_pn = int(PN * self.reduced_ratio)
        soft_group_weights = self.grouping(updated_node, reduced_pn)
        
        reduced_points = torch.einsum('bpr,bptd->brtd', soft_group_weights, updated_node)  # [B, reduced_pn, T, C]
       
        x = rearrange(reduced_points, 'b pn t c -> b c t pn')
        return x, soft_group_weights
    
    def upsample(self, x, h_x, group):
        # group: [B, origin_pn, PN]
        B,C,T,PN = x.shape
        
        x = rearrange(x, 'b c t pn -> b pn (t c)')
        x = self.up_mlp(self.up_norm(x))
        
        # 1. Centroid 기반으로 4096개의 포인트에 초기 값 매핑
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(PN)
        node = x.view(-1, T*C)  # [B*PN, T*C]
        edge_index = knn_graph(node, k=self.k, batch=batch_idx)  # [2, B*PN*k]
        
        # src, dst = edge_index
        # motion_sim = torch.exp(-torch.norm(node[src] - node[dst], dim=1))
        updated_node = self.gat(node, edge_index) # [B*PN, T*C]
        
        updated_node = updated_node.view(B, PN, T, C)

        upsampled_x = torch.einsum('brtd,bpr->bptd', updated_node, group) # [B, origin_pn, T, C]
        
        upsampled_x = rearrange(upsampled_x, 'b pn t c -> b pn (t c)')
        final_x = self.out_mlp(self.out_norm(upsampled_x))   # Residual 연결
        final_x = rearrange(final_x, 'b pn (t c) -> b c t pn', t=T) + h_x
        
        return final_x
    

class ScalingBlock(nn.Module):
    def __init__(
        self,
        dim,
        frame_num,
        sample_ratio = 0.25,
        isDown=True
        ):
        super().__init__()
        
        if isDown:
            self.vid_down = nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        else:
            self.vid_up = nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        
        self.trj_scale = PointScaling(dim*frame_num, reduced_ratio=sample_ratio, isDown=isDown)
    
    def down(self, vid, trj):
        """
        :param vid: (B, C, T, H, W)
        :param trj: (B, C, T, PN)
        """
        vid = self.vid_down(vid)
        trj, trj_cluster = self.trj_scale.downsample(trj)
        
        return vid, (trj, trj_cluster)
    
    def up(self, vid, trj, prev_trj, trj_group):
        vid = self.vid_up(vid)
        trj = self.trj_scale.upsample(trj, prev_trj, trj_group)
        
        return vid, trj

class OutBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        vid_dim_out,
        trj_dim_out,
        resnet_groups = None
    ):
        super().__init__()

        self.vid_final_conv = nn.ModuleList([ConvBlock(dim_in, dim_in//2,
                                                       conv_method="2d",
                                                       groups=resnet_groups),
                                             nn.Conv3d(dim_in//2, vid_dim_out, 1)])

        self.trj_final_block = nn.ModuleList([LinearEmbedding(dim_in, dim_in//2, dropout_rate= 0.1),
                                              LinearEmbedding(dim_in//2, trj_dim_out, dropout_rate= 0.1)])

        
    def forward(self, vid, trj):
        """
        vid: (b,c,t,h,w)
        trj: (b,c,t,pn)
        """
        # 1. vid block 
        for v_layer in self.vid_final_conv:
            vid = v_layer(vid)
        
        # 2. trj block
        B, C, T, PN = trj.shape
        trj = rearrange(trj, 'b c t pn -> b (t pn) c')    
        for t_layer in self.trj_final_block:
            trj = t_layer(trj)
        trj = rearrange(trj, 'b (t pn) c -> b c t pn', t=T)
        
        return vid, trj
        