import math
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

from einops import rearrange

from rotary_embedding_torch import RotaryEmbedding

from model.util import exists, default, EinopsToAndFrom, Residual, PreNorm, temporal_distance_to_frame_idx

from model.module.attention import (TemporalAttentionLayer, 
                                    STWAttentionLayer, RelativePositionBias, 
                                    SinusoidalPosEmb, CrossFrameAttentionLayer,
                                    CrossCondAttentionLayer, CondAttentionLayer)
from model.module.block import ResnetBlock, Downsample, Upsample
from model.module.condition import ClassCondition
from model.module.motion_module import AttentionPointSelector

class DirectUnet3D_CrossFrameAttn(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        motion_dim = None,
        out_dim = None,
        window_size = (2, 4, 4),
        dim_mults=(1, 2, 4, 8),
        attn_res = (32, 16, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        resnet_groups=8,
        frame_size = 64,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        learn_null_cond = False,
        use_final_activation = False,
        use_deconv=True,
        padding_mode="zeros",
        cond_num = 0,
        pred_num = 0,
        num_action_classes=None,
        nf=128
    ):
        self.tc = cond_num
        self.tp = pred_num
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.num_action_classes = num_action_classes
        self.nf = nf
        
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', 
                                                    TemporalAttentionLayer(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv
        init_dim = default(init_dim, dim)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning
        time_dim = self.nf * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.nf, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # if exists(self.num_action_classes):
        #     self.class_cond_mlp = ClassCondition(self.num_action_classes, time_dim)

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        
        self.learn_null_cond = learn_null_cond
        if self.learn_null_cond:
            self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        else:
            self.null_cond_emb = torch.zeros(1, cond_dim).cuda() if self.has_cond else None
        cond_dim = time_dim + int(cond_dim or 0)
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)
        
        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(ResnetBlock, groups=resnet_groups, time_emb_dim = cond_dim, motion_dim=motion_dim,)
        
        now_res = frame_size

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res)
      
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                CrossFrameAttentionLayer(dim_out, use_attn=use_attn),
                block_klass_cond(dim_out, dim_out),
                CrossFrameAttentionLayer(dim_out, use_attn=use_attn),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
            
            if not is_last:
                now_res = now_res // 2

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn1 = CrossFrameAttentionLayer(mid_dim, use_attn=use_attn)
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn2 = CrossFrameAttentionLayer(mid_dim, use_attn=use_attn)
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res)
                        
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out*2, dim_in),
                CrossFrameAttentionLayer(dim_in, use_attn=use_attn),
                block_klass_cond(dim_in, dim_in),
                CrossFrameAttentionLayer(dim_in, use_attn=use_attn),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in, use_deconv, padding_mode) if not is_last else nn.Identity()
            ]))
            if not is_last:
                now_res = now_res * 2

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim*2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
        
        if use_final_activation:
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

    def get_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
        half_dim = embedding_dim // 2
        # magic number 10000 is from transformers
        emb = math.log(max_positions) / (half_dim - 1)
        # emb = math.log(2.) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
        # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb
            
    def forward(
        self,
        x,
        time,
        cond_frames,
        cond = None
    ):
        B, C, T, H, W, device = *x.shape, x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        
        assert tc == self.tc
        assert tp == self.tp
        
        x = torch.cat([cond_frames, x], dim=2) # [B, C, 2T, H, W]
        # if motion_cond is not None:
        #     x = torch.cat([x, motion_cond], dim=1)
        
        ### 0. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(tc+tp, device = x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(tc+tp, device = x.device, frame_idx=frame_idx)

        ### 1. initial convolution & temporal attention
        ###### temporal attention using frameDistance embeding
        x = self.init_conv(x) # [B, C', T, H, W]
        r = x.clone() # for final conv layer
        
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        ### 2. embedding timestemp
        ##### didnt use action class in my model
        time = self.get_timestep_embedding(time, self.nf)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # if exists(cond) and exists(self.num_action_classes):  
        #     c = F.one_hot(cond, self.num_action_classes).type(torch.float)
        #     action_emb =  self.class_cond_mlp(c) # time_emb + cond_emb
        #     t += action_emb


        ### 4. prediction
        ####### conditioning motion feature by SPADE in Resblock.normalization layer
        h = []
        ###### 4-1. down layers
        for block1, attn1, block2, attn2, temporal_attn, downsample in self.downs:
            x = block1(x, t, cond)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, cond)
            x = attn2(x, frame_idx=frame_idx, action_class=cond)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)
            
        ###### 4-2. mid layers
        x = self.mid_block1(x, t, cond)
        x = self.mid_attn1(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_block2(x, t, cond)
        x = self.mid_attn2(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        ###### 4-3. up layers
        for block1, attn1, block2, attn2, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, cond)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, cond)
            x = attn2(x, frame_idx=frame_idx, action_class=cond)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)
        
        ###### 4-4 final conv layer
        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,tc:]

        return x_fin
    
class DirectUnet3D_CrossCondAttn(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        motion_dim = None,
        out_dim = None,
        window_size = (2, 4, 4),
        dim_mults=(1, 2, 4, 8),
        attn_res = (32, 16, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        resnet_groups=8,
        frame_size = 64,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        learn_null_cond = False,
        use_final_activation = False,
        use_deconv=True,
        padding_mode="zeros",
        cond_num = 0,
        pred_num = 0,
        num_action_classes=None,
        nf=128,
        spatial_method=None,
        point_num = 0,
        pn_prime = 0,
        track_dim = 0
    ):
        self.tc = cond_num
        self.tp = pred_num
        self.spatial_method = spatial_method
        self.point_num = point_num
        self.pn_prime = pn_prime
        self.track_dim = track_dim
        
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.num_action_classes = num_action_classes
        self.nf = nf
        
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', 
                                                    TemporalAttentionLayer(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv
        init_dim = default(init_dim, dim)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = self.nf * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        if spatial_method == "conv":
            self.spatial_mc_conv = nn.Conv3d(point_num, motion_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        elif spatial_method == "attn":
            self.traj_map_selector = AttentionPointSelector(top_k = motion_dim)
        
        self.cond_attn = nn.ModuleList([]) 
        self.cond_attn.append(nn.ModuleList([
            CondAttentionLayer(tc_dim = (self.tc+self.tp)*self.track_dim, pn_prime = self.pn_prime),
            CondAttentionLayer(tc_dim = (self.tc+self.tp)*self.track_dim, pn_prime = self.pn_prime),
            CondAttentionLayer(tc_dim = (self.tc+self.tp)*self.track_dim, pn_prime = self.pn_prime),
            CondAttentionLayer(tc_dim = (self.tc+self.tp)*self.track_dim, pn_prime = self.pn_prime, last_attn=True)
        ]))
        self.cond_mlp = nn.Sequential(
            nn.Linear(track_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # if exists(self.num_action_classes):
        #     self.class_cond_mlp = ClassCondition(self.num_action_classes, time_dim)

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        
        self.learn_null_cond = learn_null_cond
        if self.learn_null_cond:
            self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        else:
            self.null_cond_emb = torch.zeros(1, cond_dim).cuda() if self.has_cond else None
        cond_dim = time_dim + int(cond_dim or 0)
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)
        
        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(ResnetBlock, groups=resnet_groups, time_emb_dim = cond_dim, motion_dim=motion_dim,)
        
        now_res = frame_size

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res)
      
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                CrossFrameAttentionLayer(dim_out, use_attn=use_attn),
                block_klass_cond(dim_out, dim_out),
                CrossFrameAttentionLayer(dim_out, use_attn=use_attn),
                block_klass_cond(dim_out, dim_out),
                CrossCondAttentionLayer(dim_out, cond_dim=dim, use_attn=use_attn),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
            
            if not is_last:
                now_res = now_res // 2

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn1 = CrossFrameAttentionLayer(mid_dim, use_attn=use_attn)
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn2 = CrossFrameAttentionLayer(mid_dim, use_attn=use_attn)
        self.mid_block3 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn3 = CrossCondAttentionLayer(mid_dim, cond_dim=dim, use_attn=use_attn)
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res)
                        
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out*2, dim_in),
                CrossFrameAttentionLayer(dim_in, use_attn=use_attn),
                block_klass_cond(dim_in, dim_in),
                CrossFrameAttentionLayer(dim_in, use_attn=use_attn),
                block_klass_cond(dim_in, dim_in),
                CrossCondAttentionLayer(dim_in, cond_dim=dim, use_attn=use_attn),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in, use_deconv, padding_mode) if not is_last else nn.Identity()
            ]))
            if not is_last:
                now_res = now_res * 2

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim*2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
        
        if use_final_activation:
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()
            
    def get_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
        half_dim = embedding_dim // 2
        # magic number 10000 is from transformers
        emb = math.log(max_positions) / (half_dim - 1)
        # emb = math.log(2.) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
        # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb
    
    def forward(
        self,
        x,
        time,
        cond_frames,
        cond = None
    ):
        B, C, T, H, W, device = *x.shape, x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        
        assert tc == self.tc
        assert tp == self.tp
        
        x = torch.cat([cond_frames, x], dim=2) # [B, C, 2T, H, W]
        traj_mc = cond["traj"]
        spatial_mc = cond["traj_map"]
        
        ### 0. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(tc+tp, device=x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(tc+tp, device = x.device, frame_idx=frame_idx)

        ### 1. initial convolution & temporal attention
        x = self.init_conv(x) # [B, C', T, H, W]
        r = x.clone() # for final conv layer
                
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        ### 2. embedding timestemp
        ##### didnt use action class in my model
        time = self.get_timestep_embedding(time, self.nf)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # if exists(cond) and exists(self.num_action_classes):  
        #     c = F.one_hot(cond, self.num_action_classes).type(torch.float)
        #     action_emb =  self.class_cond_mlp(c) # time_emb + cond_emb
        #     t += action_emb

        ### 3. extract motion feature
        ##### 3-1. LQ Trajectory encoding
        if exists(self.cond_attn):
            for attn1, attn2, attn3, attn4 in self.cond_attn:
                mc = attn1(traj_mc)
                mc = attn2(traj_mc, mc)
                mc = attn3(traj_mc, mc)
                mc = attn4(traj_mc, mc)
        else:
            mc = None 
        mc = self.cond_mlp(mc) if exists(self.cond_mlp) else None 

        ##### 3-2. spatial_mc Selector
        if self.spatial_method == "conv":
            spatial_mc = self.spatial_mc_conv(spatial_mc)
        elif self.spatial_method == "attn":
            spatial_mc = self.traj_map_selector(traj_mc, spatial_mc)

        ### 4. prediction
        ####### conditioning motion feature by SPADE in Resblock.normalization layer
        h = []
        ###### 4-1. down layers
        for block1, attn1, block2, attn2, block3, attn3, temporal_attn, downsample in self.downs:
            x = block1(x, t, spatial_mc)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, spatial_mc)
            x = attn2(x, frame_idx=frame_idx, action_class=cond)
            x = block3(x, t, spatial_mc)
            x = attn3(x, mc)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)
            
        ###### 4-2. mid layers
        x = self.mid_block1(x, t, spatial_mc)
        x = self.mid_attn1(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_block2(x, t, spatial_mc)
        x = self.mid_attn2(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_block3(x, t, spatial_mc)
        x = self.mid_attn3(x, mc)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        ###### 4-3. up layers
        for block1, attn1, block2, attn2, block3, attn3, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, spatial_mc)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, spatial_mc)
            x = attn2(x, frame_idx=frame_idx, action_class=cond)
            x = block3(x, t, spatial_mc)
            x = attn3(x, mc)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)
        
        ###### 4-4 final conv layer
        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,tc:]
        
        return x_fin