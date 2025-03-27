import math
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

from einops import rearrange

from rotary_embedding_torch import RotaryEmbedding

from model.util import exists, default, EinopsToAndFrom, Residual, PreNorm, temporal_distance_to_frame_idx

from model.module.attention import (TemporalAttentionLayer, RelativePositionBias, SinusoidalPosEmb,
                                    CrossFrameAttentionLayer,CrossCondAttentionLayer, 
                                    CondAttentionLayer, CondAttention2DModule, 
                                    CondAttentionTemporalModule, AttentionSTModule,
                                    CondAttentionSTModule, MotionQEncoder)

from model.module.block import ResnetBlock, Downsample, Upsample
from model.module.condition import ClassCondition
from model.module.motion_module import AttentionPointSelector
from model.motion_predictor import TrackPredictor

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
                CrossFrameAttentionLayer(mid_dim, use_attn=use_attn),
                block_klass_cond(dim_out, dim_out),
                CrossFrameAttentionLayer(mid_dim, use_attn=use_attn),
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

class Unet3D_noCond(nn.Module):
    def __init__(
        self,
        dim,
        channels = 3,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        attn_res=(32, 16, 8),
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        frame_size = 64,
        resnet_groups = 8,
        use_deconv=True,
        padding_mode = "zeros",
        use_final_activation = False
        ):
        super().__init__()
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', 
                                                    TemporalAttentionLayer(dim, heads = attn_heads, 
                                                                           dim_head = attn_dim_head, 
                                                                           rotary_emb = rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet
        
        # 0. initial conv & tmp_attn
        init_dim = default(init_dim, dim)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, 
                                   (1, init_kernel_size, init_kernel_size)
                                   , padding = (0, init_padding, init_padding))
        self.init_temporal_attn = temporal_attn(init_dim)
        
        # 1. embedding timestemp
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 2. unet3d
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        resblock2D = partial(ResnetBlock, conv_method="2d", groups=resnet_groups)
        resblockTemporal = partial(ResnetBlock, conv_method="temporal", groups=resnet_groups, dropout_rate = 0.1)
                        
        ### dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        
        ### block
        now_res = frame_size
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res) 
            
            self.downs.append(nn.ModuleList([
                resblock2D(dim_in, dim_out, time_emb_dim = time_dim),
                resblockTemporal(dim_out, dim_out),
                CondAttention2DModule(dim_out, use_attn = use_attn),
                CondAttentionSTModule(dim_out, use_attn = use_attn),
                temporal_attn(dim_out),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                now_res = now_res // 2
        
        mid_dim = dims[-1]
        self.mid_res = resblock2D(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_tmp_res = resblockTemporal(mid_dim, mid_dim)
        self.mid_tmp_attn = temporal_attn(mid_dim)
                
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res) 
            
            self.ups.append(nn.ModuleList([
                resblock2D(dim_out*2, dim_in, time_emb_dim = time_dim),
                resblockTemporal(dim_in, dim_in),
                CondAttention2DModule(dim_in, use_attn = use_attn),
                CondAttentionSTModule(dim_in, use_attn = use_attn),
                temporal_attn(dim_in),
                Upsample(dim_in, use_deconv, padding_mode) if not is_last else nn.Identity(),
            ]))
            
            if not is_last:
                now_res = now_res * 2
                
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            resblock2D(dim*2, dim),
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
    
    def forward(self, x, time, cond_frames, cond = None):
        B, C, T, H, W, device = *x.shape, x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        
        x = torch.cat([cond_frames, x], dim=2)
        
        ### 0. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(tc+tp, device=x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(tc+tp, device = x.device, frame_idx=frame_idx)
        
        ### 1. initial convolution & temporal attention
        x = self.init_conv(x)
        r = x.clone() # for final conv layer
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        ### 3. embedding timestemp
        t = self.time_mlp(time)
        
        ### 4. Unet3D
        h = []
        h_for_check=[]
        ###### 4-1. down layers
        for idx, (res, tmp_res, attn, sptmp_attn, tmp_attn, downsample) in enumerate(self.downs):
            x = res(x, t)
            x = tmp_res(x)
            if idx == len(self.downs)-2:
                h_for_check.append(x[:,:,tc:])
            x = attn(x, frame_idx=frame_idx)
            x = sptmp_attn(x, frame_idx=frame_idx)
            if idx == 0:
                h_for_check.append(x[:,:,tc:])
            x = tmp_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)
            
        ###### 4-2. mid layers
        x = self.mid_res(x, t)
        x = self.mid_tmp_res(x)
        x = self.mid_tmp_attn(x, pos_bias=time_rel_pos_bias)
        
        ###### 4-3. up layers
        for idx, (res, tmp_res, attn, sptmp_attn, tmp_attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)            
            x = res(x, t)
            x = tmp_res(x)
            if idx == len(self.ups)-2:
                h_for_check.append(x[:,:,tc:])
            x = attn(x, frame_idx=frame_idx)
            x = sptmp_attn(x, frame_idx=frame_idx)
            if idx == len(self.ups)-2:
                h_for_check.append(x[:,:,tc:])
            x = tmp_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)
        
        ###### 4-4 final conv layer
        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,tc:]

        return x_fin, h_for_check
    
class Unet3D_SequentialCondAttn(nn.Module):
    def __init__(
        self,
        dim,
        channels = 3,
        out_dim = None,
        cond_num = None,
        pred_num = None,
        dim_mults=(1, 2, 4, 8),
        attn_res=(32, 16, 8),
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        selected_k = 1024,
        motion_dim = None,
        track_dim = None,
        track_dim_expanded = None,
        trj_f_dim = None,
        interaction_num = None,
        frame_size = 64,
        resnet_groups = 8,
        use_deconv=True,
        padding_mode = "zeros",
        use_final_activation = False
        ):
        super().__init__()
        self.tc, self.tp = cond_num, pred_num
        
        self.top_k = selected_k
        self.motion_dim = motion_dim
        self.track_dim = track_dim
        self.track_dim_expanded =track_dim_expanded
        self.trj_f_dim =trj_f_dim
        self.interaction_num =interaction_num
        
        self.tc_dim=(self.tc+self.tp)*self.motion_dim
        
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', 
                                                    TemporalAttentionLayer(dim, heads = attn_heads, 
                                                                           dim_head = attn_dim_head, 
                                                                           rotary_emb = rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet
        
        # 0. initial conv & tmp_attn
        init_dim = default(init_dim, dim)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, 
                                   (1, init_kernel_size, init_kernel_size)
                                   , padding = (0, init_padding, init_padding))
        self.init_temporal_attn = temporal_attn(init_dim)
        
        # 1. embedding timestemp
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 2. track predictor
        self.motion_predictor = TrackPredictor(motion_dim = self.motion_dim,
                                               trj_dim_exp = self.track_dim_expanded,
                                               trj_f_dim = self.trj_f_dim,
                                               interaction_num = self.interaction_num)
        
        # 3. unet3d
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        resblock2D = partial(ResnetBlock, conv_method="2d", groups=resnet_groups)
        resblockTemporal = partial(ResnetBlock, conv_method="temporal", groups=resnet_groups, dropout_rate = 0.1)
                        
        ### dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        
        ### block
        now_res = frame_size
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res) 
            
            self.downs.append(nn.ModuleList([
                resblock2D(dim_in, dim_out, time_emb_dim = time_dim),
                resblockTemporal(dim_out, dim_out),
                MotionQEncoder(self.tc_dim, dim_out,
                               pn_prime=now_res**2,
                               motion_dim= self.motion_dim,
                               use_attn = use_attn),
                CondAttention2DModule(dim_out, use_attn = use_attn),
                CondAttentionSTModule(dim_out, use_attn = use_attn),
                temporal_attn(dim_out),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                now_res = now_res // 2
        
        mid_dim = dims[-1]
        self.mid_res = resblock2D(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_tmp_res = resblockTemporal(mid_dim, mid_dim)
        self.mid_tmp_attn = temporal_attn(mid_dim)
                
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res) 
            
            self.ups.append(nn.ModuleList([
                resblock2D(dim_out*2, dim_in, time_emb_dim = time_dim),
                resblockTemporal(dim_in, dim_in),
                CondAttention2DModule(dim_in, cond_dim=dim_out, use_attn = use_attn),
                CondAttentionSTModule(dim_in, cond_dim=dim_out, use_attn = use_attn),
                temporal_attn(dim_in),
                Upsample(dim_in, use_deconv, padding_mode) if not is_last else nn.Identity(),
            ]))
            
            if not is_last:
                now_res = now_res * 2
                
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            resblock2D(dim*2, dim),
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
    
    def traj_select(self, trj, trj_f=None, top_k=128):
        device = trj.device
        B, C, T, PN = trj.shape
        
        x = rearrange(trj, 'b c t pn -> b pn (t c)')
        d_k = x.shape[-1]
        
        sim = torch.matmul(x, x.transpose(-2, -1)) * (d_k ** -0.5)  # (B, PN, PN)
        attn = F.softmax(sim, dim=-1) # Shape (B, PN, PN)
        
        scores = attn.mean(dim=-1) # Shape (B, PN)
        
        topk_scores, topk_indices = torch.topk(scores, top_k, dim=-1)
        trj_topk_indices = topk_indices.unsqueeze(1).unsqueeze(1) # (B, 1, 1, top_k)
        
        selected_trj = torch.gather(trj, dim=-1, index=trj_topk_indices.expand(-1, C, T, -1))
        
        selected_trj_f = None
        if trj_f:
            selected_trj_f = torch.gather(trj_f, dim=-1, index=trj_topk_indices.expand(-1, C, T, -1))
        
        return selected_trj, selected_trj_f
    
    def forward(self, x, time, cond_frames, cond = None):
        B, C, T, H, W, device = *x.shape, x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        assert tc == self.tc and tp == self.tp
        
        x = torch.cat([cond_frames, x], dim=2)
        # trj, future_trj = cond["traj"][:,:,:tc], cond["traj"][:,:,tc:]
        # trj_f, future_trj_f = cond["traj_f"][:,:,:tc], cond["traj_f"][:,:,tc:]
        trj, trj_f = cond["traj"], cond["traj_f"]
        
        ### 0. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(tc+tp, device=x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(tc+tp, device = x.device, frame_idx=frame_idx)
        
        ### 1. initial convolution & temporal attention
        x = self.init_conv(x)
        r = x.clone() # for final conv layer
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        ### 2-1. select trajectory
        ######[TODO] 미래 정보 없이 select하게 하기
        trj, trj_f = self.traj_select(trj, trj_f, self.top_k) # 256 이하로 남기기
        ### 2-2. future trj predict
        ###### inp: [B,3,Tp,PN], [B,128,Tp,PN] out: [B,C_h,Tp+Tf,PN]
        mc = self.motion_predictor(trj, trj_f)
                
        ### 3. embedding timestemp
        t = self.time_mlp(time)
        
        ### 4. Unet3D
        h = []
        mc_h = []
        h_for_check=[]
        ###### 4-1. down layers
        for idx, (res, tmp_res, mc_encoder, attn, sptmp_attn, tmp_attn, downsample) in enumerate(self.downs):
            x = res(x, t)
            x = tmp_res(x)
            e_mc, mc = mc_encoder(mc)
            if idx == len(self.downs)-1:
                h_for_check.append(x[:,:,tc:])
            x = attn(x, e_mc, frame_idx=frame_idx)
            x = sptmp_attn(x, e_mc, frame_idx=frame_idx)
            if idx == len(self.downs)-1:
                h_for_check.append(x[:,:,tc:])
            x = tmp_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            mc_h.append(e_mc)
            x = downsample(x)
            
        ###### 4-2. mid layers
        x = self.mid_res(x, t)
        x = self.mid_tmp_res(x)
        x = self.mid_tmp_attn(x, pos_bias=time_rel_pos_bias)
        
        ###### 4-3. up layers
        for idx, (res, tmp_res, attn, sptmp_attn, tmp_attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            e_mc = mc_h.pop()
            
            x = res(x, t)
            x = tmp_res(x)
            if idx == len(self.ups)-1:
                h_for_check.append(x[:,:,tc:])
            x = attn(x, e_mc, frame_idx=frame_idx)
            x = sptmp_attn(x, e_mc, frame_idx=frame_idx)
            if idx == len(self.ups)-1:
                h_for_check.append(x[:,:,tc:])
            x = tmp_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)
        
        ###### 4-4 final conv layer
        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,tc:]

        return x_fin, h_for_check