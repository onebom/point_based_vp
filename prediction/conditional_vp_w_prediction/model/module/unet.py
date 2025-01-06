import torch
from torch import nn
from functools import partial

from einops import rearrange

from rotary_embedding_torch import RotaryEmbedding

from model.util import exists, default, EinopsToAndFrom, Residual, PreNorm, temporal_distance_to_frame_idx
from model.motion_predictor import TrackMotionModel

from model.module.attention import (TemporalAttentionLayer, 
                                    STWAttentionLayer, RelativePositionBias, 
                                    SinusoidalPosEmb, CrossFrameAttentionLayer,
                                    CrossCondAttentionLayer, CondAttentionLayer)
from model.module.block import ResnetBlock, Downsample, Upsample


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        window_size = (2, 4, 4),
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        cond_channels=3,
        attn_heads = 8,
        attn_dim_head = 32,
        resnet_groups=8,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        learn_null_cond = False,
        use_final_activation = False,
        use_deconv=True,
        padding_mode="zeros",
        cond_num = 0,
        pred_num = 0,
        motion_cfg = None,
        template_cfg = None,
    ):
        self.tc = cond_num
        self.tp = pred_num
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', 
                                                    AttentionLayer(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))
        
        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        
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
        block_klass_cond = partial(ResnetBlock, groups=resnet_groups, time_emb_dim = cond_dim, motion_cfg = motion_cfg, template_cfg = template_cfg)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                Residual(PreNorm(dim_out, STWAttentionLayer(dim_out, window_size=self.window_size, shift_size=self.shift_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, STWAttentionLayer(dim_out, window_size=self.window_size, heads=attn_heads, dim_head=attn_dim_head,rotary_emb=rotary_emb))),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn1 = Residual(PreNorm(mid_dim, STWAttentionLayer(mid_dim, window_size=self.window_size, shift_size=self.shift_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        self.mid_attn2 = Residual(PreNorm(mid_dim, STWAttentionLayer(mid_dim, window_size=self.window_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                Residual(PreNorm(dim_in, STWAttentionLayer(dim_in, window_size=window_size, shift_size=self.shift_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, STWAttentionLayer(dim_in, window_size=self.window_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in, use_deconv, padding_mode) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
        
        if use_final_activation:
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond_frames,
        temporal_distance=None,
        motion_cond = None,
        cond = None,
        null_cond_prob = 0.,
        none_cond_mask=None,
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        # if x.shape[2] != self.tp and cond_frames.shape[2] != self.tc: # [B, T, C, H, W] => [B, C, T, H, W]
        #     x = x.permute(0, 2, 1, 3, 4)
        #     cond_frames = cond_frames.permute(0, 2, 1, 3, 4)
        
        batch, device = x.shape[0], x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        
        assert tc == self.tc
        assert tp == self.tp
        
        x = torch.cat([cond_frames, x], dim=2)
        time_rel_pos_bias = self.time_rel_pos_bias(tc+tp, device = x.device)

        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim = -1)
            
        # if motion_cond is not None:
        #     B, Cp, Hp, Wp = motion_cond.shape # [8, 256, 16, 16]
        #     ## TODO: add motion encoder
        #     # x = torch.cat([x, motion_cond], dim=1) # [b, c, t, h, w]

        
        x = self.init_conv(x)
        r = x.clone()
        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        h = []

        for block1, STW_attn1, block2, STW_attn2, temporal_attn, downsample in self.downs:
            x = block1(x, t, motion_cond)
            x = STW_attn1(x)
            x = block2(x, t, motion_cond)
            x = STW_attn2(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, motion_cond)
        x = self.mid_attn1(x)
        x = self.mid_block2(x, t, motion_cond)
        x = self.mid_attn2(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        for block1, STW_attn1 ,block2, STW_attn2, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, motion_cond)
            x = STW_attn1(x)
            x = block2(x, t, motion_cond)
            x = STW_attn2(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,tc:]
        
        return x_fin  
    
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
        motion_cfg = None,
        num_action_classes=None,
    ):
        self.tc = cond_num
        self.tp = pred_num
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.num_action_classes = num_action_classes
        self.motion_cfg = motion_cfg
        
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

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
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

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond_frames,
        motion_cond = None,
        temporal_distance = None,
        cond = None, # action class
        null_cond_prob = 0.,
        none_cond_mask=None,
    ):
        B, C, T, H, W, device = *x.shape, x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        
        assert tc == self.tc
        assert tp == self.tp
        
        x = torch.cat([cond_frames, x], dim=2) # [B, C, 2T, H, W]
        if motion_cond is not None:
            x = torch.cat([x, motion_cond], dim=1)
        
        ### 0. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(tc+tp, device = x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(tc+tp, device = x.device, frame_idx=frame_idx)

        ### 1. initial convolution & temporal attention
        ###### temporal attention using frameDistance embeding
        x = self.init_conv(x) # [B, C', T, H, W]
        r = x.clone() # for final conv layer
        
        time_rel_pos_bias_shape = time_rel_pos_bias.shape
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        ### 2. embedding timestemp
        ##### didnt use action class in my model
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # if exists(cond) and exists(self.num_action_classes):  
        #     c = F.one_hot(cond, self.num_action_classes).type(torch.float)
        #     action_emb =  self.class_cond_mlp(c) # time_emb + cond_emb
        #     t += action_emb

        ### TODO 3. extract motion feature
        motion_pred = None


        ### 4. prediction
        ####### conditioning motion feature by SPADE in Resblock.normalization layer
        h = []
        
        ###### 4-1. down layers
        for block1, attn1, block2, attn2, temporal_attn, downsample in self.downs:
            x = block1(x, t, motion_pred)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, motion_pred)
            x = attn2(x, frame_idx=frame_idx, action_class=cond)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)
            
        ###### 4-2. mid layers
        x = self.mid_block1(x, t, motion_pred)
        x = self.mid_attn1(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_block2(x, t, motion_pred)
        x = self.mid_attn2(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        ###### 4-3. up layers
        for block1, attn1, block2, attn2, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, motion_pred)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, motion_pred)
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
        motion_cfg = None,
        num_action_classes=None,
        pn_prime = 0,
        track_dim = 0,
        model_cfg=None
    ):
        self.tc = cond_num
        self.tp = pred_num
        self.pn_prime = pn_prime
        self.track_dim = track_dim
        
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.num_action_classes = num_action_classes
        self.motion_cfg = motion_cfg
        
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

        # motion predictor
        self.motion_predictor = TrackMotionModel(model_cfg.unet, model_cfg.motion_predictor)

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
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

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond_frames,
        motion_cond = None, # point track
        temporal_distance = None,
        cond = None, # action class
        null_cond_prob = 0.,
        none_cond_mask=None,
    ):
        B, C, T, H, W, device = *x.shape, x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        
        assert tc == self.tc
        assert tp == self.tp
        
        x = torch.cat([cond_frames, x], dim=2) # [B, C, 2T, H, W]
        
        ### 0. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(tc+tp, device=x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(tc+tp, device = x.device, frame_idx=frame_idx)

        ### 1. initial convolution & temporal attention
        ###### temporal attention using frameDistance embeding
        x = self.init_conv(x) # [B, C', T, H, W]
        r = x.clone() # for final conv layer
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        ### 2. embedding timestemp
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        ### 3. extract past motion feature
        mc=self.motion_predictor(motion_cond) #[B, T, PN, C]
        
        spatial_motion = None     
        h = []
        ###### 4-1. down layers
        for block1, attn1, block2, attn2, block3, attn3, temporal_attn, downsample in self.downs:
            x = block1(x, t, spatial_motion)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, spatial_motion)
            x = attn2(x, frame_idx=frame_idx, action_class=cond)
            x = block3(x, t, spatial_motion)
            x = attn3(x, mc)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)
            
        ###### 4-2. mid layers
        x = self.mid_block1(x, t, spatial_motion)
        x = self.mid_attn1(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_block2(x, t, spatial_motion)
        x = self.mid_attn2(x, frame_idx=frame_idx, action_class=cond)
        x = self.mid_block3(x, t, spatial_motion)
        x = self.mid_attn3(x, mc)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        
        ###### 4-3. up layers
        for block1, attn1, block2, attn2, block3, attn3, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, spatial_motion)
            x = attn1(x, frame_idx=frame_idx, action_class=cond)
            x = block2(x, t, spatial_motion)
            x = attn2(x, frame_idx=frame_idx, action_class=cond)
            x = block3(x, t, spatial_motion)
            x = attn3(x, mc)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)
        
        ###### 4-4 final conv layer
        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,tc:]
        
        return x_fin