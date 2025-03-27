import torch
from torch import nn

from model.util import default,EinopsToAndFrom, temporal_distance_to_frame_idx
from model.module.attention import TemporalAttentionLayer, RelativePositionBias, SinusoidalPosEmb
from model.module.block import MotionBlock, ConvBlock, ResnetBlock, AttentionBlock, Downsample, Upsample

from rotary_embedding_torch import RotaryEmbedding

class Unet3D_SequentialCondAttn(nn.Module):
    def __init__(
        self,
        dim,
        channels = 3,
        m_channels = 8,
        m_feature_dim = 128,
        out_dim = None,
        cond_num = None,
        pred_num = None,
        dim_mults=(1, 2, 4, 8),
        attn_res=(32, 16, 8),
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        frame_size = 64,
        resnet_groups = 8
        ):
        super().__init__()
        self.tc, self.tp = cond_num, pred_num
        self.mc_ch = m_channels
        self.mc_fea_ch = m_feature_dim
        
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
        self.motion_enc = MotionBlock(self.mc_ch, self.mc_fea_ch, dim)
        
        # 3. unet3d
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

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
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim,  resnet_groups=resnet_groups),
                AttentionBlock(dim_out, use_attn = use_attn),               
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                now_res = now_res // 2
        
        mid_dim = dims[-1]
        self.mid_res = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, resnet_groups=resnet_groups)
        self.mid_tmp_attn = temporal_attn(mid_dim)
                
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res) 
            
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out*2, dim_in, 
                            cond_dim=dim_out, 
                            time_emb_dim = time_dim,
                            resnet_groups=resnet_groups),
                AttentionBlock(dim_in, use_attn = use_attn),
                Upsample(dim_in, use_deconv=True, padding_mode="zeros") if not is_last else nn.Identity(),
            ]))
            
            if not is_last:
                now_res = now_res * 2
                
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvBlock(dim*2, dim, "2d"),
            nn.Conv3d(dim, out_dim, 1)
        )
    
    def forward(self, x, time, cond = None):
        B, C, T, H, W, device = *x.shape, x.device
        assert T == self.tc + self.tp

        ### 0. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(self.tc + self.tp, device=x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(self.tc + self.tp, device = x.device, frame_idx=frame_idx)
        
        t = self.time_mlp(time)

        emb = {"t":t, "frame_idx":frame_idx, "time_rel_pos_bias": time_rel_pos_bias}

        ### 1. motion encoding
        mc = self.motion_enc(cond["x"], cond["fea"])

        ### 2. initial convolution & temporal attention
        x = self.init_conv(x)
        r = x.clone() # for final conv layer
        x = self.init_temporal_attn(x, pos_bias=emb["time_rel_pos_bias"])
        
        ### 4. Unet3D
        h = []
        ###### 4-1. down layers
        for idx, (res, attn, downsample) in enumerate(self.downs):
            x, mc = res(x, mc, emb)
            x = attn(x, mc, emb)
            h.append(x)
            x = downsample(x)
            
        ###### 4-2. mid layers
        x, mc = self.mid_res(x, mc, emb)
        x = self.mid_tmp_attn(x, pos_bias=emb["time_rel_pos_bias"])
        
        ###### 4-3. up layers
        for idx, (res, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)            
            x, mc = res(x, mc, emb)
            x = attn(x, mc, emb)
            x = upsample(x)
        
        ###### 4-4 final conv layer
        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,self.tc:]

        return x_fin