from abc import abstractmethod

import math
import torch
from torch import nn
import torch.nn.functional as F

from model.util import default, temporal_distance_to_frame_idx
from model.module.attention import RelativePositionBias, SinusoidalPosEmb

from model.module.block import InitialBlock, ResBlock, AttentionBlock, ScalingBlock, OutBlock

class Unet3D_SequentialCondAttn(nn.Module):
    def __init__(
        self,
        dim,
        vid_channels = 3,
        trj_channels = 3,
        trj_fea_dim = None,
        cond_num = None,
        pred_num = None,
        dim_mults=(1, 2, 4, 8),
        attn_res=(32, 16, 8),
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        frame_size = 64,
        resnet_groups = 8,
        ):
        super().__init__()
        self.tc, self.tp = cond_num, pred_num
        self.block_num = len(dim_mults)
        
        # 0. embedding timestemp
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
                
        # 1. initial conv & tmp_attn
        init_dim = default(init_dim, dim)
        self.init_blocks = InitialBlock(vid_channels, 
                                        trj_channels, 
                                        trj_fea_dim,
                                        init_dim,
                                        kernel_size = init_kernel_size,
                                        attn_heads = attn_heads,
                                        attn_dim_head = attn_dim_head)
        
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
                ResBlock(dim_in, dim_out, time_emb_dim = time_dim, resnet_groups = resnet_groups),
                AttentionBlock(dim_out, self.tc+self.tp, use_attn = use_attn),
                ScalingBlock(dim_out, self.tc+self.tp, sample_ratio = 0.25, isDown=True) if not is_last else None
            ]))
            
            if not is_last:
                now_res = now_res // 2
        
        mid_dim = dims[-1]
        self.mid_res = ResBlock(mid_dim, mid_dim, time_emb_dim = time_dim, resnet_groups = resnet_groups)
        self.mid_attn = AttentionBlock(mid_dim, self.tc+self.tp, use_attn = False)
                
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            use_attn = (now_res in attn_res) 
            
            self.ups.append(nn.ModuleList([
                ResBlock(dim_out*2, dim_in, time_emb_dim = time_dim, resnet_groups = resnet_groups),
                AttentionBlock(dim_in, self.tc+self.tp, use_attn = use_attn),
                ScalingBlock(dim_in, self.tc+self.tp, sample_ratio = 0.25, isDown=False) if not is_last else None
            ]))
            
            if not is_last:
                now_res = now_res * 2
        
        self.out_blocks = OutBlock(dim*2, vid_channels, trj_channels, resnet_groups = resnet_groups)
    
    def avg_feature(self, trj_f, visibility):
        # B, C, T, PN = trj_f.shape
        # B, 1, T, PN = vis.shape
        
        weighted_features = trj_f * visibility
        
        sum_weights = visibility.sum(dim=2, keepdim=True) # (B, 1, 1, point_num)
        sum_weights = sum_weights.clamp(min=1e-6)
        
        weighted_avg = weighted_features.sum(dim=2, keepdim=True) / sum_weights # (B, C=128, 1, point_num)
        
        trj_f_predInit = weighted_avg.repeat(1,1,self.tp,1)
        return torch.cat((trj_f, trj_f_predInit), dim=2)
    
    def forward(self, x, m, time):
        """
        Apply the model to an input batch.
        :param x: an [B x C x T x H x W] Tensor of inputs.
        :param m: an {"coord": [B x C1 x T x PN], "track_f":[B x C2 x Tp x PN]} Dictionary of inputs Tensor.
        :param time: a 1-D batch of timesteps.
        """
        B, C, T, H, W, device = *x.shape, x.device
        assert T == self.tc + self.tp

        
        ### 1-1. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(self.tc+self.tp, device=x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(self.tc+self.tp, device = x.device, frame_idx=frame_idx)
        
        ### 1-2. embedding timestemp
        t = self.time_mlp(time)
        
        emb = {"t":t, "frame_idx":frame_idx, "time_rel_pos_bias": time_rel_pos_bias}
        
        h = {"out":[], "m_group": []}
        ### 1. initial convolution & temporal attention
        
        m["track_f"]=self.avg_feature(m["track_f"], m["coord"][:,-1:,:self.tc,:]) 
        # [B x C2 x Tp x PN] -> [B x C2 x T x PN]
        
        x, m = self.init_blocks(x, m)
        h["out"].append([x,m])
                
        
        ### 2. down layers
        for m_id, (res, attn, scale) in enumerate(self.downs):
            x, m = res(x, m, emb)
            x, m = attn(x, m , emb)
            h["out"].append([x,m])
            if scale is not None:
                x, (m, m_group) = scale.down(x, m)
                h["m_group"].append(m_group)
            
            
        ### 3. mid layers
        x, m = self.mid_res(x, m , emb)
        x, m = self.mid_attn(x, m , emb)
        
        ### 4. up layers
        x_h, m_h = h["out"].pop()
        for m_id, (res, attn, scale) in enumerate(self.ups):
            x = torch.cat((x,x_h), dim=1)
            m = torch.cat((m,m_h), dim=1)
            
            x, m = res(x, m, emb)
            x, m = attn(x, m , emb)
            
            if scale is not None:
                x_h, m_h = h["out"].pop()
                m_group = h["m_group"].pop()
                x, m = scale.up(x, m, m_h, m_group)

        ### 5 final conv layer
        x_h, m_h = h["out"].pop()
        x = torch.cat([x, x_h], dim=1)
        m = torch.cat([m, m_h], dim=1)
        x_fin, m_fin = self.out_blocks(x, m)

        return x_fin[:,:,self.tc:], m_fin[:,:,self.tc:]
    

if __name__=='__main__':
    import os
    import time
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    model = Unet3D_SequentialCondAttn(
        dim = 64,
        vid_channels = 3,
        trj_channels = 8,
        trj_fea_dim = 128,
        cond_num = 4,
        pred_num = 2,
        dim_mults = (1, 2, 4, 4, 8),
        attn_res = (16, 8)
    ).to("cuda")
    
    lr = 1e-5
    optim = torch.optim.AdamW(model.parameters(),lr=lr)
    model.train()
    while True:
        time_start = time.time()
        B=4
        vid = torch.randn([B, 3, 6, 64, 64]).to("cuda")
        trj = {"coord" : torch.randn([B, 8, 6, 4096]).to("cuda"), 
                 "track_f": torch.randn([B, 128, 6, 4096]).to("cuda")}
        time_index = torch.tensor([B]).to("cuda")

        vid_out, trj_out = model(vid, trj, time_index)
        
        video_target = torch.randn_like(vid_out)
        audio_target = torch.randn_like(trj_out)
        loss =  F.mse_loss(video_target, vid_out)+F.mse_loss(audio_target, trj_out)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        print(f"loss:{loss} time:{time.time()-time_start}")
  