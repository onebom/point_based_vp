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
        trj_channels = 3,
        trj_fea_dim = None,
        cond_num = None,
        pred_num = None,
        init_dim = None,
        attn_heads = 8
        ):
        super().__init__()
        self.tc, self.tp = cond_num, pred_num
        
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
        self.init_blocks = InitialBlock(trj_channels, trj_fea_dim, init_dim)
        
        # 3. unet3d
        self.blocks = nn.ModuleList([])
        in_out = [(init_dim, dim*2),(dim*2, dim)]

        ### block
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.blocks.append(nn.ModuleList([
                ResBlock(dim_in, dim_out, time_emb_dim = time_dim),
                AttentionBlock(dim_out, self.tc+self.tp),
            ]))
        
        self.out_blocks = OutBlock(dim, trj_channels)
    
    def avg_feature(self, f, visibility):
        # B, C, T, PN = trj_f.shape
        # B, 1, T, PN = vis.shape
        
        weighted_features = f * visibility
        
        sum_weights = visibility.sum(dim=2, keepdim=True) # (B, 1, 1, point_num)
        sum_weights = sum_weights.clamp(min=1e-6)
        
        weighted_avg = weighted_features.sum(dim=2, keepdim=True) / sum_weights # (B, C=128, 1, point_num)
        
        avg_f = weighted_avg.repeat(1,1,self.tp,1)
        return torch.cat((f, avg_f), dim=2)
    
    def forward(self, x, time, cond):
        """
        Apply the model to an input batch.
        :param x: an [B x C x T x H x W] Tensor of inputs.
        :param m: an {"coord": [B x C1 x T x PN], "track_f":[B x C2 x Tp x PN]} Dictionary of inputs Tensor.
        :param time: a 1-D batch of timesteps.
        """
        B, C, T, PN, device = *x.shape, x.device
        assert T == self.tc + self.tp
        m_cond = cond["track_f"]

        ### 1-1. embedding frames distance(from temporal_distance) like position embedding
        frame_idx = temporal_distance_to_frame_idx(self.tc+self.tp, device=x.device)
        time_rel_pos_bias = self.time_rel_pos_bias(self.tc+self.tp, device = x.device, frame_idx=frame_idx)
        
        emb = {"frame_idx": frame_idx, "time_rel_pos_bias": time_rel_pos_bias}

        ### 1-2. embedding timestemp
        t = self.time_mlp(time)
                
        ### 1. initial convolution & temporal attention
        visibility = x[:,2:3,:self.tc,:]
        m_cond=self.avg_feature(m_cond, visibility) # [B x C2 x Tp x PN] -> [B x C2 x T x PN]
        
        x = self.init_blocks(x, m_cond)
                        
        ### 2. down layers 
        for m_id, (res, attn) in enumerate(self.blocks):
            x = res(x, t)
            x = attn(x, emb)

        ### 5 final conv layer
        x_fin = self.out_blocks(x)

        return x_fin[:,:,self.tc:]
    

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
        trj = torch.randn([B, 8, 6, 4096]).to("cuda")
        cond = {"track_f": torch.randn([B, 128, 6, 4096]).to("cuda")}
        time_index = torch.tensor([B]).to("cuda")

        trj_out = model(trj, time_index, cond)
        
        trj_target = torch.randn_like(trj_out)
        loss =  F.mse_loss(trj_out, trj_target)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        print(f"loss:{loss} time:{time.time()-time_start}")
  