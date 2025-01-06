import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.block import ResnetBlock, Downsample, Upsample
from model.module.normalization import Normalization
from model.module.attention import STAttentionBlock

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(**cfg.encoder_params)
        self.decoder = Decoder(**cfg.decoder_params)
    def forward(self, x):
        """_summary_

        Args:
            x : [B, C, T-1, H, W]
        """
        latent = self.encoder(x)
        out = self.decoder(latent)
        
        return out, latent
    
class Encoder(nn.Module):
    def __init__(
        self,
        dim,
        z_channel,
        in_channel=3,
        channel_mult=[1, 2, 4, 4],
        resolution=64,
        attn_res = [32, 16]
    ):
        super().__init__()
        
        self.num_resolutions = len(channel_mult)
        
        self.conv_in = nn.Conv3d(in_channel, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.downs = nn.ModuleList([])
        
        curr_res = resolution
        in_channel_mult = (1,) + tuple(channel_mult)
        for i_level in range(self.num_resolutions):
            is_last = i_level >= self.num_resolutions - 2
            use_attn = (curr_res in attn_res)
            
            block_in = dim * in_channel_mult[i_level]
            block_out = dim * channel_mult[i_level]
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(block_in, block_out),
                STAttentionBlock(block_out) if use_attn else nn.Identity(),
                ResnetBlock(block_out, block_out),
                STAttentionBlock(block_out) if use_attn else nn.Identity(),
                Downsample(block_out) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                curr_res = curr_res // 2
        
        self.mid_block1 = ResnetBlock(block_out, block_out)
        self.mid_attn1 = STAttentionBlock(block_out)
        self.mid_block2 = ResnetBlock(block_out, block_out)
        
        self.norm = Normalization(block_out)
        self.nonlinearity = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out, z_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                    
                
    
    def forward(self, x):
        x = self.conv_in(x)
        
        for block1, attn1, block2, attn2, down in self.downs:
            x = block1(x)
            x = attn1(x)
            x = block2(x)
            x = attn2(x)
            x = down(x)
        x = self.mid_block1(x)
        x = self.mid_attn1(x)
        x = self.mid_block2(x)
        
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
            
        return x
        
        

class Decoder(nn.Module):
    def __init__(
        self,
        dim,
        z_channel,
        out_channel=3,
        channel_mult=[1, 2, 4, 4],
        resolution=64,
        attn_res = [16, 8]
    ):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.resolution = resolution
        
        curr_res = resolution // 2**(self.num_resolutions-2)
        in_channel_mult = (1,) + tuple(channel_mult)
        block_in = dim * in_channel_mult[-1]
        
        
        self.conv_in = nn.Conv3d(z_channel, block_in, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        self.mid_block1 = ResnetBlock(block_in, block_in)
        self.mid_attn1 = STAttentionBlock(block_in)
        self.mid_block2 = ResnetBlock(block_in, block_in)
        
        self.ups = nn.ModuleList([])
        for i_level in reversed(range(self.num_resolutions)):
            is_last = i_level <= 1
            use_attn = (curr_res in attn_res)
            
            block_out = dim * channel_mult[i_level]
            
            self.ups.append(nn.ModuleList([
                ResnetBlock(block_in, block_out),
                STAttentionBlock(block_out) if use_attn else nn.Identity(),
                ResnetBlock(block_out, block_out),
                STAttentionBlock(block_out) if use_attn else nn.Identity(),
                Upsample(block_out) if not is_last else nn.Identity(),
            ]))
            block_in = block_out
            
            if not is_last:
                curr_res = curr_res * 2
            
        self.norm = Normalization(block_out)
        self.nonlinearlity = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out, out_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        
    def forward(self, x):
        x = self.conv_in(x)
        
        x = self.mid_block1(x)
        x = self.mid_attn1(x)
        x = self.mid_block2(x)
        
        for block1, attn1, block2, attn2, up in self.ups:
            x = block1(x)
            x = attn1(x)
            x = block2(x)
            x = attn2(x)
            x = up(x)
            
        x = self.norm(x)
        x = self.nonlinearlity(x)
        x = self.conv_out(x)
            
        return x
            
            