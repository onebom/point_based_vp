import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class Normalization(nn.Module):
    def __init__(self, dim, cond_dim=128, norm_type='instance', num_groups=8, spade=False):
        super().__init__()
        
        self.spade = spade
        
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm3d(dim)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(dim)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups, dim)
        else:
            raise ValueError(f'Invalid normalization type: {norm_type}')
        
        if spade:
            self.norm = SPADENorm(self.norm, dim, cond_dim=cond_dim, kernel_size=3, padding=1)
            
    def forward(self, x, cond=None):
        return self.norm(x, cond) if self.spade else self.norm(x)
    
    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma
    
class SPADENorm(nn.Module):
    def __init__(self, param_free_norm, dim, cond_dim = 128, kernel_size=3, padding=1):
        super().__init__()
        self.param_free_norm = param_free_norm
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x, cond):
        B, C, T, H, W = x.shape
        
        normalized = self.param_free_norm(x)
        
        if len(cond.shape) == 4:
            cond = rearrange(cond, 'b c h w -> b c 1 h w')
        cond_t = cond.shape[2]
        
        cond = rearrange(cond, 'b c t h w -> (b t) c h w')
        cond = F.interpolate(cond, size=(H, W), mode='nearest')
        
        actv = self.mlp_shared(cond)
        
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        gamma = rearrange(gamma, '(b t) c h w -> b c t h w', t=cond_t)
        beta = rearrange(beta, '(b t) c h w -> b c t h w', t=cond_t)
        
        out =  normalized * (1 + gamma) + beta
        
        return out

class SPADENorm_motion(nn.Module):
    def __init__(self, dim, motion_dim = 128, kernel_size=3, padding=1, norm_type = 'GroupNorm', groups=8):
        super().__init__()
        if norm_type == 'GroupNorm':
            self.param_free_norm = nn.GroupNorm(groups, dim)
            
        else:
            raise ValueError(f'Invalid normalization type: {norm_type}')
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(motion_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x, motion_cond=None):
        """_summary_

        Args:
            x : [B, 128, 4, 64, 64]
            motion_cond : [B, 256, 4, 16, 16] or [B, 256, 16, 16]
        """
        B, C, _, H, W = x.shape
        if len(motion_cond.shape) == 4:
            motion_cond = rearrange(motion_cond, 'b c h w -> b c 1 h w')
        T = motion_cond.shape[2]
        
        normalized = self.param_free_norm(x)
        motion_cond = rearrange(motion_cond, 'b c t h w -> (b t) c h w')
        motion_cond_interp = F.interpolate(motion_cond, size=(H, W), mode='nearest')
        
        actv = self.mlp_shared(motion_cond_interp)
        
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        gamma = rearrange(gamma, '(b t) c h w -> b c t h w', t=T)
        beta = rearrange(beta, '(b t) c h w -> b c t h w', t=T)
        
        out =  normalized * (1 + gamma) + beta
        
        return out
    
class SPADENorm_template(nn.Module):
    def __init__(self, dim, motion_dim = 128, kernel_size=3, padding=1, norm_type = 'GroupNorm', groups=8):
        super().__init__()
        if norm_type == 'GroupNorm':
            self.param_free_norm = nn.GroupNorm(groups, dim)
            
        else:
            raise ValueError(f'Invalid normalization type: {norm_type}')
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(motion_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x, template_cond=None):
        normalized = self.param_free_norm(x)
        normalized = rearrange(normalized, 'b c t h w -> (b t) c h w')

        template_cond = F.interpolate(template_cond, size=x.size()[-2:], mode='nearest')
        actv = self.mlp_shared(template_cond)
        
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        out =  normalized * (1 + gamma) + beta

        return rearrange(out, '(b t) c h w -> b c t h w', t=x.shape[2])