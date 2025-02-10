import math
import torch
import random
import numpy as np
from einops import repeat, rearrange
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module.normalization import LayerNorm

import torch.fft as fft

def import_module(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def temporal_swap(x, fn, **kwargs):
    x = rearrange(x, 'b (k c) f h w -> b c (k f) h w', k=2)
    x = fn(x, **kwargs)
    x = rearrange(x, 'b c (k f) h w -> b (k c) f h w', k=2)
    return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm=LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def deform_input(input, flow):
    B, _, H_old, W_old = flow.shape
    B, C, H, W = input.shape
    
    if H_old != H or W_old != W:
        flow = F.interpolate(flow, size=(H, W), mode='bilinear')
    flow = flow.permute(0, 2, 3, 1)
    
    return F.grid_sample(input, flow)

def apply_flow(input, flow, occlusion=None):
    input = deform_input(input, flow)
    
    if exists(occlusion):
        if input.shape[2] != occlusion.shape[2] or input.shape[3] != occlusion.shape[3]:
            occlusion = F.interpolate(occlusion, size=input.shape[2:], mode='bilinear')
        out = input * occlusion
    return out

def nonlinearity(type='silu'):
    if type == 'silu':
        return nn.SiLU()
    elif type == 'gelu':
        return nn.GELU()
    elif type == 'leaky_relu':
        return nn.LeakyReLU()


def noise_sampling(shape, device, noise_cfg=None):
    b, c, f, h, w = shape
    
    if noise_cfg.noise_sampling_method == 'vanilla':
        noise = torch.randn(shape, device=device)
    elif noise_cfg.noise_sampling_method == 'pyoco_mixed':
        noise_alpha_squared = float(noise_cfg.noise_alpha) ** 2
        shared_noise = torch.randn((b, c, 1, h, w), device=device) * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared))
        ind_noise = torch.randn(shape, device=device) * math.sqrt(1 / (1 + noise_alpha_squared))
        noise = shared_noise + ind_noise
    elif noise_cfg.noise_sampling_method == 'pyoco_progressive':
        noise_alpha_squared = float(noise_cfg.noise_alpha) ** 2
        noise = torch.randn(shape, device=device)
        ind_noise = torch.randn(shape, device=device) * math.sqrt(1 / (1 + noise_alpha_squared))
        for i in range(1, noise.shape[2]):
            noise[:, :, i, :, :] = noise[:, :, i - 1, :, :] * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared)) + ind_noise[:, :, i, :, :]
    else:
        raise ValueError(f"Unknown noise sampling method {noise_cfg.noise_sampling_method}")


    return noise

def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed


def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask


def temporal_distance_to_frame_idx(total_num_frames, device):
    frame_idx = torch.tensor([v for v in range(total_num_frames)],device=device)
    frame_idx = frame_idx.repeat(1, 1)
    
    return frame_idx

def traj_to_map(traj, map_shape, sigma=0.3):
    B, C, T, PN = traj.shape
    H, W = map_shape
    
    traj_map = torch.zeros((B, PN, T, H, W), dtype=torch.float32, device=traj.device)
    for b_idx in range(B):
        for t_idx in range(T):
            centers = traj[b_idx, :, t_idx, :].transpose(0,1) #(PN, 2)
            gaussian_maps = gaussian_filter_tensor((H, W), centers, traj.device, sigma)
            traj_map[b_idx, :, t_idx, :, :] = gaussian_maps

    return traj_map

def gaussian_filter_tensor(size, centers, device, sigma):
    H, W = size
    y_grid = torch.arange(H, dtype=torch.float32, device = device).view(H, 1).repeat(1, W) 
    x_grid = torch.arange(W, dtype=torch.float32, device = device).view(1, W).repeat(H, 1)
    
    centers_y = centers[:, 1].view(-1, 1, 1)  # (PN, 1, 1)
    centers_x = centers[:, 0].view(-1, 1, 1)  # (PN, 1, 1)
    centers_v = centers[:, 2].view(-1, 1, 1)  # (PN, 1, 1)
    
    gaussians = torch.exp(-((y_grid - centers_y) ** 2 + (x_grid - centers_x) ** 2) / (2 * sigma**2))
    return gaussians * centers_v

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))