from model.module.condition import MotionVAE
import torch.nn as nn
import torch
from einops import rearrange, repeat

class MotionModel(nn.Module):
    def __init__(self, config, autoencoder=None):
        super().__init__()
        self.motion_vae = MotionVAE(**config.motion_predictor.model_params)
        
        
        
    def forward(self, cond_frames, gt_frames, temporal_distance, action, ):
        
        cond_diff = cond_frames[:, :, 1:] - cond_frames[:, :, :-1]
        gt_diff = gt_frames[:, :, 1:] - gt_frames[:, :, :-1]
        
        recon_loss, kld_loss = self.motion_vae.loss(cond_diff, gt_diff, temporal_distance, action,)
        
        return recon_loss, kld_loss
    
    @torch.inference_mode()
    def sample(self, cond_frames, gt_frames, temporal_distance, action):
        cond_diff = cond_frames[:, :, 1:] - cond_frames[:, :, :-1]
        gt_diff = gt_frames[:, :, 1:] - gt_frames[:, :, :-1]
        
        pred_diff = self.motion_vae.sample(cond_diff, temporal_distance, action)
        
        pred_frames = gt_frames[:, :, :-1] + pred_diff
        
        return cond_diff, gt_diff, pred_diff, pred_frames
        