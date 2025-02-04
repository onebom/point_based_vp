from model.module.condition import MotionConditioning, MotionPredictor
import torch.nn as nn
import torch
from einops import rearrange, repeat

class MotionModel(nn.Module):
    def __init__(self, config, autoencoder=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.motion_predictor = MotionPredictor(**config.motion_predictor.model_params)
        
        self.autoencoder.eval()
        
        
    def forward(self, cond_frames, gt_frames, temporal_distance, action, ):
        
        cond_diff = cond_frames[:, :, 1:] - cond_frames[:, :, :-1]
        gt_diff = gt_frames[:, :, 1:] - gt_frames[:, :, :-1]
        
        with torch.no_grad():
            _, cond_motion = self.autoencoder(cond_diff)
            _, gt_motion = self.autoencoder(gt_diff)
        
        pred_motion = self.motion_predictor(cond_motion, temporal_distance, action)
        
        pred_diff = self.autoencoder.decoder(pred_motion)
        pred_frames = gt_frames[:, :, :-1] + pred_diff
        
        diff_loss = torch.nn.functional.l1_loss(pred_frames, gt_frames[:, :, 1:])
        
        
        return diff_loss
    
    @torch.inference_mode()
    def sample(self, cond_frames, gt_frames, temporal_distance, action):
        cond_diff = cond_frames[:, :, 1:] - cond_frames[:, :, :-1]
        gt_diff = gt_frames[:, :, 1:] - gt_frames[:, :, :-1]
        
        _, cond_motion = self.autoencoder(cond_diff)
        _, gt_motion = self.autoencoder(gt_diff)
        
        pred_motion = self.motion_predictor(cond_motion, temporal_distance, action)
        
        pred_diff = self.autoencoder.decoder(pred_motion)
        pred_frames = gt_frames[:, :, :-1] + pred_diff
        
        return cond_motion, gt_motion, pred_motion, pred_frames
        