import torch
import torch.nn as nn

from model.diffusion import GaussianDiffusion

from model.util import exists, import_module

class VideoDirectDiffusion(nn.Module): 
    def __init__(self, config, is_train=True):
        super().__init__()

        unet_class = import_module(f"model.unet", config.unet.type) ## DirectUnet3D_motion
        self.noise_cfg = config.diffusion.noise_params ## using pyoco
        
        self.unet_params = config.unet.model_params
        self.diffusion_params = config.diffusion.diffusion_params

        self.unet = unet_class(**self.unet_params) 
            
        self.diffusion = GaussianDiffusion(self.unet, **self.diffusion_params, noise_cfg=self.noise_cfg)
        
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()            
            
    def forward(self, cond_frames, gt_frames, motion_cond=None):
        # B, C, T, H, W = cond_frames.shape
        H, W = cond_frames.shape[3], cond_frames.shape[4]
        diffusion_loss, _ = self.diffusion(cond_frames, gt_frames, cond = motion_cond)        
        return diffusion_loss
        
    @torch.inference_mode()
    def sample_video(self, cond_frames, gt_frames, motion_cond=None):
        # B, C, T, H, W = cond_frames.shape
        H, W = cond_frames.shape[3], cond_frames.shape[4]
        pred = self.diffusion.sample(gt_frames, cond_frames, cond = motion_cond)
        return pred
    
    def train_mode(self,):
        self.unet.train()
        self.diffusion.train()            
        
    @torch.inference_mode()
    def eval_mode(self,):
        self.unet.eval()
        self.diffusion.eval()
 