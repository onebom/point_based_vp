import torch
import torch.nn as nn

from model.diffusion import GaussianDiffusion

from model.util import exists, import_module

class VideoDirectDiffusion(nn.Module): 
    def __init__(self, config, autoencoder=None, is_train=True):
        super().__init__()

        unet_class = import_module(f"model.module.unet", config.unet.type) ## DirectUnet3D_motion
        self.noise_cfg = config.diffusion.noise_params ## using pyoco
        
        self.unet_params = config.unet.model_params
        self.diffusion_params = config.diffusion.diffusion_params
        
        if config.unet.type == "DirectUnet3D_CrossFrameAttn":
            self.unet = unet_class(**self.unet_params)    
        elif config.unet.type == "DirectUnet3D_CrossCondAttn":
            self.unet_cond_params = config.unet.cond_params
            self.model_params = config
            self.unet = unet_class(**self.unet_params,**self.unet_cond_params, model_cfg = self.model_params)    
                        
        self.diffusion = GaussianDiffusion(self.unet, **self.diffusion_params, noise_cfg=self.noise_cfg)
        
        self.autoencoder = autoencoder ## None
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()            
            
    def forward(self, cond_frames, gt_frames, temporal_distance=None, motion_cond=None, action=None):
        B, C, T, H, W = cond_frames.shape
        
        
        ### TODO. create motion cond
        
        # cond_of = []
        # cond_occ = []
        # gt_of = []
        # gt_occ = []
        
        # with torch.no_grad():
        #     for i in range(T-1):
        #         cond_generated = self.autoencoder.generate_sample(cond_frames[:, :, i:i+2])
        #         gt_generated = self.autoencoder.generate_sample(gt_frames[:, :, i:i+2])
                
        #         cond_of.append(cond_generated['optical_flow'].permute(0, 3, 1, 2))
        #         gt_of.append(gt_generated['optical_flow'].permute(0, 3, 1, 2))
        #         cond_occ.append(cond_generated['occlusion_map'])
        #         gt_occ.append(gt_generated['occlusion_map'])
        
        # cond_of = torch.stack(cond_of, dim=2) # [B, 2, T-1, H, W]
        # cond_occ = torch.stack(cond_occ, dim=2) # [B, 1, T-1, H, W]
        # gt_of = torch.stack(gt_of, dim=2)
        # gt_occ = torch.stack(gt_occ, dim=2)
        
        # cond_motion = torch.cat([cond_of, cond_occ*2-1], dim=1) # [B, 3, T-1, H, W]
        # gt_motion = torch.cat([gt_of, gt_occ*2-1], dim=1)
        
        # pred_motion = self.motion_predictor(cond_motion, temporal_distance, action) if exists(self.motion_predictor) else None
        
        diffusion_loss, _ = self.diffusion(cond_frames, gt_frames, motion_cond=motion_cond, temporal_distance=temporal_distance, cond=action)
        
        motion_loss = torch.tensor(0.0, device=cond_frames.device)
        
        return diffusion_loss, motion_loss
        
    
    @torch.inference_mode()
    def sample_video(self, cond_frames, gt_frames, temporal_distance, motion_cond=None, action=None, return_motion=False):
        B, C, T, H, W = cond_frames.shape
        
        # cond_of = []
        # cond_occ = []
        # gt_of = []
        # gt_occ = []
        
        # with torch.no_grad():
        #     for i in range(T-1):
        #         cond_generated = self.autoencoder.generate_sample(cond_frames[:, :, i:i+2])
        #         gt_generated = self.autoencoder.generate_sample(gt_frames[:, :, i:i+2])
                
        #         cond_of.append(cond_generated['optical_flow'].permute(0, 3, 1, 2))
        #         gt_of.append(gt_generated['optical_flow'].permute(0, 3, 1, 2))
        #         cond_occ.append(cond_generated['occlusion_map'])
        #         gt_occ.append(gt_generated['occlusion_map'])
        
        # cond_of = torch.stack(cond_of, dim=2) # [B, 2, T-1, H, W]
        # cond_occ = torch.stack(cond_occ, dim=2) # [B, 1, T-1, H, W]
        # gt_of = torch.stack(gt_of, dim=2)
        # gt_occ = torch.stack(gt_occ, dim=2)
        
        # cond_motion = torch.cat([cond_of, cond_occ*2-1], dim=1) # [B, 3, T-1, H, W]
        # gt_motion = torch.cat([gt_of, gt_occ*2-1], dim=1)
            
        # pred_motion = self.motion_predictor(cond_motion, temporal_distance, action) if exists(self.motion_predictor) else None
        
        # gt motion condition
        pred = self.diffusion.sample(gt_frames, cond_frames, motion_cond=motion_cond,temporal_distance=temporal_distance, cond=action)
        
        # if return_motion:
        #     return pred, cond_motion, gt_motion, pred_motion
        
        return pred
    
        
    
    def train_mode(self,):
        self.unet.train()
        self.diffusion.train()            
        
    @torch.inference_mode()
    def eval_mode(self,):
        self.unet.eval()
        self.diffusion.eval()
 