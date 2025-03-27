import torch
import torch.nn as nn

from model.sde import SDEs

from model.util import import_module, traj_to_map

class VideoNCSN(nn.Module): 
    def __init__(self, config, autoencoder=None, is_train=True):
        super().__init__()

        unet_class = import_module(f"model.module.unet", config.unet.type) ## DirectUnet3D_motion
        
        self.unet_params = config.unet.model_params
        self.unet_type = config.unet.type
        
        if self.unet_type == "DirectUnet3D_CrossFrameAttn":
            self.score_model = unet_class(**self.unet_params)    
        elif self.unet_type == "DirectUnet3D_CrossCondAttn":
            self.unet_cond_params = config.unet.cond_params
            self.score_model = unet_class(**self.unet_params,**self.unet_cond_params)    
                        
        self.sde_params = config.sde.sde_params
        self.sampling_params = config.sde.sampling_params
        self.noise_cfg = config.sde.noise_params
        self.SDEs = SDEs(self.score_model, self.sde_params, self.sampling_params, noise_cfg=self.noise_cfg)
            
    def forward(self, cond_frames, gt_frames, motion_cond=None, action=None):
        # B, C, T, H, W = cond_frames.shape
        H, W = cond_frames.shape[3], cond_frames.shape[4]
        
        if self.unet_type == "DirectUnet3D_CrossFrameAttn":
            cond = motion_cond
        
        elif self.unet_type == "DirectUnet3D_CrossCondAttn":
            # B, C, T, PN = motion_cond.shape
            traj_map = traj_to_map(motion_cond, (H, W))
        
            cond = {"traj": motion_cond / H,    # [B, C, T, PN]
                    "traj_map": traj_map,       # [B, PN, T, H, W]
                    "action": action}
        
        loss = self.SDEs(cond_frames, gt_frames, cond = cond)
        
        return loss
    
    @torch.inference_mode()
    def sample_video(self, cond_frames, gt_frames, motion_cond=None, action=None):
        # B, C, T, H, W = cond_frames.shape
        H, W = cond_frames.shape[3], cond_frames.shape[4]
        
        if self.unet_type == "DirectUnet3D_CrossFrameAttn":
            cond = motion_cond
        
        elif self.unet_type == "DirectUnet3D_CrossCondAttn":
            # B, C, T, PN = motion_cond.shape
            traj_map = traj_to_map(motion_cond, (H, W))
            
            cond = {"traj": motion_cond / H,
                    "traj_map": traj_map,
                    "action": action}
        # gt motion condition
        pred, nfev = self.SDEs.sample(cond_frames, gt_frames, cond = cond)
        
        # pred = np.clip(pred.cpu().numpy() * 255., 0, 255).astype(np.uint8)
        pred = torch.clamp(pred, min=0, max=1).to(dtype=torch.float32)
        
        return pred, nfev
    

            
 