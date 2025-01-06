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
            self.unet = unet_class(**self.unet_params,**self.unet_cond_params)    
                        
        self.diffusion = GaussianDiffusion(self.unet, **self.diffusion_params, noise_cfg=self.noise_cfg)
        
        self.autoencoder = autoencoder ## None
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()            
            
    def forward(self, cond_frames, gt_frames, motion_cond=None, action=None):
        # B, C, T, H, W = cond_frames.shape
        B, C, T, PN = motion_cond.shape
        H, W = cond_frames.shape[3], cond_frames.shape[4]
        
        traj_map = self.traj_to_map(motion_cond[:,:,:,:PN//2], (H, W))
    
        cond = {"traj": motion_cond / H,    # [B, C, T, PN]
                "traj_map": traj_map,       # [B, PN, T, H, W]
                "action": action}
        diffusion_loss, _ = self.diffusion(cond_frames, gt_frames, cond = cond)
        motion_loss = torch.tensor(0.0, device=cond_frames.device)
        
        return diffusion_loss, motion_loss
        
    
    @torch.inference_mode()
    def sample_video(self, cond_frames, gt_frames, motion_cond=None, action=None):
        # B, C, T, H, W = cond_frames.shape
        B, C, T, PN = motion_cond.shape
        H, W = cond_frames.shape[3], cond_frames.shape[4]
        
        traj_map = self.traj_to_map(motion_cond[:,:,:,:PN//2], (H, W))
        
        cond = {"traj": motion_cond / H,
                "traj_map": traj_map,
                "action": action}
        # gt motion condition
        pred = self.diffusion.sample(gt_frames, cond_frames, cond = cond)
        
        return pred
    
    def traj_to_map(self, traj, map_shape, sigma=0.3):
        B, C, T, PN = traj.shape
        H, W = map_shape
        
        traj_map = torch.zeros((B, PN, T, H, W), dtype=torch.float32, device=traj.device)
        for b_idx in range(B):
            for t_idx in range(T):
                centers = traj[b_idx, :, t_idx, :].transpose(0,1) #(PN, 2)
                gaussian_maps = self.gaussian_filter_tensor((H, W), centers, traj.device, sigma)
                traj_map[b_idx, :, t_idx, :, :] = gaussian_maps
    
        return traj_map
    
    def gaussian_filter_tensor(self, size, centers, device, sigma):
        H, W = size
        y_grid = torch.arange(H, dtype=torch.float32, device = device).view(H, 1).repeat(1, W) 
        x_grid = torch.arange(W, dtype=torch.float32, device = device).view(1, W).repeat(H, 1)
        
        centers_y = centers[:, 1].view(-1, 1, 1)  # (PN, 1, 1)
        centers_x = centers[:, 0].view(-1, 1, 1)  # (PN, 1, 1)
        centers_v = centers[:, 2].view(-1, 1, 1)  # (PN, 1, 1)
        
        gaussians = torch.exp(-((y_grid - centers_y) ** 2 + (x_grid - centers_x) ** 2) / (2 * sigma**2))
        return gaussians * centers_v
    
    def train_mode(self,):
        self.unet.train()
        self.diffusion.train()            
        
    @torch.inference_mode()
    def eval_mode(self,):
        self.unet.eval()
        self.diffusion.eval()
 