import torch
import torch.nn as nn

from model.diffusion import GaussianDiffusion

from model.util import exists, import_module
from datasets.dataset import normalize, denormalize

class VideoDirectDiffusion(nn.Module): 
    def __init__(self, config, autoencoder=None, is_train=True):
        super().__init__()

        unet_class = import_module(f"model.multimodal_unet", config.unet.type) ## DirectUnet3D_motion
        self.noise_cfg = config.diffusion.noise_params ## using pyoco
        
        self.unet_params = config.unet.model_params
        self.diffusion_params = config.diffusion.diffusion_params
        self.unet_type = config.unet.type
        
        if self.unet_type == "Unet3D_SequentialCondAttn":
            self.unet_cond_params = config.unet.cond_params
            self.unet = unet_class(**self.unet_params) 
            
        self.diffusion = GaussianDiffusion(self.unet, **self.diffusion_params, noise_cfg=self.noise_cfg)
        
        self.autoencoder = autoencoder ## None
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()            
            
    def forward(self, x, action=None):
        B, C, T, PN = x.shape

        track_dim = self.unet_cond_params.track_dim
        tc = self.unet_params.cond_num

        trj = x[:,:track_dim,:,:]        
        expanded_trj = self.expand_motion_info(trj) 
        
        expanded_trj, _, _ = normalize(expanded_trj)

        cond_trj = expanded_trj[:,:,:tc]
        pred_trj = expanded_trj[:,:,tc:]

        cond = {"traj_f": x[:,track_dim:, :tc], "action": action}

        diffusion_loss, _ = self.diffusion(cond_trj, pred_trj, cond = cond)        
        return diffusion_loss
        
    
    @torch.inference_mode()
    def sample_video(self, x, action=None):
        B, C, T, PN = x.shape

        track_dim = self.unet_cond_params.track_dim
        tc = self.unet_params.cond_num

        trj = x[:,:track_dim,:,:]        
        expanded_trj = self.expand_motion_info(trj) 

        expanded_trj, mean, std = normalize(expanded_trj)

        cond_trj = expanded_trj[:,:,:tc]
        pred_trj = expanded_trj[:,:,tc:]

        cond = {"traj_f": x[:,track_dim:, :tc], "action": action}

        pred_trj = self.diffusion.sample(cond_trj, pred_trj, cond = cond)
        
        pred_trj = denormalize(pred_trj, mean, std)

        return pred_trj
    
    def expand_motion_info(self, traj):
        B, C, T, PN = traj.shape
        
        x, y, v = traj[:, 0], traj[:, 1], traj[:, 2] #B,T,PN
        
        v_x = torch.diff(x, dim=1, prepend=x[:, :1]) #B,T,PN
        v_y = torch.diff(y, dim=1, prepend=x[:, :1]) #B,T,PN
        v_t = torch.sqrt(v_x**2 + v_y**2) 
        
        a_x = torch.diff(v_x, dim=1, prepend=v_x[:, :1])
        a_y = torch.diff(v_y, dim=1, prepend=v_y[:, :1])
        a_t = torch.sqrt(a_x**2 + a_y**2)
        
        theta_t = torch.atan2(v_y, v_x)
        
        rel_x = x - x[:, 0:1]
        rel_y = y - y[:, 0:1]
        
        features = torch.stack([x, y, v, v_t, a_t, theta_t, rel_x, rel_y], dim=1)
        
        return features
    
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
 