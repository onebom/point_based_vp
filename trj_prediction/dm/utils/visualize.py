import einops
import os
import mediapy as media
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.misc import grid2fig
from PIL import Image
import imageio

from utils.motion_vis import draw_point_tracks, draw_flows

import matplotlib.pyplot as plt
cmap = plt.get_cmap("viridis") 

def visualize(save_path, pred, gt, frames, cond_frame_num, skip_pic_num=1, grid_nrow=4):
    # assert pred.shape == gt.shape, f"pred ({pred.shape}) gt ({gt.shape}) shape are not equal."
    os.makedirs(save_path, exist_ok=True)
    
    # 전부 다
    index = [int(i) for i in range(len(pred))]
    save_sample(save_root_path=save_path,index=index, 
                pred=pred.cpu(), gt=gt.cpu(), frames=frames.cpu(), 
                cond_frame_num=cond_frame_num)


def save_sample(save_root_path, index, pred, gt, frames, cond_frame_num):
    B, NUM_AUTOREG, PN, T, C  = gt.shape
    all_video = torch.stack([frames,frames,frames]) # 3 B N T C H W
            
    for i in range(len(index)):
        three_video = all_video[:,i] #3 N T C H W
                                
        for auto_idx in range(NUM_AUTOREG):
            three_clip = three_video[:,auto_idx] #3 T C H W
            
            pred_track = pred[i, auto_idx] # PN, T, C
            gt_track = gt[i, auto_idx]
            pointed_clip = draw_point_tracks(three_clip, pred_track, gt_track) #3 T C H W

            pointed_clip = einops.rearrange(pointed_clip, "b t c h w -> (b h) (t w) c")
            
            save_path = os.path.join(save_root_path, f"pic_row_{index[i]}")
            os.makedirs(save_path, exist_ok=True)
            media.write_image(os.path.join(save_path, f"autonum_{auto_idx}.png"), pointed_clip.squeeze().numpy())
            


### ==========================================================
def hidden_f_vis(hidden_features, save_dir="hidden_features"):
    os.makedirs(save_dir, exist_ok=True)
    
    hidden_features = hidden_features[0] # visulize only first batch
    auto_num = len(hidden_features)
    h_num = len(hidden_features[0])
    
    B,_,T,_,_ = hidden_features[0][0].shape
    H_max = max(f.shape[3] for f in hidden_features[0])  # 최대 H
    W_max = max(f.shape[4] for f in hidden_features[0])  # 최대 W
    
    f_dic = {key: [] for key in range(auto_num*T)}
    for h_idx in range(h_num):
        for auto_idx in range(auto_num):  
            f=hidden_features[auto_idx][h_idx]
            for t_idx in range(f.shape[2]):
                time = t_idx+(auto_idx*2)
                
                t_feature = f[:,:,t_idx] # b,c,h,w
                t_f_resized = F.interpolate(t_feature, size=(H_max, W_max),
                                            mode='bilinear', 
                                            align_corners=False)
                t_f_img = t_f_resized.mean(dim=1).squeeze() #b, h, w
                f_dic[time].append(t_f_img)

    for key in f_dic.keys():
        hfs = torch.stack([norm_min_max(hf) for hf in f_dic[key]], dim=0)
        # hfs=torch.stack(f_dic[key],dim=0).unsqueeze(-1) # hf_num, b, h, w ,c
        hfs = einops.rearrange(hfs, "hf b h w -> (hf h) (b w)")
        
        hfs_colored = cmap(hfs.cpu().numpy())
        hfs_colored = hfs_colored[..., :3]
        
        save_path = os.path.join(save_dir, f"pred_{key}.png")
        media.write_image(save_path, hfs_colored)
    
def norm_min_max(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)

            
    