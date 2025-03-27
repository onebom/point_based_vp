import einops
import os
import mediapy as media

import torch
from utils.motion_vis import draw_point_tracks

import matplotlib.pyplot as plt
cmap = plt.get_cmap("viridis") 

def visualize(save_path, pred, gt, motion=None):
    os.makedirs(save_path, exist_ok=True)

    index = [int(i) for i in range(len(gt))]
    save_sample(save_root_path=save_path, index=index, 
                pred=pred.cpu(), gt=gt.cpu(), motion= motion)

def save_sample(save_root_path, index, pred, gt, motion):
    
    save_path = os.path.join(save_root_path, "gif")
    os.makedirs(save_path, exist_ok=True)
    save_gif(save_path, index, pred, gt)

    save_path = os.path.join(save_root_path, "pic_row_30")
    os.makedirs(save_path, exist_ok=True)   
    save_pic_row(save_path, index, pred, gt)

    if motion is not None:
        save_path = os.path.join(save_root_path, "pic_row_motion")
        os.makedirs(save_path, exist_ok=True) 
        save_pic_row_motion(save_path, index, pred, gt, motion.cpu())

def save_gif(save_path, index, pred, gt):
    gt = einops.rearrange(gt, "b t c h w -> b t h w c")
    pred = einops.rearrange(pred, "b t c h w -> b t h w c")

    for i in range(len(index)):
        media.write_video(os.path.join(save_path, f"{index[i]:02d}_gt.gif"), 
                          gt[i].squeeze().numpy(), 
                          codec='gif', 
                          fps=20)
        
        media.write_video(os.path.join(save_path, f"{index[i]:02d}_pred.gif"), 
                          pred[i].squeeze().numpy(), 
                          codec='gif', 
                          fps=20)

def save_pic_row(save_path, index, pred, gt, cf=4):
    all_video = torch.stack([gt, pred]) # 2, B, T, C, H, W
    
    for i in range(len(index)):
        two_video = all_video[:,i]
        two_video[1, :cf] = 1.0 

        two_video = einops.rearrange(two_video, "b t c h w -> (b h) (t w) c")
        media.write_image(os.path.join(save_path, f"{index[i]:02d}_sample.png"), two_video.squeeze().numpy())


def save_pic_row_motion(save_path, index, pred, gt, motion, cf=4):
    B, NUM_AUTOREG, C, T, PN = motion.shape #B,N,C,T(6),PN
    pf = T-cf
    all_video = torch.stack([gt,gt,pred,pred]) # 4, B, T(30), C, H, W

    for i in range(len(index)):
        four_video = all_video[:,i]

        for auto_idx in range(NUM_AUTOREG):
            four_clip = four_video[:, auto_idx*pf:(auto_idx+1)*pf+cf] #4,T,C,H,W 
            clip_motion = motion[i, auto_idx] # C,T,PN
        
            pointed_clip = draw_point_tracks(four_clip, clip_motion) # 4, T, C, H, W 
            pointed_clip[2, :cf] = 1.0
            pointed_clip[3, :cf] = 1.0

            pointed_clip = einops.rearrange(pointed_clip, "b t c h w -> (b h) (t w) c")

            clip_save_path = os.path.join(save_path, f"{index[i]:02d}_sample")
            os.makedirs(clip_save_path, exist_ok=True)
            media.write_image(os.path.join(clip_save_path, f"autonum_{auto_idx}.png"), pointed_clip.squeeze().numpy())

