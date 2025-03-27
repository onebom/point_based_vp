import cv2
import numpy as np

import torch

def draw_point_tracks(clips, clip_track):
    #4 T C H W : clips
    #C,T,PN : pred_track
    clip_gt, clip_pred = clips[1], clips[3]

    for trj_idx in range(clip_track.size(2)):
        trj = clip_track[:,:,trj_idx]
        clip_gt, clip_pred = draw_track(clip_gt, clip_pred, trj, (0,255,0))
        
    clips[1], clips[3] = clip_gt, clip_pred
    return clips

def draw_track(clip_gt, clip_pred, trj, colors):
    T,C,H,W = clip_gt.shape

    for f_idx in range(clip_gt.size(0)):
        frame_gt, frame_pred = clip_gt[f_idx], clip_pred[f_idx] # C H W
        point = trj[:,f_idx] # C

        frame_gt_pointed = draw_points(frame_gt.cpu().numpy(), point, colors = colors) # c h w
        frame_pred_pointed = draw_points(frame_pred.cpu().numpy(), point, colors = colors)

        clip_gt[f_idx] = torch.tensor(frame_gt_pointed)
        clip_pred[f_idx] = torch.tensor(frame_pred_pointed)

    return clip_gt, clip_pred

def draw_points(frame, point, colors):
    frame = frame.transpose(1,2,0)
    frame = (frame*255).astype(np.uint8) # 64,64,3
 
    x, y = int(point[0]), int(point[1])
    if 0<=x<frame.shape[1] and 0<=y<frame.shape[0]:
        frame = cv2.circle(frame.copy(), (x, y), radius=1, color=colors, thickness=-1)
    
    return (frame/255.0).transpose(2,0,1)