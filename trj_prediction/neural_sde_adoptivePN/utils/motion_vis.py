import cv2
import numpy as np
from einops import rearrange

import torch

from utils.seg_info import seg_label

UNKNOWN_FLOW_THRESH = 1e7

##==============point track==============

def draw_point_tracks(clips, pred_track, gt_track, seg=None, score=None):
    #3 T C H W : clips
    #PN Tp C : pred_track
    tp = pred_track.shape[1]
    clip_gt, clip_pred = clips[1], clips[2]
    pred_track = torch.concat([gt_track[:,:-tp], pred_track], dim=1) # PN T C

    for trj_idx in range(pred_track.size(0)):

        gt_trj1, pred_trj1 = gt_track[trj_idx], pred_track[trj_idx]
        score_trj1 = score[trj_idx] if score is not None else None
        trj_class, trj_color = valid_track_2_print(gt_trj1, seg, score_trj1)

        if trj_class is not None:
            clip_gt, clip_pred = draw_track(clip_gt, clip_pred, gt_trj1, pred_trj1, trj_color)
        
    clips[1], clips[2] = clip_gt, clip_pred
    return clips

def draw_track(clip_gt, clip_pred, gt_trj, pred_trj, colors):
    T,C,H,W = clip_gt.shape

    for f_idx in range(clip_gt.size(0)):
        frame_gt, frame_pred = clip_gt[f_idx], clip_pred[f_idx] # C H W
        gt_point, pred_point = gt_trj[f_idx], pred_trj[f_idx] # C

        frame_gt_pointed = draw_points(frame_gt.cpu().numpy(), gt_point, colors = colors) # c h w
        frame_pred_pointed = draw_points(frame_pred.cpu().numpy(), pred_point, colors = colors)

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

def valid_track_2_print(trj, seg, score):

    trj_class = 1
    trj_color = (0,255,0)

    if seg is not None:
        start_x, start_y = trj[0,0], trj[0,1]
        trj_class=int(seg[int(start_y), int(start_x)])
    
        if trj_class in seg_label.keys():
            trj_color = seg_label[trj_class]['color']

        else:
            trj_class = None
            trj_color = None

    if (score is not None) and (score < 2):
        trj_class = None
        trj_color=None

    return trj_class, trj_color

##==============optical flow==============

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel
colorwheel = make_color_wheel()

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[0, :, :]
    v = flow[1, :, :]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

## drawing tools 

def draw_flows(index, motion_cond):    
    B, NUM_AUTOREG, C, CFPF, H, W = motion_cond.shape
    
    all_flows=[]
    pred_frame_num=4
    
    for i in range(len(index)):       
        video_motion_cond = motion_cond[i]

        flow_imgs=[]
        for auto_idx, frame_motion in enumerate(video_motion_cond):
            if auto_idx == 0:
                f_range = range(pred_frame_num)
            elif auto_idx ==  NUM_AUTOREG - 1:
                f_range = range(-pred_frame_num, 0)
            else: 
                f_range = range(-pred_frame_num,pred_frame_num-CFPF)
                
            for f_idx in f_range:
                flow = frame_motion[:,f_idx]
                flow_img = torch.tensor(draw_flow(flow)) # h w c
                flow_img = rearrange(flow_img, 'h w c -> c h w')
                flow_imgs.append(flow_img)
        
        flow_imgs=torch.stack(flow_imgs) # t c h w
        all_flows.append(flow_imgs)
    
    return torch.stack(all_flows) #b t c h w

def draw_flow(flow):
    C,H,W = flow.shape
    flow_img = flow_to_image(flow)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
    return cv2.resize(flow_img, (H,W))

