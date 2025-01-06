import cv2
import numpy as np
from einops import rearrange

import torch

UNKNOWN_FLOW_THRESH = 1e7

def unnormalize_img(t):
    return (t + 1) * 0.5

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

def draw_point_tracks( two_clips, clip_point_track):
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(0, 1, clip_point_track.size(2)))
    
    #2 CFPF C H W : two_clips
    #C CFPF PN(64x64=4096) : clip_point_track
    origin, pred = two_clips[0], two_clips[1]
    
    pointed_clips=[]
    for f_idx in range(two_clips.size(1)):
        origin_frame, pred_frame = origin[f_idx], pred[f_idx]
        frame_point = clip_point_track[:,f_idx]
        
        pointed_origin = draw_points(origin_frame.cpu().numpy(), frame_point, colors = colors, interval=4) # c h w
        pointed_pred = draw_points(pred_frame.cpu().numpy(), frame_point, colors = colors, interval=4)
        
        pointed_clip = torch.stack([torch.tensor(pointed_origin),torch.tensor(pointed_pred)]) # 2 h w c
        pointed_clip = pointed_clip.permute(0,3,1,2) # 2 c h w 
        pointed_clips.append(pointed_clip)
        
    pointed_clips = torch.stack(pointed_clips) #t 2 c h w 
    pointed_clips = pointed_clips.permute(1,0,2,3,4)
    
    return pointed_clips

def draw_points(frame, track, colors, interval=None):
    c, point_num = track.shape
    frame = frame.transpose(1,2,0)
    frame = (frame*255).astype(np.uint8) # 64,64,3

    if interval == None:
        points=range(point_num)
    else:
        points = list(range(0,point_num))
        points = np.array(points).reshape(int(point_num**0.5), int(point_num**0.5))
        points = points[::4, ::4] 
        points = points.ravel().tolist()
        
    for point_idx in points:
        x = int(track[0, point_idx])
        y = int(track[1, point_idx])
        if 0<=x<frame.shape[1] and 0<=y<frame.shape[0]:
            frame = cv2.circle(frame.copy(), (x, y), radius=1, color=colors[point_idx][:3] * 255, thickness=-1)
    
    return frame