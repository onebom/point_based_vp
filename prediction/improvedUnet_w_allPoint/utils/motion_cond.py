import torch
import cv2
from einops import rearrange

import torch.nn.functional as F

def expand_motion_info(traj):
    B, C, T, PN = traj.shape
    
    x, y, vis = traj[:, 0], traj[:, 1], traj[:, 2] #B,T,PN
    
    v_x = torch.diff(x, dim=1, prepend=x[:, :1]) #B,T,PN
    v_y = torch.diff(y, dim=1, prepend=x[:, :1]) #B,T,PN
    v_t = torch.sqrt(v_x**2 + v_y**2) 
    
    a_x = torch.diff(v_x, dim=1, prepend=v_x[:, :1])
    a_y = torch.diff(v_y, dim=1, prepend=v_y[:, :1])
    a_t = torch.sqrt(a_x**2 + a_y**2)
    
    theta_t = torch.atan2(v_y, v_x)
    
    rel_x = x - x[:, 0:1]
    rel_y = y - y[:, 0:1]
    
    features = torch.stack([x, y, vis, v_t, a_t, theta_t, rel_x, rel_y], dim=1)
    
    return features

def filtered_point_diff(tracks, video_frames, patch_size=3, threshold=0.001):
    """
    param tracks: B,C,T,PN
    param video_frames: B,C,T,H,W
    param patch_size: 3x3
    param thershold: MSE 임계값
    """
    B, C, T, P = tracks.shape
    _, C_v, T_v, H, W = video_frames.shape  # C_v: 영상 채널 수

    half_patch = patch_size // 2

    # (B, C_v, T, H, W) → (B, T, C_v, H, W) 변환
    video_frames = rearrange(video_frames, 'b c t h w -> (b t) c h w')

    unfolded_patches = F.unfold(video_frames, kernel_size=patch_size, padding=half_patch)
    unfolded_patches = rearrange(unfolded_patches, '(b t) (c ph ph2) hw -> b t c (ph ph2) hw', 
                                 ph=patch_size, t=T_v, c=C_v) 

    x_coords = tracks[:,0,:T_v].long().clamp(half_patch, W - half_patch - 1)
    y_coords = tracks[:,1,:T_v].long().clamp(half_patch, H - half_patch - 1)  # (B, T, P)

    indices = y_coords * W + x_coords # (B, T, PN)
    indices = indices[:,:,None,None,:].expand(-1, -1, C_v, patch_size**2, -1) # (B, T, Cv, patch**2, PN)

    patches = unfolded_patches.gather(dim=-1, index=indices) # (B, T, C_v, Patch**2, PN)

    diff = (patches[...,:-1,:] - patches[...,1:,:]) ** 2  # (B, T, C_v, Patch**2-1, PN)
    mse_values = torch.mean(diff, dim=[1, 2])  # (B, Patch**2-1, P)
    avg_mse = torch.mean(mse_values, dim=1)  # (B, P)

    # Visibility 한프레임이라도 0이면 track 포함
    visibility_mask = (tracks[:, 2, :, :].sum(dim=1) == 0)  # (B, P) - 한 프레임이라도 0이면 True
    
    threshold_valid_mask = (avg_mse > threshold)

    track_num = threshold_valid_mask.sum(dim=-1)
    track_num = track_num.float().mean(dim=0)

    valid_mask = threshold_valid_mask | visibility_mask  # (B, P)
    valid_mask = valid_mask.unsqueeze(1).unsqueeze(1)
    masked_tracks = tracks * valid_mask.float()

    return masked_tracks, track_num

def intergrate_motion_feature(trj_f, visibility, repeat_num):
    # B, C, T, PN = trj_f.shape
    # B, 1, T, PN = vis.shape
    
    weighted_features = trj_f * visibility
    
    sum_weights = visibility.sum(dim=2, keepdim=True) # (B, 1, 1, point_num)
    sum_weights = sum_weights.clamp(min=1e-6)
    
    weighted_avg = weighted_features.sum(dim=2, keepdim=True) / sum_weights # (B, C=128, 1, point_num)
    
    trj_f_predInit = weighted_avg.repeat(1,1,repeat_num,1)
    return trj_f_predInit

#=====create_motionCond=======
def create_motion_cond(videos, cond_predictor, cond_params):
    B,C,T,H,W = videos.shape

    motion_cond = None
    if cond_params.cond_type == "point_track":
        motion_cond = create_point_tack(videos, cond_predictor, cond_params) # b c t pn
        
    elif cond_params.cond_type == "flow":
        motion_cond = create_flow(videos, cond_predictor) # b c t h w
    
    return motion_cond


def create_point_tack(videos, cond_predictor, cond_params):
    videos = videos.permute(0,2,1,3,4).contiguous().float() #B,T,C,H,W
    
    point_num = cond_params.point_track_params.point_grid**2
    g_idx = list(cond_params.point_track_params.guery_frame_idx)
    
    dim = cond_params.point_track_params.track_dim + cond_params.point_track_params.feature_dim
    point_track = torch.empty(videos.size(0),videos.size(1),point_num*len(g_idx), dim, dtype=torch.float32)
    
    for i, t_idx in enumerate(g_idx):
        # out: [b,t,point_num,2],[b,t,point_num]
        pred_tracks, pred_visibility, pred_features = cond_predictor(videos, 
                                                        grid_size=cond_params.point_track_params.point_grid, 
                                                        grid_query_frame=t_idx) 
        point_track_info = torch.cat((pred_tracks, pred_visibility.float().unsqueeze(-1), pred_features), dim=3)
        
        for frame_t in range(point_track_info.shape[1]):
            point_track[:,frame_t, point_num*i:point_num*(i+1)] = point_track_info[:,frame_t]
    
    point_track = rearrange(point_track, 'b t pn c -> b c t pn') 
    
    return point_track  

def create_flow(videos, cond_predictor):
    optical_flow = []
    for video in videos:
        video = rearrange(video, 'c t h w -> t h w c')
        frame_lst = list(torch.cat([video[:1],video], dim=0))
        
        video_flow=[]
        while len(frame_lst)>1:
            flow = cond_predictor(frame_lst[0].cpu().numpy(),frame_lst[1].cpu().numpy())
            flow = cv2.resize(flow, (videos.size(3),videos.size(4)))
            
            video_flow.append(torch.tensor(flow))
            frame_lst.pop(0)
        
        video_flow = torch.stack(video_flow, dim=0)
        optical_flow.append(video_flow)
    
    optical_flow = rearrange(torch.stack(optical_flow, dim=0), 'b t h w c -> b c t h w')    
    return optical_flow