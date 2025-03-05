import torch
import cv2
from einops import rearrange

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