import os
import sys

import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK

from datasets.builder import build_dataloader, build_dataset
from utils.config import load_config
from utils.motion_cond import create_motion_cond

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="")
    args = parser.parse_args()
    return args

def main(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps,
        mixed_precision=cfg.train_params.mixed_precision,
        log_with= "wandb" if cfg.wandb.enable else None,
        project_dir=cfg.checkpoint.output,
        kwargs_handlers=[DDPK(find_unused_parameters=True)]
    )
    
    # 1. cond_predictor 불러오기 
    motion_cond_predictor=None
    if cfg.dataset.cond_params.cond_type == "point_track":
        motion_cond_predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        motion_cond_predictor.to(accelerator.device)
    elif cfg.dataset.cond_params.cond_type == "flow":
        from utils.raft import Raft
        motion_cond_predictor = Raft('./utils/raft/raft_model/raft_sintel_iter20_240x320.onnx')
    
    # 2. clip과 파일 이름과 함께 불러오기    
    train_dataset, val_dataset = build_dataset(cfg.dataset)
    train_loader, _ = build_dataloader(cfg.dataset, train_dataset, val_dataset)
    
    for _,batch in tqdm(enumerate(train_loader)):
        clips_past, clips_future, clips_path = batch['cond'], batch['gt'], batch['clip_paths']
        clips_past, clips_future = clips_past.to(accelerator.device), clips_future.to(accelerator.device)
        clips_path = np.array(clips_path).transpose() # B,T

        origin_vids = torch.cat([clips_past, clips_future], dim=2)
        motion_cond = create_motion_cond(origin_vids, motion_cond_predictor, cfg.dataset.cond_params) #b c t pn
        
        if cfg.dataset.cond_params.cond_type == "point_track":
            # motion_cond shape : [batch, c, frame, PN] PN => [0,3]기준으로 각 4096, 총 8192 생성
            # clips_path shape : [frame_idx, batch_idx]
            B,C,T,PN = motion_cond.shape
            point_num = PN//2
            
            save_root_path = clips_path[:,cfg.dataset.cond_params.point_track_params.guery_frame_idx] # B,2[0기준, 3기준]
            for b_idx in range(save_root_path.shape[0]):
                save_path = save_root_path[b_idx]
                save_ref_frame_num = [path.split("/")[-1].split("_")[2] for path in save_root_path[b_idx]] ## [0000,0003]
                
                #save_ref_frame_num[0] : 0000
                track_save_path = save_path[0].split(".")[0] 
                track_save_path  = track_save_path.replace('/leftImg8bit_sequence/','/motion_condition2/point_track/')
                os.makedirs(track_save_path, exist_ok=True)
                
                track_motion = motion_cond[b_idx, : , :, :point_num] # C,T,point_num
                path_name = f"from_{str(int(save_ref_frame_num[0])).zfill(6)}.npy"
                np.save(os.path.join(track_save_path, path_name), track_motion)
                
                # track_motion2 = motion_cond[b_idx, : , :, point_num:]        
                # path_name = f"from_{str(int(save_ref_frame_num[1])).zfill(6)}.npy"
                # np.save(os.path.join(track_save_path, path_name), track_motion2)
                            
        elif cfg.dataset.cond_params.cond_type == "flow":
            B, C, T, H, W = motion_cond.shape
            for b_idx in range(clips_path.shape[0]):
                for f_idx, frame_path in enumerate(list(clips_path[b_idx])):
                    flow_save_path = frame_path.split(".")[0] 
                    flow_save_path  = flow_save_path.replace('/leftImg8bit_sequence/','/motion_condition/flow/')
                    os.makedirs("/".join(flow_save_path.split("/")[:-1]), exist_ok=True)
                    
                    flow_motion = motion_cond[b_idx,:,f_idx]
                    np.save(flow_save_path+".npy", flow_motion)

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg)