import os
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
import argparse
from utils.config import load_config

import torch
import torch.backends.cudnn as cudnn

import timeit
from tqdm import tqdm
import numpy as np
from einops import rearrange, repeat
import scipy.stats as st

from accelerate import Accelerator
from accelerate.utils import set_seed

from datasets.dataset import normalize_img, normalize_trj, unnormalize_trj
from datasets.builder import build_dataloader, build_dataset
from utils.motion_cond import create_motion_cond, expand_motion_info, filtered_point_diff
from model.motionPredictor import TrackMotionModel

from metrics.calculate_ade_fde import calculate_ade_fde
from utils.visualize import visualize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=r'./config/valid.yaml')

    args = parser.parse_args()
    return args

@torch.inference_mode()
def valid(cfg, accelerator, model, valid_loader, mean, std, motion_cond_predictor, global_step):
    if cfg.train_params.seed is not None:
        set_seed(cfg.train_params.seed)

    start = timeit.default_timer()

    model = accelerator.unwrap_model(model)
    model.eval()
    model.to(accelerator.device)
    
    total_pred_frames = cfg.dataset.valid_params.pred_frames
    cf, pf = cfg.dataset.train_params.cond_frames, cfg.dataset.train_params.pred_frames
    trj_dim = cfg.dataset.cond_params.point_track_params.track_dim

    from math import ceil
    NUM_ITER    = ceil(cfg.dataset.valid_params.num_videos / cfg.dataset.valid_params.batch_size)
    NUM_AUTOREG = ceil(total_pred_frames / cfg.dataset.train_params.pred_frames)

    ade_avg = 0
    fde_avg = 0
    for i_iter, batch in enumerate(valid_loader):
        if i_iter >= NUM_ITER: break

        data, seg = batch["data"].to(accelerator.device), batch["seg"].to(accelerator.device) # b c t h w
        data = repeat(data, 'b c t h w -> (b n) c t h w', n=cfg.dataset.valid_params.num_samples)
        
        gt = []
        pred = []
        frames = []
        for auto_step in tqdm(range(NUM_AUTOREG), desc='sampling loop'):
            total_frames = data[:,:,auto_step*pf:cf+(auto_step+1)*pf]
            motion=create_motion_cond(total_frames, motion_cond_predictor, cfg.dataset.cond_params)
            motion = motion[:,:trj_dim,:,::5].to(accelerator.device)     
            motion, val_track_num = filtered_point_diff(motion, total_frames[:,:,:cf])      

            cond_mo = motion[:,:,:cf].clone()

            cond_mo = normalize_trj(cond_mo, mean, std)
            cond_mo = expand_motion_info(cond_mo)

            cond_vid = normalize_img(total_frames)

            pred_mo = model(cond_mo, cond_vid[:,:,:cf])
            pred_mo = unnormalize_trj(pred_mo[:,:trj_dim], mean, std) 

            pred.append(pred_mo)
            gt.append(motion)
            frames.append(total_frames)

            # torch.cuda.empty_cache()
        
        res_pred=torch.stack(pred, dim=1) # B,auto,C,Tf,PN     
        res_gt=torch.stack(gt, dim=1) # B,auto,C,T,PN 
        res_frames=torch.stack(frames, dim=1)

        res_pred = rearrange(res_pred, 'b n c t pn -> b n pn t c')
        res_gt = rearrange(res_gt, 'b n c t pn -> b n pn t c')
        res_frames = rearrange(res_frames, 'b n c t h w -> b n t c h w')
        
        label = res_gt[:,:,:,cf:,:]
        ade, fde = calculate_ade_fde(res_pred, label)
        ade_avg += ade.mean()
        fde_avg += fde.mean()

        visualize_path = os.path.join(cfg.checkpoint.output, f'vdm_steps_{global_step}','val_vis_result',f'batch_{i_iter}')
        visualize(
            save_path=visualize_path,
            pred=res_pred,
            gt=res_gt,
            frames = res_frames,
            seg_map = None,
            score_info = None
        )

    print("Total frame performance")    
    print("[ADE    {:.5f}]".format(ade_avg/NUM_ITER))
    print("[FDE   {:.5f}]".format(fde_avg/NUM_ITER))

    log_dir = os.path.join(cfg.dataset.valid_params.log_dir, f"step{global_step}_{cfg.dataset.valid_params.num_videos}")
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/metrics.txt', 'w') as f:
        f.write("[ADE    {:.5f}]\n".format(ade_avg/NUM_ITER))
        f.write("[FDE    {:.5f}]\n".format(fde_avg/NUM_ITER))

def main(cfg):
    cudnn.enabled = True
    cudnn.benchmark = True
    
    if cfg.train_params.seed is not None:
        set_seed(cfg.train_params.seed)
    
    accelerator = Accelerator(
        mixed_precision=cfg.train_params.mixed_precision,
    )
    print(f"Assigned device: {accelerator.device}")
    print(f"Number of processes (GPUs) used by Accelerator: {accelerator.num_processes}")
    
    ### ::: 1. load model
    model = TrackMotionModel(cfg.model)
    model.to(accelerator.device)
    
    motion_cond_predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    motion_cond_predictor.to(accelerator.device)
    
    ### ::: 2. load data
    train_dataset, val_dataset = build_dataset(cfg.dataset)
    train_loader, val_loader = build_dataloader(cfg.dataset, train_dataset, val_dataset)
    
    model, val_loader = accelerator.prepare(
        model, val_loader
    )
    accelerator.load_state(cfg.checkpoint.resume)
    global_step = int(cfg.checkpoint.resume.split("/")[-1].split("_")[-1].split(".")[0])
    
    def count_parameters(model):
        res = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"count_training_parameters: {res}")
        res = sum(p.numel() for p in model.parameters())
        print(f"count_all_parameters:      {res}")
    count_parameters(model)
        
    print("***** Running Inference *****")
    print(f"  Num examples = {len(val_dataset)}")
    print(f"  Instantaneous batch size per device = {cfg.dataset.valid_params.batch_size}")
    print(f"  Condition frame = {cfg.dataset.valid_params.cond_frames}")
    print(f"  Prediction frame = {cfg.dataset.valid_params.pred_frames}")
    print(f"  Total frame = {cfg.dataset.valid_params.total_frames}")

    mo_mean, mo_std = train_dataset.get_data_mean_std()
    mo_mean, mo_std = mo_mean.to(accelerator.device), mo_std.to(accelerator.device)
    valid(cfg, accelerator, model, val_loader, mo_mean, mo_std, motion_cond_predictor, global_step=global_step)


if __name__ == '__main__':    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    args = parse_args()
    cfg = load_config(args.config)
    
    os.makedirs(cfg.train_params.save_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint.output, exist_ok=True)
    
    print(cfg)
    main(cfg)