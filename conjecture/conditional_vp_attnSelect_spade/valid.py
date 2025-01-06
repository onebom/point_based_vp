import os

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

from datasets.dataset import normalize_img
from datasets.builder import build_dataloader, build_dataset
from utils.motion_cond import create_motion_cond
from model.video_direct_diffusion import VideoDirectDiffusion

from metrics.calculate_fvd    import get_feats, calculate_fvd2
from metrics.calculate_psnr   import calculate_psnr2
from metrics.calculate_ssim   import calculate_ssim2
from metrics.calculate_lpips  import calculate_lpips2

def metric_stuff(metric):
    avg_metric, std_metric = metric.mean().item(), metric.std().item()
    conf95_metric = avg_metric - float(st.norm.interval(confidence=0.95, loc=avg_metric, scale=st.sem(metric))[0])
    return avg_metric, std_metric, conf95_metric

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=r'/data/onebom/project/point_based_vp/ongoing/wo_cond_original_vp/config/cityscapes/val_without_motion.yaml')

    args = parser.parse_args()
    return args

@torch.inference_mode()
def valid(cfg, accelerator, model, valid_loader, motion_cond_predictor, global_step):
    start = timeit.default_timer()
    
    model = accelerator.unwrap_model(model)
    model.eval_mode()
    model.to(accelerator.device)
    
    total_pred_frames = cfg.dataset.valid_params.pred_frames
    cf, pf = cfg.dataset.train_params.cond_frames, cfg.dataset.train_params.pred_frames
    
    from math import ceil
    NUM_ITER    = ceil(cfg.dataset.valid_params.num_videos / cfg.dataset.valid_params.batch_size)
    NUM_AUTOREG = ceil(total_pred_frames / cfg.dataset.train_params.pred_frames)
    
    # b t c h w [0-1]
    origin_vids = []
    result_vids = []
    result_motion_cond = []

    for i_iter, batch in enumerate(valid_loader):
        if i_iter >= NUM_ITER: break

        data = batch.to(accelerator.device) # b c t h w
        bs = data.size(0)
        
        data = repeat(data, 'b c t h w -> (b n) c t h w', n=cfg.dataset.valid_params.num_samples)
        
        result_frames = [] 
        result_motions = []
        
        cond_frames = data[:,:,:cf]
        for i_autoreg in range(NUM_AUTOREG):
            origin_frames = data[:,:,cf+i_autoreg*pf:cf+(i_autoreg+1)*pf]
            
            motion_cond=None
            if cfg.dataset.cond_params.cond_type is not None:
                total_frames = data[:,:,i_autoreg*pf:cf+(i_autoreg+1)*pf] # origin video
                # total_frames = torch.cat([cond_frames, origin_frames], dim=2) #reproduced video 
                motion_cond = create_motion_cond(total_frames, motion_cond_predictor, cfg.dataset.cond_params) 
                motion_cond = motion_cond.to(accelerator.device)
                
                result_motions.append(motion_cond.unsqueeze(1))#[b 1 c cf+pf pn] or [b 1 c cf+pf h w]
                motion_cond = motion_cond/cfg.dataset.dataset_params.frame_shape
            
            pred_frames = model.sample_video(cond_frames = normalize_img(cond_frames), 
                                            gt_frames = normalize_img(origin_frames), 
                                            motion_cond = motion_cond) 
            
            pred_frames = torch.stack(list(pred_frames), dim=0)
            print(f'[{i_autoreg+1}/{NUM_AUTOREG}] i_pred_video: {pred_frames.shape}')
            result_frames.append(pred_frames)
            
            cond_frames = torch.cat([cond_frames, pred_frames], dim=2)
            cond_frames = cond_frames[:,:,-cf:]
            
            torch.cuda.empty_cache()

        result_frames = torch.cat(result_frames, dim=2)
        
        result_vids.append(torch.cat([data[:,:,:cf],result_frames],dim=2))
        origin_vids.append(data[:,:,:cf+total_pred_frames])
        
        if result_motions:
            result_motions = torch.cat(result_motions, dim=1) #[b NumAuto c cf+pf pn] or [b NumAuto c cf+pf h w]
            result_motion_cond.append(result_motions)
        
        print(f'[{i_iter+1}/{NUM_ITER}] test videos generated.')
        print(f'result_video generated: {result_vids[-1].shape}')
        
        torch.cuda.empty_cache()
    
    # (b n) c t h w
    print('generating Done')
    origin_vids = torch.cat(origin_vids, dim=0) # [B, C, T, H, W]
    result_vids = torch.cat(result_vids, dim=0) # [B, C, T, H, W]
    
    # (b n) c t h w -> b n t c h w
    origin_vids = rearrange(origin_vids, '(b n) c t h w -> b n t c h w', n = cfg.dataset.valid_params.num_samples)
    result_vids = rearrange(result_vids, '(b n) c t h w -> b n t c h w', n = cfg.dataset.valid_params.num_samples)
    # origin_videos = rearrange(origin_videos, 'b c t h w -> b t c h w')
    # result_videos = rearrange(result_videos, 'b c t h w -> b t c h w')
    
    if result_motion_cond:
        result_motion_cond = torch.cat(result_motion_cond, dim=0) # [B, NUM_AUTOREG, C, cf+pf, PN] or [B, NUM_AUTOREG, C, cf+pf, H, W]
        if cfg.dataset.cond_params.cond_type == "point_track":
            result_motion_cond = rearrange(result_motion_cond, '(b n) auto c t pn -> b n auto c t pn', n = cfg.dataset.valid_params.num_samples)
        elif cfg.dataset.cond_params.cond_type == "flow":
            result_motion_cond = rearrange(result_motion_cond, '(b n) auto c t h w -> b n auto c t h w', n = cfg.dataset.valid_params.num_samples)
        print("result motion shape: ", result_motion_cond.shape)
    
    print("original video shape: ", origin_vids.shape)
    print("result video shape: ", result_vids.shape)    
    ############################ fvd ################################
    # b n t c h w 
    fvd_list = []

    origin_feats = get_feats(          origin_vids[:, 0]                           , torch.device("cuda"), mini_bs=16)
    result_feats = get_feats(rearrange(result_vids, 'b n t c h w -> (b n) t c h w'), torch.device("cuda"), mini_bs=16)
    # avg_fvd = calculate_fvd2(origin_feats, result_feats)

    for traj in tqdm(range(cfg.dataset.valid_params.num_samples), desc='fvd_feature'):
        result_feats_ = get_feats(result_vids[:, traj], torch.device("cuda"), mini_bs=16)
        fvd_list.append(calculate_fvd2(origin_feats, result_feats_))
    
    # print(avg_fvd, fvd_list)
    print(fvd_list)
    fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = metric_stuff(np.array(fvd_list))

    ####################### psnr ssim lpips ###########################

    psnr_list  = []
    ssim_list  = []
    lpips_list = []
    select_scores = []

    for i_bs in tqdm(range(len(result_vids))):
        # get [n t c h w]
        a_origin_videos = origin_vids[i_bs, :, cf:]
        a_result_videos = result_vids[i_bs, :, cf:]
        psnr_list.append (calculate_psnr2 (a_origin_videos, a_result_videos))
        ssim_list.append (calculate_ssim2 (a_origin_videos, a_result_videos))
        lpips_list.append(calculate_lpips2(a_origin_videos, a_result_videos, torch.device("cuda")))
        select_scores.append([ np.abs(origin_feats[i_bs]-result_feats[i_bs*cfg.dataset.valid_params.num_samples+i]).sum() for i in range(cfg.dataset.valid_params.num_samples) ])
    
    selected_index = np.argmin(np.array(select_scores), axis=-1)
    best_videos = torch.from_numpy(np.array([result_vids[i, selected_index[i]].cpu() for i in range(selected_index.shape[0])]))
    best_feats = get_feats(best_videos, torch.device("cuda"), mini_bs=16)
    fvd_best = calculate_fvd2(origin_feats, best_feats)
        
    origin_feats_all = get_feats(origin_vids[:, 0, cf:], torch.device("cuda"), mini_bs=16)
    result_feats_all = get_feats(rearrange(result_vids, 'b n t c h w -> (b n) t c h w')[:,cf:], torch.device("cuda"), mini_bs=16)
    fvd_all = calculate_fvd2(origin_feats_all, result_feats_all)

    avg_psnr, std_psnr, conf95_psnr    = metric_stuff(np.array(psnr_list))
    avg_ssim, std_ssim, conf95_ssim    = metric_stuff(np.array(ssim_list))
    avg_lpips, std_lpips, conf95_lpips = metric_stuff(np.array(lpips_list))
    # avg_fvd, std_fvd, conf95_fvd = metric_stuff(np.array(fvd_list))

    vid_metrics = {
        'psnr':  avg_psnr,  'psnr_std':  std_psnr,  'psnr_conf95':  conf95_psnr,
        'ssim':  avg_ssim,  'ssim_std':  std_ssim,  'ssim_conf95':  conf95_ssim,
        'lpips': avg_lpips, 'lpips_std': std_lpips, 'lpips_conf95': conf95_lpips,
        'fvd_all': fvd_all, 'fvd_best':  fvd_best, 'fvd_traj_mean': fvd_traj_mean, 'fvd_traj_std': fvd_traj_std, 'fvd_traj_conf95': fvd_traj_conf95
    }

    print(vid_metrics)

    end = timeit.default_timer()
    delta = end - start

    print("[ fvd_all  ]", fvd_all      )
    print("[ fvd_best ]", fvd_best     )
    print("[ ssim     ]", avg_ssim     )
    print("[ psnr     ]", avg_psnr     )
    print("[ lpips    ]", avg_lpips    )
    print("[ time     ]", delta , 'seconds.')

    log_dir = os.path.join(cfg.dataset.valid_params.log_dir, f"step{global_step}_{cfg.dataset.valid_params.num_videos}")
    os.makedirs(log_dir, exist_ok=True)

    with open(f'{log_dir}/metrics.txt', 'w') as f:
        f.write(f"[ fvd_all  ] {fvd_all      }\n")
        f.write(f"[ fvd_best ] {fvd_best    }\n")
        f.write(f"[ ssim     ] {avg_ssim     }\n")
        f.write(f"[ psnr     ] {avg_psnr     }\n")
        f.write(f"[ lpips    ] {avg_lpips    }\n")
        f.write(f"[ time     ] {delta        } seconds.\n")

    ################################# save visualize ###########################
    # b n t c h w

    from utils.visualize import visualize

    print(origin_vids.shape, result_vids.shape)
    
    # result visulization 
    visualize_path = os.path.join(log_dir,'vis_result')
    visualize(
        save_path=visualize_path,
        origin=origin_vids[:,0],
        result=result_vids[:,0],
        motion_cond = result_motion_cond[:,0],
        motion_type = cfg.dataset.cond_params.cond_type,
        save_pic_num=10,
        select_method='linspace',
        grid_nrow=4,
        save_gif_grid=False,
        save_pic_row=True,
        save_gif=True,
        save_pic=True, 
        skip_pic_num=1,
        epoch_or_step_num=global_step, 
        cond_frame_num=cf,
    )

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
    model = VideoDirectDiffusion(cfg.model)
    model.to(accelerator.device)
    
    motion_cond_predictor=None
    if cfg.dataset.cond_params.cond_type == "point_track":
        motion_cond_predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        motion_cond_predictor.to(accelerator.device)
    elif cfg.dataset.cond_params.cond_type == "flow":
        from utils.raft import Raft
        motion_cond_predictor = Raft('./utils/raft/raft_model/raft_sintel_iter20_240x320.onnx')
    
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

    valid(cfg, accelerator, model, val_loader, motion_cond_predictor, global_step=global_step)


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