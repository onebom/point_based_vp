import argparse
import timeit
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
import math
from tqdm import tqdm
from datetime import timedelta
from einops import rearrange

import torch
import torch.backends.cudnn as cudnn

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate import InitProcessGroupKwargs

from torchinfo import summary
from contextlib import redirect_stdout
from safetensors.torch import load_file

from utils.config import load_config
from utils.meter import RunningAverageMeter
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lrscheduler
from utils.checkpoint import load_checkpoint_with_accelerator, save_checkpoint_with_accelerator
from utils.motion_cond import create_motion_cond, expand_motion_info, filtered_point_diff, intergrate_motion_feature
from utils.visualize import visualize

from datasets.builder import build_dataloader, build_dataset
from datasets.dataset import normalize_img, normalize_trj, unnormalize_trj

from model.video_direct_diffusion import VideoDirectDiffusion
from model.motionPredictor import TrackMotionModel

from metrics.calculate_fvd import calculate_fvd1
from metrics.calculate_psnr import calculate_psnr1
from metrics.calculate_ssim import calculate_ssim1
from metrics.calculate_lpips import calculate_lpips1

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config/train.yaml")
    args = parser.parse_args()
    return args

def train(cfg):
    import wandb
    if cfg.wandb.enable:
        wandb.init(project=cfg.wandb.project,
                   entity=cfg.wandb.entity, 
                   resume=cfg.checkpoint.auto_resume,
                   dir = cfg.train_params.save_dir
        )
        wandb.run.name = cfg.model_name
    
    if cfg.train_params.seed is not None:
        set_seed(cfg.train_params.seed)
    
    ipg_handler = InitProcessGroupKwargs(
        timeout=timedelta(seconds=10000)
        )

    ### ::: 1. setting Accelerator for multi-gpu 
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps,
        mixed_precision=cfg.train_params.mixed_precision,
        log_with= "wandb" if cfg.wandb.enable else None,
        project_dir=cfg.checkpoint.output,
        kwargs_handlers=[ipg_handler, DDPK(find_unused_parameters=True)]
    )
    print(f"Assigned device: {accelerator.device}")
    print(f"Number of processes (GPUs) used by Accelerator: {accelerator.num_processes}")
   
    ### ::: 2. load model
    model = VideoDirectDiffusion(cfg.model)
    model.to(accelerator.device)

    motion_tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    motion_tracker.to(accelerator.device)

    predictor_state = load_file(cfg.checkpoint.predictor_state, device='cpu')
    track_predictor = TrackMotionModel(cfg.mo_model)
    track_predictor.load_state_dict(predictor_state, strict=False)
    track_predictor.to(accelerator.device)

    for param in track_predictor.parameters():
        param.requires_grad = False
    track_predictor.eval()
    
    # Meter setting
    lr_meter = RunningAverageMeter()
    losses = RunningAverageMeter()
    
    ### ::: 3. load data
    train_dataset, test_dataset = build_dataset(cfg.dataset)
    train_loader, test_loader = build_dataloader(cfg.dataset, train_dataset, test_dataset)
                
    total_batch_size = cfg.dataset.train_params.batch_size * accelerator.num_processes * cfg.train_params.grad_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / total_batch_size)
    final_step = steps_per_epoch * cfg.train_params.max_epochs
    
    ### ::: 4. optimizer & lr_scheduler setting
    optimizer = build_optimizer(cfg.train_params, model)
    lr_scheduler = build_lrscheduler(cfg.train_params, optimizer, final_step=final_step)
    
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    
    ### ::: 5. load checkpoint
    if cfg.checkpoint.resume:
        global_step, epoch_cnt, lr_meter, losses = load_checkpoint_with_accelerator(cfg, accelerator, lr_meter, losses)
    else:
        global_step = 0
        epoch_cnt = 0
    
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.model_name)
    
    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {cfg.train_params.max_epochs}")
        print(f"  Instantaneous batch size per device = {cfg.dataset.train_params.batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {cfg.train_params.grad_accumulation_steps}")
        print(f"  Total optimization steps = {final_step}")
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))
    
    progress_bar = tqdm(range(global_step, final_step), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    ### ::: 6. training
    mo_mean, mo_std = train_dataset.get_data_mean_std()
    mo_mean, mo_std = mo_mean.to(accelerator.device), mo_std.to(accelerator.device)
    cf, pf = cfg.dataset.train_params.cond_frames, cfg.dataset.train_params.pred_frames
    trj_dim = cfg.dataset.cond_params.point_track_params.track_dim

    for epoch in range(epoch_cnt, cfg.train_params.max_epochs):
        for i_iter, batch in enumerate(train_loader):
            iter_end = timeit.default_timer()
            with accelerator.accumulate(model):                
                cond, gt, cond_motion = batch["cond"], batch["gt"], batch["motion"]
                cond, gt, cond_motion = cond.to(accelerator.device), gt.to(accelerator.device), cond_motion[:,:,:cf,::5].to(accelerator.device)
                cond_motion, _ = filtered_point_diff(cond_motion, cond)
                cond_mo_xy, cond_mo_fea = cond_motion[:,:trj_dim], cond_motion[:,trj_dim:]
                
                cond = normalize_img(cond) ### scale to [-1,1]
                gt = normalize_img(gt)

                cond_mo = normalize_trj(cond_mo_xy, mo_mean, mo_std) #b c t pn
                cond_mo = expand_motion_info(cond_mo)

                pred_mo = track_predictor(cond_mo, cond)
                pred_mo_fea = intergrate_motion_feature(cond_mo_fea, cond_mo_xy[:,-1:], repeat_num = pf)

                mo = torch.cat([cond_mo, pred_mo], dim=2)
                mo_fea = torch.cat([cond_mo_fea, pred_mo_fea], dim=2)
                mc = {"x": mo, "fea":mo_fea}

                with accelerator.autocast():
                    # template loss 사용안함. (=0.0)
                    loss = model(cond, gt, mc)
                    
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 0.3)
                    
                optimizer.step()    
                lr_scheduler.step()
                optimizer.zero_grad()
                
                losses.synchronize_and_update(accelerator, loss, global_step)
                lr_meter.update(lr_scheduler.get_last_lr()[0], global_step)
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                ### ::: 8. save model
                if global_step % cfg.train_params.save_ckpt_freq == 0:
                    save_checkpoint_with_accelerator(cfg, accelerator, global_step, epoch, lr_meter, losses)
                
                ### ::: 9. validation
                if global_step % cfg.train_params.valid_freq == 0:
                    if accelerator.is_main_process:
                        meters = valid(cfg, 
                                       accelerator, 
                                       model, 
                                       test_loader, 
                                       motion_tracker, 
                                       track_predictor, 
                                       mo_mean, mo_std, 
                                       global_step)
                        logs = {'FVD': meters['metrics/fvd'], 'SSIM' : meters['metrics/ssim'], 'PSNR' : meters['metrics/psnr'], 
                                'LPIPS' : meters['metrics/lpips']}
                        accelerator.log(logs, step=global_step)
                    model.train()
                   
            ### ::: 10. train logging         
            logs = {'loss': loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch}
            progress_bar.set_postfix(logs)
            accelerator.log(logs, step=global_step)
            if global_step >= final_step:
                break
        accelerator.wait_for_everyone()
    accelerator.end_training()
    
@torch.inference_mode()
def valid(cfg, 
          accelerator, 
          model, 
          val_loader, 
          motion_tracker, 
          track_predictor,
          mo_mean, mo_std, 
          global_step
          ):
    cudnn.enabled=True
    cudnn.benchmark=True
    if cfg.train_params.seed is not None:
        set_seed(cfg.train_params.seed)
    
    model = accelerator.unwrap_model(model)
    model.eval()
    model.to(accelerator.device)
    
    from math import ceil
    NUM_ITER    = ceil(cfg.dataset.valid_params.num_videos / cfg.dataset.valid_params.batch_size)    
    NUM_AUTOREG = ceil(cfg.dataset.valid_params.pred_frames / cfg.dataset.train_params.pred_frames)
    
    cf, pf = cfg.dataset.train_params.cond_frames, cfg.dataset.train_params.pred_frames
    trj_dim = cfg.dataset.cond_params.point_track_params.track_dim
    
    score = {"global_step": global_step,
             "metrics/fvd": 0, 
             "metrics/ssim": 0, 
             "metrics/psnr": 0, 
             "metrics/lpips": 0}
    
    for i_iter, batch in enumerate(val_loader):
        data = batch.to(accelerator.device)
        cond_frames = data[:,:, :cf]

        pred_v = []
        pred_v_m = []
        for auto_step in tqdm(range(NUM_AUTOREG), desc='sampling loop'):
            # total_frames = data[:,:,auto_step*pf:cf+(auto_step+1)*pf] # origin video
            gt_frames = data[:,:,cf+auto_step*pf:cf+(auto_step+1)*pf]
            total_frames = torch.cat([cond_frames, gt_frames], dim=2) #reproduced video 
            
            motion_cond = create_motion_cond(total_frames, motion_tracker, cfg.dataset.cond_params) #[b c cf+pf pn] or [b c cf+pf h w]
            motion_cond = motion_cond[:,:,:,::5].to(accelerator.device)
            motion_cond, _ = filtered_point_diff(motion_cond, cond_frames)

            cond_motion, gt_motion = motion_cond[:,:,:cf], motion_cond[:,:,cf:]

            cond_mo_xy, cond_mo_fea = cond_motion[:,:trj_dim], cond_motion[:,trj_dim:]

            cond_x = normalize_img(cond_frames) ### scale to [-1,1]
            gt_x = normalize_img(gt_frames)

            cond_mo = normalize_trj(cond_mo_xy, mo_mean, mo_std) #b c t pn
            cond_mo = expand_motion_info(cond_mo)
            
            pred_mo = track_predictor(cond_mo, cond_x)
            pred_mo_fea = intergrate_motion_feature(cond_mo_fea, cond_mo_xy[:,-1:], repeat_num = pf) 

            mo = torch.cat([cond_mo, pred_mo], dim=2)
            mo_fea = torch.cat([cond_mo_fea, pred_mo_fea], dim=2)
            mc = {"x": mo, "fea":mo_fea}

            # B,C,T,H,W 
            pred_frames = model.sample_video(cond_x, gt_x, mc)           
            cond_frames = torch.cat([cond_frames, pred_frames], dim=2)[:,:,-cf:]
            pred_v.append(pred_frames)
            
            pred_mo_xy = unnormalize_trj(pred_mo[:,:trj_dim], mo_mean, mo_std)
            pred_v_m.append(torch.cat([cond_mo_xy, pred_mo_xy], dim=2))
        
        res_pred_v = torch.cat(pred_v, dim=2)     # B,C,Tf(26),H,W
        res_pred_v = torch.cat([data[:,:, :cf], res_pred_v],dim=2)  # B,C,T(30),H,W

        res_pred_v_m = torch.stack(pred_v_m, dim=1)     # B,auto_num,C,T(6),PN

        origin_vids = rearrange(data, 'b c t h w -> b t c h w')
        result_vids = rearrange(res_pred_v, 'b c t h w -> b t c h w')

        fvd = calculate_fvd1(origin_vids, result_vids, torch.device("cuda"), mini_bs=16)
        videos1 = origin_vids[:, cf:]
        videos2 = result_vids[:, cf:]
        ssim = calculate_ssim1(videos1, videos2)[0]
        psnr = calculate_psnr1(videos1, videos2)[0]
        lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]

        score["metrics/fvd"] += fvd
        score["metrics/ssim"] += ssim
        score["metrics/psnr"] += psnr
        score["metrics/lpips"] += lpips

        visualize_path = os.path.join(cfg.checkpoint.output, f'vdm_steps_{global_step}','val_vis_result',f'batch_{i_iter}')
        visualize(
            save_path = visualize_path,
            pred = result_vids,
            gt = origin_vids,
            motion = res_pred_v_m
        )

        print(f"generate sample {(i_iter+1)*batch.size(0)}/{NUM_ITER*batch.size(0)} \n")

    
    score["metrics/fvd"] /= NUM_ITER
    score["metrics/ssim"] /= NUM_ITER
    score["metrics/psnr"] /= NUM_ITER
    score["metrics/lpips"] /= NUM_ITER

    print("Total frame performance")    
    print("[FVD    {:.5f}]".format(score["metrics/fvd"]))
    print("[SSIM   {:.5f}]".format(score["metrics/ssim"]))
    print("[LPIPS  {:.5f}]".format(score["metrics/psnr"]))
    print("[PSNR   {:.5f}]".format(score["metrics/lpips"]))

    return score


def main(cfg):
    os.makedirs(cfg.train_params.save_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint.output, exist_ok=True)
    
    train(cfg)
    pass
    


if __name__ == '__main__':        
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    args = parse_args()
    cfg = load_config(args.config)
    
    print(cfg)
    main(cfg)
