import argparse
import timeit
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
import math
from tqdm import tqdm

import torch
from utils.config import load_config
from utils.meter import RunningAverageMeter
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lrscheduler
from utils.checkpoint import load_checkpoint_with_accelerator, save_checkpoint_with_accelerator
from utils.motion_cond import create_motion_cond

from einops import rearrange

from datasets.builder import build_dataloader, build_dataset
from utils.visualize import visualize


from model.video_ncsn import VideoNCSN
from model.video_direct_diffusion import VideoDirectDiffusion
from datasets.video_dataset import DatasetRepeater
from datasets.dataset import normalize_img

import torch.backends.cudnn as cudnn

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from accelerate import DistributedDataParallelKwargs as DDPK

from torchinfo import summary
from contextlib import redirect_stdout

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config/wo_motion_VP_ver3_ncsn.yaml")
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
    
    ### ::: 1. setting Accelerator for multi-gpu 
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps,
        mixed_precision=cfg.train_params.mixed_precision,
        log_with= "wandb" if cfg.wandb.enable else None,
        project_dir=cfg.checkpoint.output,
        kwargs_handlers=[DDPK(find_unused_parameters=True)]
    )
    print(f"Assigned device: {accelerator.device}")
    print(f"Number of processes (GPUs) used by Accelerator: {accelerator.num_processes}")
   
    ### ::: 2. load model
    if cfg.model.sde.use_ncsn:
        model = VideoNCSN(cfg.model)
    else:
        model = VideoDirectDiffusion(cfg.model)
    model.to(accelerator.device)
    
    motion_cond_predictor=None
    if cfg.dataset.cond_params.cond_type == "point_track":
        motion_cond_predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        motion_cond_predictor.to(accelerator.device)
    elif cfg.dataset.cond_params.cond_type == "flow":
        from utils.raft import Raft
        motion_cond_predictor = Raft('./utils/raft/raft_model/raft_sintel_iter20_240x320.onnx')
    
    # Meter setting
    lr_meter = RunningAverageMeter()
    losses = RunningAverageMeter()
    
    ### ::: 3. load data
    train_dataset, test_dataset = build_dataset(cfg.dataset)
    train_loader, test_loader = build_dataloader(cfg.dataset, train_dataset, test_dataset)
    val_batch1 = next(iter(test_loader))
    
    b, c, t, h, w = val_batch1.shape
    model_summary=summary(model, 
                          input_size = ((b, c, cfg.dataset.train_params.cond_frames, h, w),
                                        (b, c, cfg.dataset.train_params.pred_frames, h, w),
                                        (b, c, 6, 4096)),
                          depth=5)
    with open(os.path.join(cfg.train_params.save_dir,"model_architecture.txt"), "w") as f:
        f.write(str(model_summary))
                
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
    for epoch in range(epoch_cnt, cfg.train_params.max_epochs):
        for i_iter, batch in enumerate(train_loader):
            iter_end = timeit.default_timer()
            with accelerator.accumulate(model):                
                cond, gt = batch["cond"], batch["gt"]
                cond, gt = cond.to(accelerator.device), gt.to(accelerator.device)
                
                # motion_cond is None when cond_type==None
                motion_cond = None
                if cfg.dataset.cond_params.cond_type is not None:
                    if cfg.dataset.cond_params.mode == "use_preprocessd":
                        motion_cond = batch["motion_cond"]
                        motion_cond = motion_cond.to(accelerator.device)
                    else:
                        origin_vids = torch.cat([cond, gt], dim=2)
                        motion_cond = create_motion_cond(origin_vids, motion_cond_predictor, cfg.dataset.cond_params) 
                        motion_cond = motion_cond.to(accelerator.device)

                cond = normalize_img(cond) ### scale to [-1,1]
                gt = normalize_img(gt)

                with accelerator.autocast():
                    # template loss 사용안함. (=0.0)
                    loss = model(cond, gt, motion_cond)
                    
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
                        meters = valid(cfg, accelerator, model, [val_batch1], motion_cond_predictor, global_step)
                        logs = {'FVD': meters['metrics/fvd'], 'SSIM' : meters['metrics/ssim'], 'PSNR' : meters['metrics/psnr'], 
                                'LPIPS' : meters['metrics/lpips'], 'NFE': meters['metrics/nfev']}
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
def valid(cfg, accelerator, model, val_loader, motion_cond_predictor, global_step, vis=False):
    cudnn.enabled=True
    cudnn.benchmark=True
    if cfg.train_params.seed is not None:
        set_seed(cfg.train_params.seed)
    
    model = accelerator.unwrap_model(model)
    model.eval()
    model.to(accelerator.device)
    
    origin_vids = []
    result_vids = []
    result_motion_cond = []
    
    total_pred_frames = cfg.dataset.valid_params.pred_frames
    
    from math import ceil    
    NUM_AUTOREG = ceil(cfg.dataset.valid_params.pred_frames / cfg.dataset.train_params.pred_frames)
    
    cf, pf = cfg.dataset.train_params.cond_frames, cfg.dataset.train_params.pred_frames
    # 첫번째 배치만 뽑아서 확인할 예정
    for i_iter, batch in enumerate(val_loader):
        # if i_iter != 0: continue
        if i_iter != 0 : break
        
        data = batch.to(accelerator.device)
        bs = data.size(0)
        cond_frames = data[:,:, :cf]
        
        result_frames = []
        result_motions = []
        
        for auto_step in tqdm(range(NUM_AUTOREG), desc='sampling loop'):
            origin_frames = data[:,:,cf+auto_step*pf:cf+(auto_step+1)*pf]
            
            motion_cond=None
            # validation은 무조건 inference해서 뽑기
            if cfg.dataset.cond_params.cond_type:
                total_frames = data[:,:,auto_step*pf:cf+(auto_step+1)*pf] # origin video
                # total_frames = torch.cat([cond_frames, origin_frames], dim=2) #reproduced video 
                
                motion_cond=create_motion_cond(total_frames, motion_cond_predictor, cfg.dataset.cond_params) #[b c cf+pf pn] or [b c cf+pf h w]
                motion_cond=motion_cond.to(accelerator.device)
                
                result_motions.append(motion_cond.unsqueeze(1)) #[b 1 c cf+pf pn] or [b 1 c cf+pf h w]
                motion_cond = motion_cond/cfg.dataset.dataset_params.frame_shape
            
            # B,C,T,H,W 
            pred_frames, nfev = model.sample_video(cond_frames = normalize_img(cond_frames), 
                                            gt_frames = normalize_img(origin_frames), 
                                            motion_cond = motion_cond) 
            
            pred_frames = torch.stack(list(pred_frames), dim=0)
            result_frames.append(pred_frames)
            
            cond_frames = torch.cat([cond_frames, pred_frames], dim=2)
            cond_frames = cond_frames[:,:,-cf:]
        
        result_frames = torch.cat(result_frames, dim=2)
        
        result_vids.append(torch.cat([data[:,:,:cf],result_frames],dim=2))
        origin_vids.append(data[:,:,:cf+total_pred_frames])
        
        if result_motions:
            result_motions = torch.cat(result_motions, dim=1) #[b NumAuto c cf+pf pn] or [b NumAuto c cf+pf h w]
            result_motion_cond.append(result_motions)
        
        print(f'result_video generated: {result_vids[-1].shape}')
    
    print('generating Done')
    origin_vids = torch.cat(origin_vids, dim=0) # [B, C, T, H, W]
    result_vids = torch.cat(result_vids, dim=0) # [B, C, T, H, W]

    origin_vids = rearrange(origin_vids, 'b c t h w -> b t c h w')
    result_vids = rearrange(result_vids, 'b c t h w -> b t c h w')
    
    if result_motion_cond:
        result_motion_cond = torch.cat(result_motion_cond, dim=0) # [B, NUM_AUTOREG, C, cf+pf, PN] or [B, NUM_AUTOREG, C, cf+pf, H, W]
    
    # # performance metrics
    
    from metrics.calculate_fvd import calculate_fvd,calculate_fvd1
    from metrics.calculate_psnr import calculate_psnr,calculate_psnr1
    from metrics.calculate_ssim import calculate_ssim,calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips,calculate_lpips1
    
    fvd = calculate_fvd1(origin_vids, result_vids, torch.device("cuda"), mini_bs=16)
    videos1 = origin_vids[:, cf:]
    videos2 = result_vids[:, cf:]
    ssim = calculate_ssim1(videos1, videos2)[0]
    psnr = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
    print("Total frame performance")    
    print("[FVD    {:.5f}]".format(fvd))
    print("[SSIM   {:.5f}]".format(ssim))
    print("[LPIPS  {:.5f}]".format(lpips))
    print("[PSNR   {:.5f}]".format(psnr))

    for i in range(NUM_AUTOREG):
        videos1 = origin_vids[:, cf+pf*i:cf+pf*(i+1)]
        videos2 = result_vids[:, cf+pf*i:cf+pf*(i+1)]
        
        local_ssim = calculate_ssim1(videos1, videos2)[0]
        local_psnr = calculate_psnr1(videos1, videos2)[0]
        local_lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
        print(f"{cf+pf*i} ~ {cf+pf*(i+1)-1}th frame prediction performance")
        print("[SSIM   {:.5f}]".format(local_ssim))
        print("[LPIPS  {:.5f}]".format(local_lpips))
        print("[PSNR   {:.5f}]".format(local_psnr))

    # result visulization 
    visualize_path = os.path.join(cfg.checkpoint.output, f'vdm_steps_{global_step}','vis_result')
    visualize(
        save_path=visualize_path,
        origin=origin_vids,
        result=result_vids,
        motion_cond = result_motion_cond,
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
    import wandb
    if cfg.dataset.cond_params.cond_type=="point_track":
        logs = {'pred_gif': wandb.Video(os.path.join(visualize_path,"all","gif","0","gif_result_0.gif"), fps=60, format="gif"), 
                'gt_gif': wandb.Video(os.path.join(visualize_path,"all","gif","0","gif_origin_0.gif"), fps=60, format="gif"), 
                'pic_row_w_motion': wandb.Image(os.path.join(visualize_path,"all","pic_row_w_cond","pic_row_0","autonum_0.png"),mode="RGB")}
    elif cfg.dataset.cond_params.cond_type=="flow":
        logs = {'pred_gif': wandb.Video(os.path.join(visualize_path,"all","gif","0","gif_result_0.gif"), fps=60, format="gif"), 
                'gt_gif': wandb.Video(os.path.join(visualize_path,"all","gif","0","gif_origin_0.gif"), fps=60, format="gif"), 
                'pic_row_w_motion': wandb.Image(os.path.join(visualize_path,"all","pic_row_w_cond","pic_row_0.png"),mode="RGB")}
    else:
        logs = {'pred_gif': wandb.Video(os.path.join(visualize_path,"all","gif","0","gif_result_0.gif"), fps=60, format="gif"), 
                'gt_gif': wandb.Video(os.path.join(visualize_path,"all","gif","0","gif_origin_0.gif"), fps=60, format="gif")}
    accelerator.log(logs, step=global_step)

    return {
        'global_step': global_step,
        'metrics/fvd': fvd,
        'metrics/ssim': ssim,
        'metrics/psnr': psnr,
        'metrics/lpips': lpips,
        'metrics/nfev': nfev
    }

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
