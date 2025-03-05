import argparse
import timeit
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import math
from math import ceil    

from tqdm import tqdm
from datetime import timedelta
import wandb

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

from model.video_direct_diffusion import VideoDirectDiffusion

import torch.backends.cudnn as cudnn

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from accelerate import InitProcessGroupKwargs
from accelerate import DistributedDataParallelKwargs as DDPK

from torchinfo import summary
from contextlib import redirect_stdout

from metrics.calculate_ade_fde import calculate_ade, calculate_fde

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config/all_pointTrack_vp.yaml")
    args = parser.parse_args()
    return args

def train(cfg):
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
    ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=5400)
            )
    
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
    val_batch = []
    while len(val_batch)*cfg.dataset.valid_params.batch_size < cfg.dataset.valid_params.num_videos:
        val_batch.append(next(iter(test_loader)))
    
    # b, c, t, h, w = val_batch1.shape
    # model_summary=summary(model, 
                        #   input_size = (b, c, 6, 4096)),
                        #   depth=5)
    # with open(os.path.join(cfg.train_params.save_dir,"model_architecture.txt"), "w") as f:
    #     f.write(str(model_summary))
                
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
                # motion_cond is None when cond_type==None\
                motion_cond = batch["motion_cond"]
                motion_cond = motion_cond.to(accelerator.device)
                
                with accelerator.autocast():
                    # template loss 사용안함. (=0.0)
                    total_loss = model(motion_cond)

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 0.3)

                optimizer.step()    
                lr_scheduler.step()
                optimizer.zero_grad()
                
                losses.synchronize_and_update(accelerator, total_loss, global_step)
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
                        meters = valid(cfg, accelerator, model, val_batch, motion_cond_predictor, global_step)
                        logs = {'ADE': meters['metrics/ade'], 'FDE' : meters['metrics/fde']}
                        accelerator.log(logs, step=global_step)
                    model.train()
                   
            ### ::: 10. train logging         
            logs = {'loss': total_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0], 
                    "epoch": epoch}
            progress_bar.set_postfix(logs)
            accelerator.log(logs, step=global_step)
            if global_step >= final_step:
                break
        accelerator.wait_for_everyone()
    accelerator.end_training()
    
@torch.inference_mode()
def valid(cfg, accelerator, model, val_loader, motion_cond_predictor, global_step):
    cudnn.enabled=True
    cudnn.benchmark=True
    if cfg.train_params.seed is not None:
        set_seed(cfg.train_params.seed)
    
    model = accelerator.unwrap_model(model)
    model.eval()
    model.to(accelerator.device)
    
    NUM_AUTOREG = ceil(cfg.dataset.valid_params.pred_frames / cfg.dataset.train_params.pred_frames)
    cf, pf = cfg.dataset.train_params.cond_frames, cfg.dataset.train_params.pred_frames
    
    res_pred=[]
    res_gt=[]
    res_frames=[]
    for i_iter, batch in enumerate(val_loader):
        data = batch.to(accelerator.device)

        gt=[]
        pred=[]
        frames = []
        for auto_step in tqdm(range(NUM_AUTOREG), desc='sampling loop'):
            if cfg.dataset.cond_params.cond_type:
                total_frames = data[:,:,auto_step*pf:cf+(auto_step+1)*pf] # origin video
                motion_cond=create_motion_cond(total_frames, motion_cond_predictor, cfg.dataset.cond_params) #[b c cf+pf pn] or [b c cf+pf h w]
                motion_cond=motion_cond.to(accelerator.device)
                            
            pred_trj = model.sample_video(motion_cond) # B,C,Tf,PN

            pred.append(pred_trj)
            gt.append(motion_cond) # B,C,T,PN
            frames.append(total_frames) # B,C,T,H,W

        res_pred.append(torch.stack(pred, dim=1)) # B,auto,C,Tf,PN     
        res_gt.append(torch.stack(gt, dim=1)) # B,auto,C,T,PN 
        res_frames.append(torch.stack(frames, dim=1))

    res_pred = torch.cat(res_pred, dim=0)    
    res_gt = torch.cat(res_gt, dim=0)
    res_frames = torch.cat(res_frames, dim=0)
    print('generating Done:', res_pred.shape)

    # Evaluation
    res_pred = rearrange(res_pred, 'b n c t pn -> b n pn t c')
    res_gt = rearrange(res_gt, 'b n c t pn -> b n pn t c')
    res_frames = rearrange(res_frames, 'b n c t h w -> b n t c h w')

    label = res_gt[:,:,:,cf:]
    ade = calculate_ade(res_pred, label)
    fde = calculate_fde(res_pred, label)
    print("Total frame performance")    
    print("[ADE    {:.5f}]".format(ade))
    print("[FDE   {:.5f}]".format(fde))

    # Visulization
    visualize_path = os.path.join(cfg.checkpoint.output, f'vdm_steps_{global_step}','vis_result')
    visualize(
        save_path=visualize_path,
        pred=res_pred,
        gt=res_gt,
        frames = res_frames,
        cond_frame_num=cf
    )
    logs = {'vis': wandb.Image(os.path.join(visualize_path,"pic_row_0","autonum_0.png"),mode="RGB")}
    
    accelerator.log(logs, step=global_step)

    return {
        'global_step': global_step,
        'metrics/ade': ade,
        'metrics/fde': fde,
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
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # os.environ["NCCL_TIMEOUT"] = "3600"
    
    args = parse_args()
    cfg = load_config(args.config)
    
    print(cfg)
    main(cfg)
