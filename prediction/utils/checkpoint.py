import torch
import os.path as osp

from collections import defaultdict
from mmcv.runner import CheckpointLoader

from omegaconf import read_write
from utils.logger import get_logger


def load_checkpoint(config, model, optimizer, lr_scheduler):
    
    print(f'Loading checkpoint from {config.checkpoint.resume}')
    
    if osp.isfile(config.checkpoint.resume):
        checkpoint = CheckpointLoader.load_checkpoint(config.checkpoint.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        metrics = defaultdict(float)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            with read_write(config):
                config.step = checkpoint['step']
                config.epoch = checkpoint['epoch']
        print(f"Loaded successfully '{config.checkpoint.resume}' (epoch {checkpoint['epoch']})")
        # metrics = checkpoint['metrics']
    del checkpoint
    torch.cuda.empty_cache()
    return metrics

def save_checkpoint(config, epoch, step, model, optimizer, lr_scheduler, metrics=None):
    
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'metrics': metrics,
        'epoch': epoch,
        'step': step,
    }
    if metrics is not None:
        for k, v in metrics.items():
            save_state[k] = v
        
    filename = f'ckpt_epoch_{epoch}_step_{step}.pth'
    
    torch.save(save_state, osp.join(config.checkpoint.output, filename))
    torch.save(save_state, osp.join(config.checkpoint.output, 'checkpoint.pth'))
    
    print(f"Saved checkpoint to '{osp.join(config.checkpoint.output, filename)}' (epoch {epoch})")
    
def load_checkpoint_with_accelerator(config, accelerator, lr_meter, losses):
    print("Loading checkpoint from", config.checkpoint.resume)
    if osp.exists(config.checkpoint.resume):
        accelerator.load_state(config.checkpoint.resume)
    if osp.exists(config.checkpoint.resume + '.pt'):
        state_dict = torch.load(config.checkpoint.resume + '.pt')
        global_step = state_dict['global_step']
        save_epoch = state_dict['epoch']
        lr_meter.load(state_dict['lr_meter'])
        losses.load(state_dict['losses'])
        print(f"Loaded successfully '{config.checkpoint.resume}' (epoch {state_dict['epoch']}) (global step {state_dict['global_step']})")
    else:
        global_step = 0
        save_epoch = 0
        lr_meter.reset()
        losses.reset()
        print(f'Failed to load checkpoint from {config.checkpoint.resume}')    
    return global_step, save_epoch, lr_meter, losses

def save_checkpoint_with_accelerator(config, accelerator, global_step, epoch, lr_meter, losses):
    save_path = osp.join(config.checkpoint.output, f'vdm_steps_{global_step}')
    accelerator.save_state(save_path)
    
    save_path_file = save_path + ".pt"
    accelerator.save({
        'global_step': global_step,
        'epoch': epoch,
        'lr_meter': lr_meter.ckpt(),
        'losses': losses.ckpt(),
    }, save_path_file)
    print(f"Saved checkpoint to '{save_path}' (epoch {epoch}) (global step {global_step})")
    
def load_checkpoint_for_inference(config, accelerator):
    print("Loading checkpoint from", config.checkpoint.resume)
    if osp.exists(config.checkpoint.resume):
        accelerator.load_state(config.checkpoint.resume)
    else:
        ValueError("No checkpoint found.")