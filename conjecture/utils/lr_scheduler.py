from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import MultiStepLR

from diffusers.optimization import get_scheduler

def build_lrscheduler(config, optimizer, final_step):

    lr_scheduler = None
    if config.lr_scheduler.type == 'MultiStepLR':
        lr_scheduler = MultiStepLR(
            optimizer,
            **config.lr_scheduler.params
        )
    elif config.lr_scheduler.type == 'cosine':
        lr_scheduler = get_scheduler(
            config.lr_scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=config.lr_scheduler.params.lr_warmup_steps,
            num_training_steps=final_step,
        )
    else:
        raise NotImplementedError(f'lr scheduler {config.lr_scheduler.name} not implemented')

    return lr_scheduler