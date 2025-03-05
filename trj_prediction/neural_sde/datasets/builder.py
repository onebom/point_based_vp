from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.transform import *

def build_dataset(config):
    transform = transforms.Compose([
        VidCCropResize((config.dataset_params.frame_shape, config.dataset_params.frame_shape)),
        VidToTensor(), 
        ])
    
    from datasets.dataset import CityscapesDataset
    train_dataset, val_dataset = CityscapesDataset(
        data_dir = config.dataset_params.data_dir,
        cond_cfg = config.cond_params,
        transform = transform,
        num_observed_frames_train = config.train_params.cond_frames,
        num_predict_frames_train = config.train_params.pred_frames,
        num_observed_frames_val = config.valid_params.cond_frames,
        num_predict_frames_val = config.valid_params.pred_frames,
        )()
    
    return train_dataset, val_dataset


def build_dataloader(config, train_dataset, val_dataset):
    
    if config.cond_params.mode == "preprocess_ing":
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_params.batch_size,
            shuffle=False,
            num_workers=config.train_params.dataloader_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_params.batch_size,
            shuffle=True,
            num_workers=config.train_params.dataloader_workers,
            pin_memory=True,
            drop_last=False,
        )
        
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.valid_params.batch_size,
        shuffle=False,
        num_workers=config.train_params.dataloader_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_dataloader, val_dataloader 