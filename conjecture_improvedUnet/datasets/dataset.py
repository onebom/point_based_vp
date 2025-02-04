from torch.utils import data
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms as T, utils
import torchvision.transforms.functional as F
from functools import partial
import random
import torchvision.transforms as transforms
import cv2
import numpy as np
from einops import rearrange

import glob
import os


CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


#### Datasets
class CityscapesDataset(object):
    def __init__(self, 
                 data_dir, cond_cfg, transform,
                 num_observed_frames_train, num_predict_frames_train,
                 num_observed_frames_val, num_predict_frames_val                 
                 ):
        np.random.seed(0)
        self.num_observed_frames_train = num_observed_frames_train
        self.num_predict_frames_train = num_predict_frames_train
        self.num_observed_frames_val = num_observed_frames_val
        self.num_predict_frames_val = num_predict_frames_val
        
        self.clip_length_train = num_observed_frames_train + num_predict_frames_train
        self.clip_length_val = num_observed_frames_val + num_predict_frames_val

        self.cond_cfg=cond_cfg
        self.transform = transform
        self.color_mode = 'RGB'
        
        self.data_path = Path(data_dir).absolute()

        video_paths = self.__getVideosFolder__(os.path.join(self.data_path,"train"))
        val_video_paths = self.__getVideosFolder__(os.path.join(self.data_path,"test"))
        
        self.video_data = self.__getTrainData__(video_paths)
        self.val_video_data = self.__getValData__(val_video_paths)
        
    def __call__(self):
        train_dataset = ClipTrainDataset(self.num_observed_frames_train, self.num_predict_frames_train, self.video_data, self.transform, self.color_mode, self.cond_cfg)
        val_dataset = ClipValDataset(self.num_observed_frames_val, self.num_predict_frames_val, self.val_video_data, self.transform, self.color_mode)
        return train_dataset, val_dataset

    def __getVideosFolder__(self, data_dir):
        filenames_all = sorted(glob.glob(os.path.join(data_dir, '*', '*.png')))
        video_paths = np.array(filenames_all).reshape(-1, 30)

        return video_paths
    
    def __getTrainData__(self, video_paths):
        clips = []
        temp_dists = []
        
        for v_idx in range(len(video_paths)):      
            for cond_timestep in range(0, len(video_paths[v_idx]) - self.clip_length_train + 1, self.num_observed_frames_train):
                pred_timestep = cond_timestep + self.num_observed_frames_train
                clip = video_paths[v_idx][cond_timestep:cond_timestep + self.num_observed_frames_train].tolist() + video_paths[v_idx][pred_timestep:pred_timestep + self.num_predict_frames_train].tolist()
                
                clips.append(clip)

        return {'clips':clips}
    
    def __getValData__(self, video_paths):
        clips = []
        for v_idx in range(len(video_paths)): 
            clip = video_paths[v_idx][:]
            clips.append(clip.tolist())

        return {'clips': clips}
    
        
class ClipTrainDataset(data.Dataset):
    """
    Video clips dataset
    """
    def __init__(self, num_observed_frames, num_predict_frames, video_data, transform, color_mode, cond_cfg):
        """
        Args:
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clips --- List of video clips frames file path
            transfrom --- torchvision transforms for the image
            color_mode --- 'RGB' for RGB dataset, 'grey_scale' for grey_scale dataset

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_observed_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_predict_frames, C, H, W)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.video_data = video_data
        self.transform = transform
        self.cond_cfg = cond_cfg
        self.cond_type = cond_cfg.cond_type
        self.cond_mode = cond_cfg.mode
        
        if color_mode != 'RGB' and color_mode != 'grey_scale':
            raise ValueError("Unsupported color mode!!")
        else:
            self.color_mode = color_mode

    def __len__(self):
        return len(self.video_data['clips'])
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_observed_frames, C, H, W)
            future_clip: Tensor with shape (num_predict_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_img_paths = self.video_data['clips'][index]
        
        imgs = []
        conds = []
        for img_path in clip_img_paths:
            
            img_path = Path(img_path)
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(img)
            
        if self.cond_type is not None and self.cond_mode=="use_preprocessd":
            conds = self.load_cond(clip_img_paths) #tensor: [c t pn] or [c t h w]

        original_clip = rearrange(self.transform(imgs), 't c h w -> c t h w')
        past_clip = original_clip[:, 0:self.num_observed_frames]
        future_clip = original_clip[:, -self.num_predict_frames:]
        
        out = {"cond":past_clip, "gt":future_clip}
        if self.cond_mode=="preprocess_ing":
            out["clip_paths"] = clip_img_paths
        elif self.cond_mode == "use_preprocessd":
            out["motion_cond"] = conds
        
        return out
    
    def load_cond(self, clip_img_paths): 
        if self.cond_type == "flow":
            conds = []
            for img_path in clip_img_paths:
                cond_path1 = img_path.replace('/leftImg8bit_sequence/',f'/motion_condition/{self.cond_type}/')
                cond_path = cond_path1.replace('.png', '.npy')
                cond = np.load(cond_path) #array: c,h,w
                conds.append(torch.tensor(cond))
            conds = torch.stack(conds) #tensor: t,c,h,w
            conds = conds.transpose(0,1) #tensor: c,t,h,w

        elif self.cond_type == "point_track": 
            cond_path1 = clip_img_paths[0].replace('/leftImg8bit_sequence/',f'/motion_condition/{self.cond_type}/')
            cond_path = cond_path1.replace('.png', '/')
            
            q_idx = list(self.cond_cfg.point_track_params.guery_frame_idx)
            conds_path = [os.path.join(cond_path, os.listdir(cond_path)[i]) for i in range(len(q_idx))]
            
            conds= []
            for path in conds_path:
                conds.append(np.load(path)) #array: c t pn(64x64)
     
            conds = torch.tensor(np.concatenate(conds, axis=-1))
            
        return conds
            

class ClipValDataset(data.Dataset):
    def __init__(self, num_observed_frames, num_predict_frames, video_data, transform, color_mode):
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.video_data = video_data
        self.transform = transform
        self.color_mode = color_mode
        
    def __len__(self):
        return len(self.video_data['clips'])        
    
    def __getitem__(self, index):
        clip_img_paths = self.video_data['clips'][index]
        
        imgs = []
        for img_path in clip_img_paths:
            
            img_path = Path(img_path)
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(img)
        
        video = rearrange(self.transform(imgs), 't c h w -> c t h w')
        return video 