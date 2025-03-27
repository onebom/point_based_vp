from torch.utils import data
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
import numpy as np
from einops import rearrange

import glob
import os
import pickle

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

def normalize_trj(t, mean, std):
    mean = mean.view(2,1,1)
    std = std.view(2,1,1)

    normalized_t = (t[:, :2] - mean) / std
    t[:,:2]=normalized_t
    return t

def unnormalize_trj(t, mean, std):
    mean = mean.view(2,1,1)
    std = std.view(2,1,1)

    unnormalized_t = (t[:,:2] * std) + mean
    t[:,:2]=unnormalized_t
    return t

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
                 num_observed_frames_val, num_predict_frames_val, val_videos_num, wSeg=False                 
                 ):
        np.random.seed(0)
        self.wSeg = wSeg
        self.num_observed_frames_train = num_observed_frames_train
        self.num_predict_frames_train = num_predict_frames_train
        self.num_observed_frames_val = num_observed_frames_val
        self.num_predict_frames_val = num_predict_frames_val
        self.val_videos_num = val_videos_num
        
        self.clip_length_train = num_observed_frames_train + num_predict_frames_train
        self.clip_length_val = num_observed_frames_val + num_predict_frames_val

        self.cond_cfg=cond_cfg
        self.transform = transform
        self.color_mode = 'RGB'
        
        self.data_path = Path(data_dir).absolute()

        video_paths = self.__getVideosFolder__(os.path.join(self.data_path,"train"))
        self.video_data = self.__getTrainData__(video_paths)

        val_video_paths = self.__getVideosFolder__(os.path.join(self.data_path, "test"), self.val_videos_num)
        self.val_video_data = self.__getValData__(val_video_paths)
        
    def __call__(self):
        train_dataset = ClipTrainDataset(self.num_observed_frames_train, 
                                         self.num_predict_frames_train, 
                                         self.video_data, self.transform, self.color_mode, self.cond_cfg)
        val_dataset = ClipValDataset(self.num_observed_frames_val, 
                                    self.num_predict_frames_val, 
                                    self.val_video_data, self.transform, self.color_mode)
            
        return train_dataset, val_dataset

    def __getVideosFolder__(self, data_dir, num_videos=None):
        filenames_all = sorted(glob.glob(os.path.join(data_dir, '*', '*.png')))
        video_paths = np.array(filenames_all).reshape(-1, 30)

        if num_videos is not None:
            video_paths = video_paths[:num_videos]

        return video_paths
    
    def __getVideoSegFolder__(self, data_dir):
        filenames_all = sorted(glob.glob(os.path.join(data_dir, '**', '*_labelIds.png')))
        video_paths = np.array(filenames_all)
        return video_paths

    def __getTrainData__(self, video_paths):
        clips = []
        
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
        self.transform = transform["vid"]
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
            out["motion"] = conds
        
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
            cond_path1 = clip_img_paths[0].replace('/leftImg8bit_sequence/',f'/motion_condition2/{self.cond_type}/')
            cond_path = cond_path1.replace('.png', '/')
            
            q_idx = list(self.cond_cfg.point_track_params.guery_frame_idx)
            conds_path = [os.path.join(cond_path, os.listdir(cond_path)[i]) for i in range(len(q_idx))]
            
            conds= []
            for path in conds_path:
                conds.append(np.load(path)) #array: c t pn(64x64)
     
            conds = torch.tensor(np.concatenate(conds, axis=-1))
            
        return conds
    
    def get_data_mean_std(self):
        root_path = "/data/onebom/data/Cityscapes/leftImg8bit_sequence_trainvaltest/motion_condition2/point_track/train"
        
        with open(os.path.join(root_path,"mean_std.pkl"), "rb") as f:
            mean_std = pickle.load(f)
        
        return mean_std["mean"], mean_std["std"]    
                    
class ClipValDataset(data.Dataset):
    def __init__(self, num_observed_frames, num_predict_frames, video_data, transform, color_mode):
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.video_data = video_data
        self.transform = transform["vid"]
        self.color_mode = color_mode
        
    def __len__(self):
        return len(self.video_data['clips'])      
    
    def __getitem__(self, index):
        clip_img_paths = self.video_data['clips'][index]
        
        imgs = []
        for img_path in clip_img_paths:
            img_path = Path(img_path).absolute().as_posix()
            if self.color_mode == 'RGB':
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.open(img_path).convert('L')
            imgs.append(img)

        video = rearrange(self.transform(imgs), 't c h w -> c t h w')

        return video

class ClipValSegDataset(data.Dataset):
    def __init__(self, 
                 num_observed_frames, 
                 num_predict_frames, 
                 video_data, 
                 transform, 
                 color_mode
                 ):
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.video_data = video_data
        self.vid_transform = transform["vid"]
        self.seg_transform = transform["seg"]
        self.color_mode = color_mode

    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, index):
        clip_seg_path = self.video_data[index]
        clip_img_paths = self.get_matching_frame(clip_seg_path)

        imgs = []
        for img_path in clip_img_paths:
            img_path = Path(img_path).absolute().as_posix()
            if self.color_mode == 'RGB':
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.open(img_path).convert('L')
            imgs.append(img)
        video = rearrange(self.vid_transform(imgs), 't c h w -> c t h w')

        seg = self.seg_transform([Image.open(clip_seg_path)]) # h w
        seg = torch.tensor(np.array(seg))[0]

        out = {"data": video, "seg": seg}

        return out
    
    def get_matching_frame(self, seg_path):
        start_clip_path = seg_path.replace('/gtFine/','/leftImg8bit_sequence/').replace('gtFine_labelIds', 'leftImg8bit')
        start_num = int(start_clip_path.split("/")[-1].split("_")[2])
        # /data/onebom/data/Cityscapes/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
        clip_img_paths = []
        for i in range(self.num_observed_frames+self.num_predict_frames):
            clip_num = start_num+i
            clip_path = start_clip_path.replace(f'_{start_num:06d}_', f'_{clip_num:06d}_')
            clip_img_paths.append(clip_path)

        return clip_img_paths