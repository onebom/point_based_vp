from model.module.condition import MotionConditioning, MotionPredictor
import torch.nn as nn
import torch
from einops import rearrange, repeat

class MotionModel(nn.Module):
    def __init__(self, config, autoencoder=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.motion_predictor = MotionPredictor(**config.motion_predictor.model_params)
        
        
    def forward(self, cond_frames, gt_frames, temporal_distance, action, return_result=False):
        B, C, T, H, W = cond_frames.shape
        
        cond_of = []
        cond_occ = []
        gt_of = []
        gt_occ = []
        
        with torch.no_grad():
            for i in range(T-1):
                cond_generated = self.autoencoder.generate_sample(cond_frames[:, :, i:i+2])
                gt_generated = self.autoencoder.generate_sample(gt_frames[:, :, i:i+2])
                
                cond_of.append(cond_generated['optical_flow'].permute(0, 3, 1, 2))
                gt_of.append(gt_generated['optical_flow'].permute(0, 3, 1, 2))
                cond_occ.append(cond_generated['occlusion_map'])
                gt_occ.append(gt_generated['occlusion_map'])
        
        cond_of = torch.stack(cond_of, dim=2) # [B, 2, T-1, H, W]
        cond_occ = torch.stack(cond_occ, dim=2) # [B, 1, T-1, H, W]
        gt_of = torch.stack(gt_of, dim=2)
        gt_occ = torch.stack(gt_occ, dim=2)
        
        cond_motion = torch.cat([cond_of, cond_occ*2-1], dim=1) # [B, 3, T-1, H, W]
        gt_motion = torch.cat([gt_of, gt_occ*2-1], dim=1)
        
        pred_motion = self.motion_predictor(cond_motion, temporal_distance, action)
        
        pred_of = pred_motion[:, :2]
        pred_occ = (pred_motion[:, 2].unsqueeze(1) + 1) * 0.5
        
        pred_out_img_list = []
        pred_warped_img_list = []
        for i in range(T-1):
            if i == 0:
                pred_generated = self.autoencoder.generator.forward_with_flow(
                    source_image=gt_frames[:,:,i], optical_flow=pred_of[:,:,i].permute(0, 2, 3, 1), occlusion_map=pred_occ[:,:,i]
                )
            else:
                pred_generated = self.autoencoder.generator.forward_with_flow(
                    source_image=pred_out_img_list[-1], optical_flow=pred_of[:,:,i].permute(0, 2, 3, 1), occlusion_map=pred_occ[:,:,i]
                )
            pred_out_img_list.append(pred_generated['prediction'])
            pred_warped_img_list.append(pred_generated['deformed'])
        pred_out_vid = torch.stack(pred_out_img_list, dim=2)
        pred_warped_vid = torch.stack(pred_warped_img_list, dim=2)
        
        rec_loss = torch.nn.functional.l1_loss(pred_out_vid, gt_frames[:, :, 1:])
        rec_warp_loss = torch.nn.functional.l1_loss(pred_warped_vid, gt_frames[:, :, 1:])
        
        
        if return_result:
            return pred_out_vid, pred_warped_vid
        
        return rec_loss, rec_warp_loss