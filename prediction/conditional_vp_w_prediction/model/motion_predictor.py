import torch
import torch.nn as nn

from einops import rearrange, repeat

from model.module.attention import CondAttentionLayer
from model.module.motion_module import MotionEncoder, ConvGRUCell

class TrackMotionModel(nn.Module):
    def __init__(self, cfg_unet, cfg_motion_predictor):
        super().__init__()
        self.tc = cfg_unet.model_params.cond_num
        self.tp = cfg_unet.model_params.pred_num
        
        self.dim = cfg_unet.model_params.dim
        self.pn_prime = cfg_unet.cond_params.pn_prime
        self.track_dim = cfg_unet.cond_params.track_dim
        
        self.cfg_motion_encoder = cfg_motion_predictor.MotionEncoder
        
        ## --- track_representation ---
        self.cond_attn = nn.ModuleList([]) 
        self.cond_attn.append(nn.ModuleList([
            CondAttentionLayer(tc_dim = self.tc*self.track_dim, pn_prime = self.pn_prime),
            CondAttentionLayer(tc_dim = self.tc*self.track_dim, pn_prime = self.pn_prime),
            CondAttentionLayer(tc_dim = self.tc*self.track_dim, pn_prime = self.pn_prime),
            CondAttentionLayer(tc_dim = self.tc*self.track_dim, pn_prime = self.pn_prime, last_attn=True)
        ]))
        
        tr_dim = self.track_dim * 2
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.track_dim, tr_dim),
            nn.GELU(),
            nn.Linear(tr_dim, tr_dim)
        )
        
        ## --- track_context_encoder ---
        self.cfg_motion_encoder.in_channels = tr_dim
        self.motion_encoder = MotionEncoder(self.cfg_motion_encoder)
        
        ## --- context_representation ---
        n_downs = self.cfg_motion_encoder.n_downs
        motion_dim = self.cfg_motion_encoder.model_channels*(2**n_downs)
        self.conv_gru_cell = ConvGRUCell(motion_dim, motion_dim, kernel_size = 3, stride = 1, padding = 1)
        
    
    def forward(self, track_motion):
        B, Tc, PN, C = track_motion.shape
        
        track_motion = self.track_representation(track_motion) # [B, Tc, pn_prime, C*2]
        track_motion = rearrange(track_motion, 'B T PN C -> B T C PN')
        
        motion_feature = self.track_context_encode(track_motion) # [B, Tc, model_ch*(2^n_down), pn_prime//(2^n_down)]
        motion_context = self.context_representation(motion_feature) # [B C PN]
        
        motion_pred = motion_context.unsqueeze(1).repeat(1, self.tp, 1, 1) # [B Tp C PN]
        # if sde:
        #     motion_pred = sde_prediction(motion_context)
            
        mc = torch.cat((motion_feature,motion_pred), dim=1) #[B, Tc+Tp, C, PN]
        return rearrange(mc, 'B T C PN -> B T PN C')
    
    def track_representation(self, track_motion):
        """
        in : track motion [B, Tc, PN, C]
        out: mc [B, Tc, pn_prime, C*2]
            - pn_prime : (default) 1024
        """
        
        for attn1, attn2, attn3, attn4 in self.cond_attn:
            mc = attn1(track_motion)
            mc = attn2(track_motion, mc)
            mc = attn3(track_motion, mc)
            mc = attn4(track_motion, mc)

        mc = self.cond_mlp(mc)
        return mc
    
    def track_context_encode(self, track_motion):
        """
        in : track motion [B, Tc, C, PN]
        out: mc [B, Tc, model_ch * (2 ** n_down), PN // (2 ** n_down)]
            - model_ch: (default) 8
            - n_down : (default) 3
            ... shape in default : B, Tc, 64, 128
        """
        
        B, Tc, C, PN = track_motion.shape
        x = track_motion.flatten(0, 1)
        
        mc = self.motion_encoder(x)
        mc = rearrange(mc, '(B T) C PN -> B T C PN', B=B, T=Tc)
        return mc
    
    def context_representation(self, motion_feature):
        """
        in : motion_feature [B, Tc, C, PN]
        out: m [B C PN]
        """
        B, T, C, PN = motion_feature.shape
        m = torch.zeros((C, PN), device = motion_feature.device)
        m = repeat(m, 'C PN -> B C PN', B=motion_feature.shape[0])
        
        for i in range(motion_feature.shape[1]):
            m = self.conv_gru_cell(motion_feature[:, i, ...], m)
            
        return m
        