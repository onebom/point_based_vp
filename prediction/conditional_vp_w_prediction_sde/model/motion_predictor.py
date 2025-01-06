import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from model.module.attention import CondAttentionLayer
from model.module.motion_module import MotionEncoder, ConvGRUCell

from torchsde import sdeint_adjoint as sdeint 

class TrackMotionModel(nn.Module):
    def __init__(self, cfg_unet, cfg_motion_predictor):
        super().__init__()
        self.tc = cfg_unet.model_params.cond_num
        self.tp = cfg_unet.model_params.pred_num
        
        self.dim = cfg_unet.model_params.dim
        self.pn_prime = cfg_unet.cond_params.pn_prime
        self.track_dim = cfg_unet.cond_params.track_dim
        
        self.cfg_motion_encoder = cfg_motion_predictor.MotionEncoder
        self.sde_cfg = cfg_motion_predictor.sde
        
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
        selected_pn = self.pn_prime // (2 ** n_downs)
        self.conv_gru_cell = ConvGRUCell(motion_dim, motion_dim, kernel_size = 3, stride = 1, padding = 1)
        
        ## --- future_prediction ---
        self.in_channel = motion_dim
        self.hidden_channels = motion_dim
        self.motion_feature_size = (motion_dim, selected_pn)
        self.sde_unet = SDEUnet(self.sde_cfg, self.in_channel, self.hidden_channels,
                                self.motion_feature_size)
        
    
    def forward(self, track_motion, frame_idx):
        B, C, Tc, PN = track_motion.shape
                
        track_motion = self.track_representation(track_motion) # [B, Tc, pn_prime, C*2]
        track_motion = rearrange(track_motion, 'B T PN C -> B T C PN')
        
        motion_feature = self.track_context_encode(track_motion) # [B, Tc, model_ch*(2^n_down), pn_prime//(2^n_down)]
        motion_context = self.context_representation(motion_feature) # [B C PN]
        
        if self.sde_cfg.use_sde:
            motion_pred = self.future_predict(motion_context, frame_idx[:,-(self.tp+1):]) # [B Tp C PN]
        else:
            motion_pred = motion_context.unsqueeze(1).repeat(1, self.tp, 1, 1) # [B Tp C PN]
            
        mc = torch.cat((motion_feature,motion_pred), dim=1) #[B, Tc+Tp, C, PN]
        return rearrange(mc, 'B T C PN -> B T PN C')
    
    def track_representation(self, track_motion):
        """
        in : track motion [B, C, Tc, PN]
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
    
    def future_predict(self, motion_context, idx_p):
        B, C, PN = motion_context.shape
        B, Tp_p1 = idx_p.shape

        m_future = sdeint(self.sde_unet, motion_context.flatten(1), idx_p[0], 
                            method = self.sde_cfg.method, dt=self.sde_cfg.sde_options.dt,
                            rtol = self.sde_cfg.sde_options.rtol, atol = self.sde_cfg.sde_options.atol,
                            adaptive = self.sde_cfg.sde_options.adaptive) #(Tp_p1, B, C*PN)
        m_future = rearrange(m_future, 'T B (C PN) -> B T C PN', C = C)
        
        return m_future[:,1:, ...]
    
    
class SDEUnet(nn.Module):
    def __init__(self, sde_cfg, in_channels, hidden_channels,motion_feature_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.n_layers = sde_cfg.unet.n_layers
        self.nonlienar = sde_cfg.unet.nonlinear
        
        self.sde_unet_f = OdeSdeFuncNet(self.in_channels, self.hidden_channels, self.out_channels, self.n_layers, self.nonlienar)
        self.sde_unet_g = OdeSdeFuncNet(self.in_channels, self.hidden_channels, self.out_channels, self.n_layers, self.nonlienar)
        
        self.noise_type = sde_cfg.sde_options.noise_type
        self.sde_type = sde_cfg.sde_options.sde_type
        
        self.motion_feature_size = motion_feature_size
        
    def forward(self, t, x):
        return self.sde_unet_f(x)
    
    def f(self, t, x):
        """sde drift"""
        C, PN = self.motion_feature_size
        x = rearrange(x,"B (C PN) -> B C PN", C=C)
        x = self.sde_unet_f(x)
        
        return x.flatten(1)
    
    def g(self, t, x):
        """sde diffusion"""
        C, PN = self.motion_feature_size
        x = rearrange(x,"B (C PN) -> B C PN", C=C)
        x = self.sde_unet_g(x)
        x = F.tanh(x)
        
        return x.flatten(1)

class OdeSdeFuncNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, nonlinear = 'Tanh'):
        super().__init__()
        if nonlinear == 'tanh':
            nonlinear_layer = nn.Tanh()
        
        layers = []
        layers.append(nn.Conv1d(in_channels, hidden_channels, 3, 1, 1))
        for i in range(n_layers):
            layers.append(nonlinear_layer)
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, 3, 1, 1))
        layers.append(nonlinear_layer)
        layers.append(zero_module(nn.Conv1d(hidden_channels, out_channels, 3, 1, 1)))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module