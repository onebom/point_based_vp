import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from model.module.attention import CondAttentionLayer,MoitonInteractMoudle
from model.module.motion_module import MotionEncoder, ConvGRUCell

from torchsde import sdeint_adjoint as sdeint 

class TrackPredictor(nn.Module):
    def __init__(
        self,
        motion_dim = 128,
        trj_dim_exp = 8,
        trj_f_dim = 128,
        interaction_num = 3
        ):
        super().__init__()
        # self.tc, self.tp = tc, tp
        self.motion_dim = motion_dim
        self.trj_dim_exp = trj_dim_exp
        self.trj_f_dim = trj_f_dim
        self.interaction_num = interaction_num
        
        self.trj_embed = SingleInputEmbedding(in_channels=self.exp_trj_dim, 
                                              out_channel=self.motion_dim)
        self.aggregator = GlobalInteractor(self.motion_dim, self.trj_f_dim, self.interaction_num)
        
        self.encoder = None
        self.decoder = None
        
    def forward(self, trj, trj_f=None):
        b, c, t, pn = trj.shape 
        assert c==self.exp_trj_dim
        
        # 0. trj embedding
        trj_emb = self.trj_embed(trj)
        
        # 1. [TODO] grouping
        
        # 2. track encoder(SDE-GRU)
        ## 2-1. past track feature encoding
        # tp_drift_loc, all_drift_loc, all_diff_z= self.encoder(past_track_loc)
        ## 2-2. past track feature encoding loss
        
        # 3. track feature aggregation
        track_emb = self.aggregator(trj_emb, feature_emb=trj_f)
        
        # 4. track decoder(SDE)
        ## 4-1. future track SDE predict
        ## 4-2. future track feature loss
        # out = self.decoder(data=data, local_embed=local_embed, global_embed=global_embed)
        # out['diff_in'], out['diff_out'], out['label_in'], out['label_out'] = diffusions_in, diffusionts_out, in_labels, out_labels
        
        return track_emb
    
class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel
                 ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1)),
            nn.LayerNorm([out_channel, 1, 1]), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 1)),
            nn.LayerNorm([out_channel, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 1)),
            nn.LayerNorm([out_channel, 1, 1])
        )

    def forward(self, x):
        return self.embed(x)

class GlobalInteractor(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        interation_num
        ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.interation_num = interation_num
        
        self.global_interactor_attn = nn.ModuleList(
            [MoitonInteractMoudle(self.dim,self.cond_dim) for _ in range(self.interation_num)])
        
        self.multi_head_porj = nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1))
        self.norm = nn.LayerNorm([self.dim,1,1])

    def forward(self, trj, trj_f):
        b, c, t, pn = trj.shape 
        
        x = trj 
        for layer in self.global_interactor_attn:
            x = layer(x, trj_f)
    
        x = self.multihead_proj(self.norm(x))
        
        return x

class LocalEncoderSDESepPara2(nn.Module):
    def __init__(self):
        super().__init__()
        # hyper param
        self.point_num = point_num
        self.track_dim = track_dim
        self.step_num = step_num
        self.minimum_step = None
        self.rtol = None
        self.atol = None
        self.method = None
        
        self.hidden = nn.Parameter(torch.Tensor(self.track_dim))
        
        self.lsde_func = None
        
    def forward(self, track_loc):
        B, C, Tp, PN = track_loc.shape
        
        past_time_steps = torch.linspace(-(Tp-1), 0 ,self.step_num)
        past_time_steps = -1*past_time_steps
        prev_t, t_i = past_time_steps[0] - 0.01, past_time_steps[0]
        
        prev_hidden = self.hidden[None,:,None].repeat(B,1,PN)
        
        drift_loc=[]
        diff_z=[]
        for t in range(0, self.step_num):
            ## 2-1. past track SDE predict
            first_point = prev_hidden
            time_steps_to_predict = torch.tensor([prev_t, t_i])
            
            pred_x, diff_noise = sdeint_dual(self.lsde_func, first_point, time_steps_to_predict,
                                             dt=self.minimum_step, 
                                             rtol = self.rtol, 
                                             atol = self.atol, 
                                             method = self.method) # shape: 2(prev_t, t_i),B,C,PN
            ode_sol = pred_x[-1] 
            # if torch.mean(pred_y[0, :, :] - prev_hidden) >= 0.001:
            #     print("Error: first point of the ODE is not equal to initial value")
            #     print(torch.mean(ode_sol[:, :, 0] - prev_hidden))
            #     exit()
            xt = aa_out[t] #B,C,PN
        
            ## 2-2. past track GRU update
            yt = ode_sol
            yt = self.gru_unit(xt, ode_sol) #B,C,PN
            
            # return to iteration
            prev_hidden = yt
            if t+1 < past_time_steps.size(-1):
                prev_t, t_i = past_time_steps[t], past_time_steps[t+1]
            
            drift_loc.append(yt)
            diff_z.append(diff_noise)
        
        all_drift_loc = torch.stack(drift_loc, dim=1)
        all_diff_z = torch.stack(diff_z, dim=1)
        
        tp_idx = range(0, self.step_num, (self.step_num-1)//(Tp-1))
        drift_loc = all_drift_loc[:,tp_idx,:,:]
        
        return drift_loc, all_drift_loc, all_diff_z

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