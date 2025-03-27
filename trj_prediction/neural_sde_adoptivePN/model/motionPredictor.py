import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from model.module.attention import AttentionModule, CrossAttentionModule
from model.module.motion_module import ConvGRUCell
from model.module.block import ConvBlock

from torchdiffeq import odeint_adjoint as odeint

def create_net(dim,
               n_layers=1,
               n_units=100, 
               nonlinear=nn.Tanh):
    layers = [nn.Linear(dim, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, dim))
    return nn.Sequential(*layers)

def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

#--------track predictor-------------------

class TrackMotionModel(nn.Module):
    def __init__(self,
                 cfg):
        super().__init__()
        self.tc = cfg.cond_num
        self.tp = cfg.pred_num

        self.dim = cfg.dim
        self.track_dim = cfg.track_dim
        self.vid_dim = cfg.vid_dim
        self.attn_num= cfg.attn_num
        
        self.ode_steps = self.tp+1
        self.odeint_rtol = cfg.odeint_rtol
        self.odeint_atol = cfg.odeint_atol
        self.ode_method = cfg.ode_method
        self.ode_layers = cfg.ode_layers
        self.ode_hidden_dim = cfg.ode_hidden_dim

        
        ## ---vid encoder---
        self.vid_enc_2d = ConvBlock(self.vid_dim, self.dim, conv_method="2d", groups=8, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.vid_enc_temp = ConvBlock(self.dim, self.dim, conv_method="temporal", groups=8, dropout_rate = 0.1)

        ## ---projector---
        self.enc_project = nn.Sequential(
            nn.Linear(self.track_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )
        self.dec_project = nn.Sequential(
            nn.Linear(self.dim, self.track_dim),
            nn.GELU(),
            nn.Linear(self.track_dim, self.track_dim)
        )
        
        ## --- Ecoder attention ---
        self.attn_block = nn.ModuleList([])
        for i in range(self.attn_num):
            self.attn_block.append(nn.ModuleList([
                AttentionModule(self.dim*self.tc, shape = "b tk (t c)"),
                AttentionModule(self.dim, shape = "(b t) tk c")
                ]))
        
        ## --- context_representation ---
        self.conv_gru_cell = ConvGRUCell(self.dim, self.dim)
        
        ## --- future_prediction ---        
        ode_func_net = create_net(self.dim, 
                                  n_layers=self.ode_layers, 
                                  n_units=self.ode_hidden_dim)
        self.ode_net = ODEFunc(ode_func_net=ode_func_net)
    
    def forward(self, mo, vid):
        """
        param mo: past trajectory, [b,c,tc,pn]
        param vide: past vid, [b,c,tc,hw]
        out pred_mo: future trajectory, [b,c,tp,pn]
        """
        B, C, Tc, PN = mo.shape

        mo = rearrange(mo, 'b c t pn -> b pn t c')

        # encoder
        mc = self.enc_project(mo)
        mc = rearrange(mc, 'b pn t c -> b c t pn')

        vc = self.vid_enc_2d(vid)
        vc = self.vid_enc_temp(vc)
        vc = rearrange(vc, 'b c t h w -> b c t (h w)')

        for idx, (attnS, attnC) in enumerate(self.attn_block):
            mc = attnS(mc)
            mc = attnC(mc, vc)

        # predictor
        mc = self.predictor(mc)
        pred_mo = self.dec_project(mc)
        
        pred_mo = rearrange(pred_mo, 'b pn t c -> b c t pn')

        return pred_mo
    
    def predictor(self, mc):
        B, C, T, PN = mc.shape

        # GRU encoding
        mc = rearrange(mc, 'b c t pn -> t b c pn')
        h = torch.zeros((B, C, PN), device = mc.device) # init h_0
        for i in range(T):
            h = self.conv_gru_cell(mc[i], h)

        # ODE prediction
        h = rearrange(h, 'b c pn -> pn b c')
        t = torch.linspace(0, self.tp, steps=self.ode_steps, device = mc.device)
        h_future = odeint(self.ode_net, h, t,
                          rtol=self.odeint_rtol,
                          atol=self.odeint_atol,
                          method=self.ode_method)
        h_future = rearrange(h_future, 'T PN B C -> B PN T C')

        return h_future[:,:,1:]
    
class ODEFunc(nn.Module):
    def __init__(self, ode_func_net):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()
        init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point
        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)    
