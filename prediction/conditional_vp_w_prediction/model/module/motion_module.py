import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionEncoder(nn.Module):
    def __init__(self, motion_encoder_cfg):
        super().__init__()

        input_dim=motion_encoder_cfg.in_channels
        ch=motion_encoder_cfg.model_channels
        n_downs=motion_encoder_cfg.n_downs
    
        model = []
        model += [nn.Conv1d(input_dim, ch, 5, padding = 2)]
        model += [nn.ReLU()]
        
        
        for _ in range(n_downs - 1):
            model += [nn.MaxPool1d(2)]
            model += [nn.Conv1d(ch, ch * 2, 5, padding = 2)]
            model += [nn.ReLU()]
            ch *= 2
        
        model += [nn.MaxPool1d(2)]
        model += [nn.Conv1d(ch, ch * 2, 7, padding = 3)]
        model += [nn.ReLU()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        x: (N, C, H, W)
        out: (N, C*(2^n_downs), H//(2^n_downs), W//(2^n_downs))
        """
        out = self.model(x)
        return out
    
    
class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.GateConv = nn.Conv1d(in_channels+hidden_channels, 2*hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.NewStateConv = nn.Conv1d(in_channels+hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, inputs, prev_h):
        """
        in: 
            - input : [B, in_channels, PN]
            - prev_h : [B, in_channels, PN]
        out : [B, hidden_channels, PN]
        """
        gates = self.GateConv(torch.cat((inputs, prev_h), dim = 1))
        u, r = torch.split(gates, self.hidden_channels, dim = 1)
        u, r = F.sigmoid(u), F.sigmoid(r)
        h_tilde = F.tanh(self.NewStateConv(torch.cat((inputs, r*prev_h), dim = 1)))
        new_h = (1 - u)*prev_h + h_tilde

        return new_h