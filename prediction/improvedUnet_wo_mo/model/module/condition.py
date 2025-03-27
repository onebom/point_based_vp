import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import trunc_normal_

from model.util import exists, EinopsToAndFrom, Residual, PreNorm, temporal_distance_to_frame_idx
from model.module.attention import STAttentionBlock, RelativePositionBias, CrossFrameAttentionLayer
from model.module.block import Mlp, ResnetBlock
from model.module.normalization import Normalization

def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000.):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.GateConv = nn.Conv2d(in_channels+hidden_channels, 2*hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.NewStateConv = nn.Conv2d(in_channels+hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, inputs, prev_h):
        """
        inputs: (N, in_channels, H, W)
        Return:
            new_h: (N, hidden_channels, H, W)
        """
        gates = self.GateConv(torch.cat((inputs, prev_h), dim = 1))
        u, r = torch.split(gates, self.hidden_channels, dim = 1)
        u, r = F.sigmoid(u), F.sigmoid(r)
        h_tilde = F.tanh(self.NewStateConv(torch.cat((inputs, r*prev_h), dim = 1)))
        new_h = (1 - u)*prev_h + h_tilde

        return new_h


class MotionEncoder(nn.Module):
    """
    Modified from
    https://github.com/sunxm2357/mcnet_pytorch/blob/master/models/networks.py
    """
    def __init__(self, motion_encoder_cfg):
        super().__init__()

        input_dim=motion_encoder_cfg.in_channels
        ch=motion_encoder_cfg.model_channels
        n_downs=motion_encoder_cfg.n_downs
    
        model = []
        model += [nn.Conv2d(input_dim, ch, 5, padding = 2)]
        model += [nn.ReLU()]
        
        
        for _ in range(n_downs - 1):
            model += [nn.MaxPool2d(2)]
            model += [nn.Conv2d(ch, ch * 2, 5, padding = 2)]
            model += [nn.ReLU()]
            ch *= 2
        
        model += [nn.MaxPool2d(2)]
        model += [nn.Conv2d(ch, ch * 2, 7, padding = 3)]
        model += [nn.ReLU()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        out: (B, C*(2^n_downs), H//(2^n_downs), W//(2^n_downs))
        """
        #TO DO: condition on diffferent diffusion timesteps
        out = self.model(x)
        return out

class MotionConditioning(nn.Module):
    def __init__(self, motion_encoder_cfg):
        super().__init__()
        
        self.motion_encoder_cfg = motion_encoder_cfg
        
        n_downs = self.motion_encoder_cfg.n_downs
        image_size = self.motion_encoder_cfg.image_size
        H = W = int(image_size/(2**n_downs))
        motion_C = self.motion_encoder_cfg.model_channels*(2**n_downs)
        self.motion_feature_size = (motion_C, H, W)
        
        self.motion_encoder = MotionEncoder(self.motion_encoder_cfg)
        self.conv_gru_cell = ConvGRUCell(motion_C, motion_C, kernel_size = 3, stride = 1, padding = 1)
        pass
    
    def context_encode(self, x):
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'b c t h w -> b t c h w')
        
        diff_images = x[:, 1:, ...] - x[:, 0:-1, ...] #(B, T-1, C, H, W)
        h = self.condition_enc(diff_images) #(B, T-1, C, H, W)

        m = torch.zeros(self.motion_feature_size, device = h.device)
        m = repeat(m, 'C H W -> B C H W', B=B)
        #update m given the first observed frame conditional feature
        m = [self.conv_gru_cell(h[:, 0, ...], m)]

        #recurrently calculate the context motion feature by GRU
        To = h.shape[1]
        
        for i in range(1, To):
            m.append(self.conv_gru_cell(h[:, i, ...], m[-1]))
            
        m = torch.stack(m, dim=2)
        
        return m
    
    def global_context_encode(self, x):
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'b c t h w -> b t c h w')
        
        diff_images = x[:, 1:, ...] - x[:, 0:-1, ...] #(B, T-1, C, H, W)
        h = self.condition_enc(diff_images) #(B, T-1, C, H, W)

        m = torch.zeros(self.motion_feature_size, device = h.device)
        m = repeat(m, 'C H W -> B C H W', B=B)
        #update m given the first observed frame conditional feature
        m = self.conv_gru_cell(h[:, 0, ...], m)

        #recurrently calculate the context motion feature by GRU
        To = h.shape[1]
        for i in range(1, To):
            m = self.conv_gru_cell(h[:, i, ...], m)

        return m
    
    def condition_enc(self, x):
        B, T, _, _, _ = x.shape
        x = x.flatten(0, 1)
        x = self.motion_encoder(x)
        
        return rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)
    
class LFPredictor(nn.Module):
    def __init__(
        self,
        dim,
        action_class_dim=None,
        heads=8,
        dim_head=32,
        depth_e=6,
        depth_d=2,
        mlp_ratio=4.0,
        motion_shape = (256, 3, 32, 32),
        num_action_classes=6,
        max_frame_idx=28
    ):
        super().__init__()
        self.num_action_calsses = num_action_classes
        self.max_frame_idx = max_frame_idx
        
        self.query_emb = nn.Parameter(torch.zeros(motion_shape))
        trunc_normal_(self.query_emb, std=.02)
        
        self.norm_q = Normalization(dim, norm_type='layer')
        
        self.conv_in = nn.Conv3d(3, dim, (1, 3, 3), padding=(0, 1, 1))
        
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        
        for _ in range(depth_d):
            self.encoder.append(STAttentionBlock(dim, heads=heads, dim_head=dim_head))
        
        for _ in range(depth_d):
            self.decoder.append(nn.ModuleList([
                STAttentionBlock(dim, heads=heads, dim_head=dim_head),
                STAttentionBlock(dim, heads=heads, dim_head=dim_head, is_cross=True),
            ]))
        
        self.fusion = Mlp(in_features=dim+2, hidden_features=int(dim*mlp_ratio), out_features=dim)
        
        self.conv_out = nn.Sequential(
            ResnetBlock(dim, 64),
            nn.Conv3d(64, 3, 1)
        )
            
        
    def forward(self, cond_feat, temporal_distance=None, action_class=None):
        b, c, t, h, w = cond_feat.shape
        
        query_emb = repeat(self.query_emb, 'c t h w -> b c t h w', b=b)
        
        cond_feat = self.conv_in(cond_feat) # [B, 256, T-1, H, W]
        
        x = torch.cat([cond_feat, query_emb], dim=2)
        B, C, T, H, W = x.shape
        
        frame_idx = temporal_distance_to_frame_idx(temporal_distance, t, device=cond_feat.device) # [B, T]
        
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        frame_emb = repeat(frame_idx, 'b t -> (b t) (h w) 1', t=T, h=H, w=W)
        action_emb = repeat(action_class, 'b -> (b t) (h w) 1', t=T, h=H, w=W) 
        
        x = torch.cat([x, frame_emb, action_emb], dim=2) # [BT, HW, C+2]
        x = self.fusion(x) # [BT, HW, C]
        x = rearrange(x, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)
        
        cond_feat, query_emb = x[:, :, :t], x[:, :, t:]
        
        for self_attn in self.encoder:
            cond_feat = self_attn(cond_feat)
        
        for self_attn, cross_attn in self.decoder:
            query_emb = self_attn(query_emb)
            query_emb = cross_attn(query_emb, context=cond_feat)
        
        query_emb = self.conv_out(query_emb)
        return query_emb

class MotionVAE(nn.Module):
    def __init__(
        self,
        dim,
        in_channel=3,
        heads=8,
        dim_head=32,
        depth_e=6,
        depth_d=2,
        mlp_ratio=4.0,
        num_action_classes=6,
        max_frame_idx=28
    ):
        super().__init__()
        self.num_action_calsses = num_action_classes
        self.max_frame_idx = max_frame_idx
        
        self.cond_proj = nn.Sequential(
            nn.Conv3d(in_channel, dim, (1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
        )
        self.pred_proj = nn.Sequential(
            nn.Conv3d(in_channel, dim, (1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
        )
        
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for _ in range(depth_e):
            self.encoder.append(CrossFrameAttentionLayer(dim, heads=heads, dim_head=dim_head, use_attn=True))
        for _ in range(depth_d):
            self.decoder.append(CrossFrameAttentionLayer(dim, heads=heads, dim_head=dim_head, use_attn=True))
        
        self.fc_mu = nn.Linear(dim, dim)
        self.fc_var = nn.Linear(dim, dim)
        
        self.conv_out = nn.Sequential(
            nn.ConvTranspose3d(dim, 64, (1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.GELU(),
            nn.ConvTranspose3d(64, 3, (1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),
        )
            
        
    def encode(self, cond, pred, temporal_distance=None, action_class=None):
        """_summary_
        Args:
            cond : [B, 3, 3, H, W]
            pred : [B, 3, 3, H, W]
            temporal_distance (_type_, optional): [B,]
            action_class (_type_, optional): [B,]
        """
        cond = self.cond_proj(cond)
        pred = self.pred_proj(pred) # [B, 256, 3, 16, 16]
        
        x = torch.cat([cond, pred], dim=2)
        B, C, T, H, W = x.shape
        frame_idx = temporal_distance_to_frame_idx(temporal_distance-1, T//2, device=x.device) # [B, T]
        
        for cross_attn in self.encoder:
            x = cross_attn(x, frame_idx, action_class)
        
        cond, pred = x[:, :, :T//2], x[:, :, T//2:]
        mu = self.fc_mu(rearrange(pred, 'b c t h w -> b t h w c')).permute(0, 4, 1, 2, 3)
        log_var = self.fc_var(rearrange(pred, 'b c t h w -> b t h w c')).permute(0, 4, 1, 2, 3)
        
        return mu, log_var
    
    def decode(self, z, cond, temporal_distance=None, action_class=None):
        B, C, T, H, W = z.shape # [B, 256, 3, 16, 16]
        cond = self.cond_proj(cond)
        
        frame_idx = temporal_distance_to_frame_idx(temporal_distance-1, T, device=z.device)
        z = torch.cat([cond, z], dim=2)
        
        for cross_attn in self.decoder:
            z = cross_attn(z, frame_idx, action_class)
        
        z = self.conv_out(z) # [B, 3, 6, 64, 64]
        
        return z[:, :, T:]
    
    def loss(self, cond, pred, temporal_distance, action_class, kld_weight=0.0025):
        mu, log_var = self.encode(cond, pred, temporal_distance, action_class)
        B, C, T, H, W = mu.shape
        z = self.reparameterize(mu, log_var)
        pred_hat = self.decode(z, cond, temporal_distance, action_class)
        
        recon_loss = F.mse_loss(pred_hat, pred)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
        kld_loss = kld_loss / (B * T * H * W)
        kld_loss = kld_weight * kld_loss
        
        return recon_loss, kld_loss
    
    def sample(self, cond, temporal_distance, action_class, shape=(256, 3, 16, 16)):
        bs = cond.shape[0]
        z = torch.randn((bs, shape[0], shape[1], shape[2], shape[3]), device=cond.device)
        
        x_hat = self.decode(z, cond, temporal_distance, action_class)
        
        return x_hat

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

class TemporalCondition(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Linear(dim+1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x, temporal_distance=None):
        if temporal_distance == None:
            return x
        
        if len(x.shape) == 3:
            B, N, C = x.shape
            temporal_distance = repeat(temporal_distance, 'B -> B N 1', N=N)
            x = torch.cat([x, temporal_distance], dim=-1)
            x = self.fc(x)
            
        elif len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, 'B C T H W -> (B H W) T C')
            temporal_distance = repeat(temporal_distance, 'B -> (B H W) T', T=T, H=H, W=W)
            
            x = torch.cat([x, temporal_distance.unsqueeze(-1)], dim=-1) # (B*H*W, T, C+1)
            x = self.fc(x) # (B*H*W, T, C)
            
            x = rearrange(x, '(B H W) T C -> B C T H W', B=B, H=H, W=W)
        
        return x
    
class TemplateAttnBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        mlp_ratio=4.0,
        drop=0.,
        norm_type='layer',
    ):
        super().__init__()
        
        self.attn = STAttentionBlock(dim, heads, norm_type=norm_type)
        self.cross_attn = STAttentionBlock(dim, heads, norm_type=norm_type, is_cross=True)
        
    def forward(self, template, cond, temporal_distance=None, pos_bias=None):
        
        template = self.attn(template, temporal_distance=temporal_distance, pos_bias=pos_bias)
        template = self.cross_attn(template, cond=cond, temporal_distance=temporal_distance, pos_bias=pos_bias)
        
        return template

class FeatureCondition(nn.Module):
    def __init__(
        self,
        dim,
        channels=3,
        heads=4,
        depth=4,
        img_size=64,
        patch_size=4,
        norm_type='layer',
    ):
        super().__init__()
        
        self.patch_emb = PatchEmbed(img_size=img_size, embed_dim=dim, in_chans=channels, patch_size=patch_size)
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(STAttentionBlock(dim, heads, norm_type=norm_type))
        
    
        
    def forward(self, x, temporal_distance=None, pos_bias=None):
        
        x = self.patch_emb(x) # [B, C, T, h, w]
        for block in self.blocks:
            x = block(x, temporal_distance=temporal_distance, pos_bias=pos_bias)
        
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, embed_dim=128, in_chans=3, patch_size=4):
        super().__init__()
    
        self.embed_dim = embed_dim    
        self.h, self.w = img_size // patch_size, img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = build_2d_sincos_position_embedding(self.h, self.w, self.embed_dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> (b t) (h w) c', h=self.h, w=self.w, t=T)
        x += self.pos_emb
        x = rearrange(x, '(b t) (h w) c -> b c t h w', h=self.h, w=self.w, t=T)
        
        return x
        
        
class TemplateCondition(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        num_channels=3,
        depth_e=4,
        depth_d=4,
        frame_size=64,
        patch_size=4,
        frame_num=4,
        is_train=False
    ):
        super().__init__()
        
        self.h, self.w = frame_size // patch_size, frame_size // patch_size
        self.is_train = is_train
        # template query
        self.template_query = nn.Parameter(
            torch.zeros(1, dim, frame_num, self.h, self.w) # [B, C, T, H, W]
        )
        trunc_normal_(self.template_query, std=.02)
        
        self.pos_emb = build_2d_sincos_position_embedding(self.h, self.w, dim)
        
        blocks = []
        for i in range(depth_d):
            blocks.append(TemplateAttnBlock(dim, heads))
        self.blocks = nn.ModuleList(blocks)
        
        self.feature_cond = FeatureCondition(dim, heads=heads, depth=depth_e)
        
        self.time_rel_pos_bias = RelativePositionBias(heads=heads, max_distance=32)

        self.displacement_head = nn.Conv2d(dim, 3, 1)

        # if is_train:
        self.first = SameBlock2d(num_channels, dim//4, kernel_size=(7,7), padding=(3,3))
        self.final = nn.Conv2d(dim//4, num_channels, kernel_size=(7,7), padding=(3,3))
        
        
    def loss(self, template, cond, gt):
        template = rearrange(template, 'b c t h w -> (b t) c h w')
        cond = rearrange(cond, 'b c t h w -> (b t) c h w')
        gt = rearrange(gt, 'b c t h w -> (b t) c h w')
        
        B, C, H, W = cond.shape
        
        template = F.interpolate(template, size=(H, W), mode='bilinear', align_corners=False)
        
        flow = template[:, :2] # [B, 2, H, W]
        occlusion = template[:, 2:] # [B, 1, H, W]
        cond = self.first(cond)
        grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
        grid = torch.stack((grid_x, grid_y), 2).float().to(cond.device) # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        vgrid = grid + flow.permute(0, 2, 3, 1)
        vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
        vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
        
        out = F.grid_sample(cond, vgrid, align_corners=False)
        out *= (1-occlusion)
        out = self.final(out)
        
        
        loss = F.l1_loss(out, gt)
        
        return loss
        
        
        
    def forward(self, cond, gt, temporal_distance = None, action=None):
        B, C, T, H, W = cond.shape
        
        template = self.template_query.repeat(B, 1, 1, 1, 1)
        
        cond_emb = self.feature_cond(cond, temporal_distance, pos_bias=self.time_rel_pos_bias(T, device=cond.device))
        
        template = rearrange(template, 'b c t h w -> (b t) (h w) c')
        template = template + self.pos_emb
        template = rearrange(template, '(b t) (h w) c -> b c t h w', h=self.h, w=self.w, t=T)
        
        for block in self.blocks:
            template = block(template, cond_emb, temporal_distance, pos_bias=self.time_rel_pos_bias(T, device=cond.device))
        
        template_displacement = self.displacement_head(rearrange(template, 'b c t h w -> (b t) c h w'))
        template_displacement = rearrange(template_displacement, '(b t) c h w -> b c t h w', b=B)
        
        if self.is_train:
            template_loss = self.loss(template_displacement, cond, gt)
        else:
            template_loss = None        
        return template_loss, template
    
    def warp_with_template(self, cond, template):
        B, C, T, H, W = cond.shape
        template = rearrange(template, 'b c t h w -> (b t) c h w')
        cond = rearrange(cond, 'b c t h w -> (b t) c h w')
        
        template = F.interpolate(template, size=(H, W), mode='bilinear', align_corners=False)
        
        flow = template[:, :2] # [B, 2, H, W]
        occlusion = template[:, 2:] # [B, 1, H, W]
        cond = self.first(cond)
        grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
        grid = torch.stack((grid_x, grid_y), 2).float().to(cond.device) # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B*T, 1, 1, 1)
        
        vgrid = grid + flow.permute(0, 2, 3, 1)
        vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
        vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
        
        out = F.grid_sample(cond, vgrid, align_corners=False)
        out *= (1-occlusion)
        out = self.final(out)
        
        return rearrange(out, '(b t) c h w -> b c t h w', b=B, t=T)
    
    def return_template(self, cond, gt, temporal_distance=None, action=None):
        B, C, T, H, W = cond.shape
        
        template = self.template_query.repeat(B, 1, 1, 1, 1)
        
        cond_emb = self.feature_cond(cond, temporal_distance, pos_bias=self.time_rel_pos_bias(T, device=cond.device))
        
        template = rearrange(template, 'b c t h w -> (b t) (h w) c')
        template = template + self.pos_emb
        template = rearrange(template, '(b t) (h w) c -> b c t h w', h=self.h, w=self.w, t=T)
        
        for block in self.blocks:
            template = block(template, cond_emb, temporal_distance, pos_bias=self.time_rel_pos_bias(T, device=cond.device))
        
        template_displacement = self.displacement_head(rearrange(template, 'b c t h w -> (b t) c h w'))
        template_displacement = rearrange(template_displacement, '(b t) c h w -> b c t h w', b=B)
          
        return template_displacement
    
class MotionAdaptor(nn.Module):
    def __init__(
        self,
        dim,
        temp_emb_dim,
        attn_heads=8,
        attn_dim_head=32,
        depth=4,
    ):
        super().__init__()
        
        self.proj = nn.Conv3d(dim, dim, 1)
        
        self.blocks = nn.ModuleList([])
        for d in range(depth):
            self.blocks.append(
                STAttentionBlock(dim, heads=attn_heads, dim_head=attn_dim_head)
            )
        
        
        
    def forward(self, motion_feature, temporal_distance=None, action_emb=None):
        B, C, T, H, W = motion_feature.shape # [B, 256, T-1, 16, 16]
        
        motion_feature += rearrange(action_emb, 'b c -> b c 1 1 1')
        
        motion_feature = self.proj(motion_feature)
        
        for block in self.blocks:
            motion_feature = block(motion_feature, temporal_distance=temporal_distance)
        
        return motion_feature[:, :, -1] # [B, C, H, W]
        
class ClassCondition(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        
        self.input_dim = input_dim
        
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.layers(x)
    
class MotionAdapter(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        pass
    
    def forward(self, motion_cond):
        """_summary_

        Args:
            motion_cond : [B, 256, T-1, 16, 16]
        """
        pass
        
        
        
        
        
        
        

        
        
        
        