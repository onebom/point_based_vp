import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class AttentionPointSelector(nn.Module):
    def __init__(self, top_k=10):
        super().__init__()
        self.top_k = top_k
        
    def forward(self, x, traj_map):
        # B, C, T, PN = x.shape
        B, PN, T, H, W = traj_map.shape
        x = rearrange(x, 'b c t pn -> b pn (t c)')
        d_k = x.shape[-1]
        
        sim = torch.matmul(x, x.transpose(-2, -1)) * (d_k ** -0.5)  # Shape (B, PN, PN)
        attn = F.softmax(sim, dim=-1) # Shape (B, PN, PN)
        
        scores = attn.mean(dim=-1) # Shape (B, PN)

        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape (B, 128, 1, 1, 1)
        
        # Select from traj_map using topk_indices, result shape will be (B, 128, T, H, W)
        selected_traj_map = torch.gather(traj_map, dim=1, index=topk_indices.expand(-1, -1, T, H, W))

        return selected_traj_map