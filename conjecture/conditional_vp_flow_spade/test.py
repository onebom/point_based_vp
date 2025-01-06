import torch
from torch import einsum

# Dummy tensor
x = torch.randn(6, 8192, 18)  # Shape (b, pn, d)

# Check einsum
try:
    sim = torch.matmul(x, x.transpose(-1,-2))*(x.shape[-1] ** -0.5)
    print(sim.shape)
except Exception as e:
    print(f"Error: {e}")