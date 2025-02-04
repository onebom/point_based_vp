import torch
import numpy as np
import random

def count_parameters(model):
    res = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"count_training_parameters: {res}")
    res = sum(p.numel() for p in model.parameters())
    print(f"count_all_parameters:      {res}")
    
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True