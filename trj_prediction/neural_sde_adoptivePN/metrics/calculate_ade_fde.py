import torch
def calculate_ade_fde(pred_tensor, gt_tensor):
    b,n,pn,t,c = pred_tensor.shape

    diff = pred_tensor - gt_tensor                       # batch x auto_num x point_num x frames x 2
    dist = torch.norm(diff, dim=-1)                      # batch x auto_num x point_num x frames
    
    ade = dist.mean(axis=-1)                             # batch x auto_num x point_num
    fde = dist[..., -1]                                  # batch x auto_num x point_num
                     
    return ade.mean(), fde.mean()  