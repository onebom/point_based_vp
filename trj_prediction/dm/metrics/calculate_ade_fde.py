import numpy as np
def calculate_ade(pred_arr, gt_arr):
    b,n,pn,t,c = pred_arr.shape

    diff = pred_arr - np.expand_dims(gt_arr, axis=1)        # batch x auto_num x point_num x frames x 2
    dist = np.linalg.norm(diff, axis=-1)                    # batch x auto_num x point_num x frames
    ade = dist.mean(axis=-1)                                # batch x auto_num x point_num
                     
    return ade.mean()


def calculate_fde(pred_arr, gt_arr):
    b,n,pn,t,c = pred_arr.shape

    diff = pred_arr - np.expand_dims(gt_arr, axis=1)        # batch x auto_num x point_num x frames x 2
    dist = np.linalg.norm(diff, axis=-1)                    # batch x auto_num x point_num x frames
    fde = dist[..., -1]                                     # batch x auto_num x point_num 
    
    return fde.mean()  