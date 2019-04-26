import numpy as np

import torch
from tqdm import tqdm

from collections import OrderedDict

def svd_compress(state_dict, k=100, tqdm=False):
    new_state_dict = OrderedDict()
    
    keys_iter = state_dict.keys()
    if tqdm:
        keys_iter = tqdm(keys_iter)
    for key in keys_iter:
        mat = state_dict[key]

        if len(mat.shape) == 2:
            i, j = mat.shape
            U, S, V = torch.svd(mat)
            S[k:] = 0
            recovered_mat = (U @ torch.diag(S) @ V.t())
        else:
            recovered_mat = mat

        new_state_dict[key] = recovered_mat 
    
    return new_state_dict

def weight_prune(state_dict, p=.2, tqdm=False):
    new_state_dict = OrderedDict()
    
    keys_iter = state_dict.keys()
    if tqdm:
        keys_iter = tqdm(keys_iter)
    
    for key in keys_iter:
        mat = state_dict[key]
        abs_mat = mat.abs()

        np_abs_mat = abs_mat.cpu().numpy()
        thresh = np.percentile(np_abs_mat.flatten(), 100 * p)
        
        mat[abs_mat < thresh] = 0
        
        new_state_dict[key] = mat
    
    return new_state_dict
        
