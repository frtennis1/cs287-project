import numpy as np

import torch
from tqdm import tqdm

from collections import OrderedDict
from operator import add, sub


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
        

def binop_apply(s1, s2, op):
    assert s1.keys() == s2.keys()
    new_state_dict = OrderedDict([])
    for k in s1.keys():
        new_state_dict[k] = op(s1[k], s2[k])
    return new_state_dict

class to_bert:
    def __init__(self, base_model):
        self.base_model = base_model
    
    def __call__(self, f, **kwargs):
        base_model = self.base_model
        def wrapped_f(current_model, **kwargs):
            new_state_dict = OrderedDict([])
            diff = binop_apply(current_model, base_model, sub)
            new_diff = f(diff, **kwargs)
            new_state_dict = binop_apply(base_model, new_diff, add)
            return new_state_dict
        return wrapped_f
