import numpy as np

import torch
from tqdm import tqdm

from collections import OrderedDict
from operator import add, sub


from sklearn.cluster import MiniBatchKMeans

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

def quantization(state_dict, n_clusters=128):
    flattened = torch.cat([param.flatten() for param in state_dict.values()]) \
                    .cpu().numpy().reshape(-1, 1)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             max_no_improvement=2).fit(flattened)
    clustered = [kmeans.cluster_centers_[index]
                 for index in kmeans.predict(flattened)]

    # replace the original parameters with cluster centers
    cursor = 0
    for key in state_dict.keys():
        shape = state_dict[key].shape
        size = state_dict[key].numel()
        state_dict[key] = torch.Tensor(clustered[cursor:cursor + size]) \
                                .reshape(shape)
        cursor += size
    return state_dict

def count_param(dict_):
    num_param = sum([x.numel() for x in dict_.values()])
    print("{:,}".format(num_param))
    return num_param

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
