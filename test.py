import os
import sys
import time
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from collections import OrderedDict


def mod_weight(file_name='/afs/chenxu/weight/supernet_fcvid_f16.pth'):
    state_dict = torch.load(file_name)['state_dict']
    out_dict = OrderedDict()
    remove_keys = []
    for k,v in list(state_dict.items()):
        if k.startswith('classifier.'):
            state_dict['final_'+k] = v
            remove_keys.append(k)
            # del state_dict[k]
        elif k.startswith('feat_classifier.'):
            state_dict[k.replace('feat_', '')] = v
            remove_keys.append(k)
            # del state_dict[k]
        else:
            pass
    for k in remove_keys:
        del state_dict[k]
    torch.save(state_dict, '/afs/chenxu/weight/supernet_fcvid_f16_rec.pth', _use_new_zipfile_serialization=False)

if __name__=='__main__':
    mod_weight()
