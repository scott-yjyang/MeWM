import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args):
    if 'CT' in args.modality and 'wMask' in args.model_CT:
        from .aggregator_wMask import aggregator_wMask
        print('aggregator_wMask')
        return aggregator_wMask(args)
    else:
        from .aggregator import aggregator
        return aggregator(args)