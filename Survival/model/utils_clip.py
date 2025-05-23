import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args):
    from .aggregator_clip import aggregator
    return aggregator(args)