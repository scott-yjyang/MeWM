import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import math

class simpleFCs_v1d(nn.Module):
    def __init__(self, args):
        super(simpleFCs_v1d, self).__init__()
        
        self.args = args
        
        self.fc = nn.Sequential(nn.Linear(len(self.args.clinical_features) * math.ceil(512/len(self.args.clinical_features)), 512), nn.ReLU())
    
    def forward(self, x):
        return self.fc(x)