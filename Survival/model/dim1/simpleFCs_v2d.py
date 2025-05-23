import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class simpleFCs_v2d(nn.Module):
    def __init__(self, args):
        super(simpleFCs_v2d, self).__init__()
        
        self.args = args
        
        self.fc = nn.Sequential(nn.Linear(27*19, 512), nn.ReLU())
    
    def forward(self, x):
        return self.fc(x)