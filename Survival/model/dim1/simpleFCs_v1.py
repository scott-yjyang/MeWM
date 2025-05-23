import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class simpleFCs_v1(nn.Module):
    def __init__(self, args):
        super(simpleFCs_v1, self).__init__()
        
        self.args = args
        
        self.fc = nn.Sequential(nn.Linear(len(self.args.clinical_features), len(self.args.clinical_features)), nn.ReLU(),
                                nn.Linear(len(self.args.clinical_features), 512), nn.ReLU())
    
    def forward(self, x):
        return self.fc(x)