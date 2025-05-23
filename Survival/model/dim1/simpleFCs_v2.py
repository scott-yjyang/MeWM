import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class simpleFCs_v2(nn.Module):
    def __init__(self, args):
        super(simpleFCs_v2, self).__init__()
        
        self.args = args
        # self.fc = nn.Sequential(nn.Sequential(nn.Linear(len(self.args.clinical_features), 1024), nn.ReLU()),
        #                         nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        # )
        self.fc = nn.Sequential(nn.Linear(27, 27), nn.ReLU(),
                                nn.Linear(27, 512), nn.ReLU())
    
    def forward(self, x):
        return self.fc(x)