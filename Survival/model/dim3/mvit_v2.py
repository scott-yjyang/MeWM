import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ..dim1.TransMIL import TransMIL

import math

class MViT_v2(nn.Module):
    def __init__(self, args, weights='DEFAULT', progress=True):
        super(MViT_v2, self).__init__()
        
        self.args = args
        # self.downsampling = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.model = models.video.mvit_v2_s(weights=weights, progress=progress)
        num_features = self.model.head[1].in_features
        self.model.head[1] = nn.Linear(num_features, 512)
        
        self.TransMIL = TransMIL(n_classes=self.args.num_classes)
    
    def forward(self, x): # [B, 1, C, H, W] --(squeeze)--> [B, C, H, W]
        # x = self.downsampling(x)
        x_features = torch.zeros((x.shape[0], math.ceil(x.shape[2]/3), 512), dtype=x.dtype, device=x.device)
        x = F.pad(x, (0,0, 0,math.ceil(x.shape[2]/3)-x.shape[2], 0,0, 0,0), "constant", 0)
        for i in range(math.ceil(x.shape[2]/3)):
            x_features[:, i, :] = self.model(x[:, 3*i:3*(i+1), :, :].clone())
        
        return self.TransMIL(x_features)