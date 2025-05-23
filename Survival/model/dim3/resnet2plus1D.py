import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Resnet2plus1D_18(nn.Module):
    def __init__(self, args, weights='DEFAULT', progress=True):
        super(Resnet2plus1D_18, self).__init__()
        
        self.args = args
        self.downsampling = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0))
        self.model = models.video.r2plus1d_18(weights=weights, progress=progress)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(),
                                      nn.Linear(num_features, num_features), nn.ReLU())
    
    def forward(self, x):
        return self.model(self.downsampling(x))