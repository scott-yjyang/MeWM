import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Resnet_18(nn.Module):
    def __init__(self, args, weights='DEFAULT', progress=True):
        super(Resnet_18, self).__init__()
        
        self.args = args
        self.downsampling = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.model = models.resnet18(weights=weights, progress=progress)
        num_features = self.model.fc.out_features
        self.last_fc = nn.Sequential(nn.Dropout(0.5),
                                     nn.Linear(num_features, self.args.num_classes))
    
    def forward(self, x):
        if self.args.activationF == 'sigmoid':
            return torch.sigmoid(self.last_fc(self.model(self.downsampling(x))))
        elif self.args.activationF == 'softmax':
            return torch.softmax(self.last_fc(self.model(self.downsampling(x))), dim=1)