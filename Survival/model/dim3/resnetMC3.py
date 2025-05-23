import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResnetMC3_18(nn.Module):
    def __init__(self, args, weights='DEFAULT', progress=True):
        super(ResnetMC3_18, self).__init__()
        
        self.args = args
        # if self.args.spacing[0] == 2.0:
        self.downsampling = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(1,1,1), stride=(1,1,1), padding=(1,1,1))
        # elif self.args.spacing[0] == 0.6869:
        #     self.downsampling = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1))
        self.model = models.video.mc3_18(weights=weights, progress=progress)
        # self.model = models.video.mc3_18()
        
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, 512)
    
    def forward(self, x):
        input = self.downsampling(x)
        output0 = self.model.stem(input)
        output1 = self.model.layer1(output0)
        output2 = self.model.layer2(output1)
        output3 = self.model.layer3(output2)
        output4 = self.model.layer4(output3)
        # output5 = self.model.avgpool(output4)
        
        return output4
        
        # return self.model(self.downsampling(x))
        # return self.model(x)