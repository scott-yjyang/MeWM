import torch
import torch.nn as nn
import torchvision.models as models

class ResnetMC3_18_wMask(nn.Module):
    def __init__(self, args, weights='DEFAULT', progress=True):
        super(ResnetMC3_18_wMask, self).__init__()

        self.args = args
        self.downsampling = nn.Conv3d(in_channels=2, out_channels=3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.model = models.video.mc3_18(weights=weights, progress=progress)
        # self.model = models.video.mc3_18()
        
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, 512)
    
    def forward(self, x, mask):
        input = self.downsampling(torch.cat([x,mask],dim=1))
        output0 = self.model.stem(input)
        output1 = self.model.layer1(output0)
        output2 = self.model.layer2(output1)
        output3 = self.model.layer3(output2)
        output4 = self.model.layer4(output3)
        # output5 = self.model.avgpool(output4)
        
        return output4