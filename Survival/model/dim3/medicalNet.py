import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .MedicalNet.models import resnet

class medicalNet(nn.Module):
    def __init__(self, args):
        super(medicalNet, self).__init__()
        
        self.args = args

        # if self.args.spacing[0] == 2.0:
        #     input_W = 224
        #     input_H = 224
        # elif self.args.spacing[0] == 0.6869:
        #     input_W = 512
        #     input_H = 512
        # input_D = 160

        self.MedicalNet = resnet.resnet101(sample_input_W = 96,
                                    sample_input_H = 96,
                                    sample_input_D = 96,
                                    shortcut_type = 'B',
                                    no_cuda = False,
                                    num_seg_classes = 3)
        # self.MedicalNet = self.MedicalNet.cuda()

        net_dict = self.MedicalNet.state_dict()
        
        ckpt = torch.load("model/dim3/MedicalNet/pretrain/resnet_101.pth")
        ckpt_dict = {k: v for k, v in ckpt['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(ckpt_dict)
        self.MedicalNet.load_state_dict(net_dict)
        
        # for para in self.MedicalNet.parameters():
        #     para.requires_grad = False

        # # self.channel_extend = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        # self.resnet3D = models.video.mc3_18()
        # num_features = self.resnet3D.fc.in_features
        # self.resnet3D.fc = nn.Linear(num_features, 512)
    
    def forward(self, x):
        # return self.resnet3D(self.MedicalNet(x))
        
        output_conv1 = self.MedicalNet.conv1(x)
        output_bn1 = self.MedicalNet.bn1(output_conv1)
        output_relu = self.MedicalNet.relu(output_bn1)
        output_maxpool = self.MedicalNet.maxpool(output_relu)
        output1 = self.MedicalNet.layer1(output_maxpool)
        output2 = self.MedicalNet.layer2(output1)
        # output3 = self.MedicalNet.layer3(output2)
        
        return output2