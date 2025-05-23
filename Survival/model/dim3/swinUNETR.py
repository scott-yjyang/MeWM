import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from monai.networks.nets import SwinUNETR as SwinUNETR_
import random

class SwinUNETR(nn.Module):
    def __init__(self, args, weights='DEFAULT', progress=True):
        super(SwinUNETR, self).__init__()
        
        self.args = args
        
        self.model = SwinUNETR_(
            img_size=(96,96,96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True
        )
        weight = torch.load("./model/dim3/model_swinvit.pt")
        self.model.load_from(weights=weight)
        for para in self.model.parameters():
            para.requires_grad = False
        
        self.normalize = True
        self.n_subsample = 100
        
        # input : N x L(768)
        self.L = 768
        self.D = 192
        self.K = 1
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x_features = torch.zeros((self.n_subsample, self.L)).cuda(non_blocking=True) # 100xL
        for i in range(self.n_subsample):
            h_start = random.randint(0,int(x.shape[-3]-96))
            w_start = random.randint(0,int(x.shape[-2]-96))
            d_start = random.randint(0,int(x.shape[-1]-96))
            hidden_states_out = self.model.swinViT(x[h_start:h_start+96,w_start:w_start+96,d_start:d_start+96], self.normalize)
            x_features[i,:] = F.avg_pool3d(hidden_states_out[4], (3,3,3)).view(-1,hidden_states_out[4].shape[1])
        
        A_V = self.attention_V(x_features) # 100xD
        A_U = self.attention_U(x_features) # 100xD
        A = self.attention_weights(A_V * A_U) # 100xK
        
        A = torch.transpose(A, -2, -1) # Kx100
        A = F.softmax(A, dim=1)
        
        M = torch.matmul(A, x_features) # (Kx100)x(100xL)=(KxL)
        
        return M