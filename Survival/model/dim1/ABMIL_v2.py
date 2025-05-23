import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ABMIL_v2(nn.Module):
    def __init__(self, args):
        super(ABMIL_v2, self).__init__()
        
        self.args = args
        
        # input : N x L(768)
        self.L = 768
        self.D = 192
        self.K = 1

        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
        #     nn.Linear(self.D, self.K)
        #     )
        
        # self.fc = nn.Sequential(
        #     nn.Linear(self.L, 512),
        #     nn.ReLU()
        #     )
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        # self.classifier = nn.Sequential(
        #     # nn.Linear(self.L*self.K, 1)
        #     nn.Linear(self.L*self.K, 512)
        #     # nn.Sigmoid()
        # )
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x, BpRc_class):
        x = x.squeeze(0) # NxL
        x = self.dropout1(x) # NxL
        
        # A = self.attention(x) # NxK
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        
        A = torch.transpose(A, -2, -1)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        
        M = torch.matmul(A, x)  # KxL
        # M = self.dropout2(M)
        
        # return self.fc(M)
        # return self.classifier(M)
        
        M = torch.cat([M, BpRc_class], dim=1) # BpRc_class : torch.Size([1])
        
        return M