import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ABMIL(nn.Module):
    def __init__(self, args, L=768, D=192, K=1):
        super(ABMIL, self).__init__()
        
        # input : N x L(768)
        self.L = L
        self.D = D
        self.K = K

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

    def forward(self, x):
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
        return M