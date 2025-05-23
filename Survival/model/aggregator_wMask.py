import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from .sam.transformer import TwoWayTransformer

class aggregator_wMask(nn.Module):
    def __init__(self, args):
        super(aggregator_wMask, self).__init__()
        
        self.args = args
        embedding_dim = 512
        
        weights = 'DEFAULT'
        progress = True
        if self.args.model_CT == 'resnet2plus1d_18_wMask':
            from .dim3 import Resnet2plus1D_18_wMask
            self.extractor_CT1 = Resnet2plus1D_18_wMask(self.args, weights=weights, progress=progress)
        elif self.args.model_CT == 'resnetMC3_18_wMask':
            from .dim3 import ResnetMC3_18_wMask
            self.extractor_CT1 = ResnetMC3_18_wMask(self.args, weights=weights, progress=progress)
        elif self.args.model_CT == 'medicalNet':
            from .dim3 import medicalNet
            self.extractor_CT1 = medicalNet(self.args)
        elif self.args.model_CT == 'SwinUNETR_wMask':
            from .dim3 import SwinUNETR_wMask
            self.extractor_CT1 = SwinUNETR_wMask(self.args, weights=weights, progress=progress)
        elif self.args.model_CT == 'MViT_wMask':
            from .dim3 import MViT_v2_wMask
            self.extractor_CT1 = MViT_v2_wMask(self.args, weights=weights, progress=progress)
        
        if self.args.model_CT == 'resnet2plus1d_18_wMask':
            from .dim3 import Resnet2plus1D_18_wMask
            self.extractor_CT2 = Resnet2plus1D_18_wMask(self.args, weights=weights, progress=progress)
        elif self.args.model_CT == 'resnetMC3_18_wMask':
            from .dim3 import ResnetMC3_18_wMask
            self.extractor_CT2 = ResnetMC3_18_wMask(self.args, weights=weights, progress=progress)
        elif self.args.model_CT == 'medicalNet':
            from .dim3 import medicalNet
            self.extractor_CT2 = medicalNet(self.args)
        elif self.args.model_CT == 'SwinUNETR_wMask':
            from .dim3 import SwinUNETR_wMask
            self.extractor_CT2 = SwinUNETR_wMask(self.args, weights=weights, progress=progress)
        elif self.args.model_CT == 'MViT_wMask':
            from .dim3 import MViT_v2_wMask
            self.extractor_CT2 = MViT_v2_wMask(self.args, weights=weights, progress=progress)
        # self.extractor_CT2 = self.extractor_CT1
        max_seq_len = 100000
        self.pe = torch.zeros((max_seq_len, embedding_dim))
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embedding_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embedding_dim)))
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = self.pe.unsqueeze(0)
        
        self.TwoWayTransformer = TwoWayTransformer(
            args=self.args,
            depth=2,
            embedding_dim=embedding_dim,
            num_heads=8,
            mlp_dim=2048,
        )

        if self.args.aggregator == 'TransMIL_seperate':
            from .dim1 import TransMIL
            self.aggregator_CT1 = TransMIL(n_classes=1, L=embedding_dim)  
            self.aggregator_CT2 = TransMIL(n_classes=1, L=embedding_dim)  
            from .dim1 import ABMIL
            self.aggregator = ABMIL(args, L=embedding_dim)
        elif self.args.aggregator == 'ABMIL':
            from .dim1 import ABMIL
            self.aggregator = ABMIL(args, L=embedding_dim)
        elif self.args.aggregator == 'ABMIL_v2':
            from .dim1 import ABMIL_v2
            self.aggregator = ABMIL_v2(args, L=embedding_dim)
        elif self.args.aggregator == 'TransMIL':
            from .dim1 import TransMIL
            self.aggregator = TransMIL(n_classes=1, L=embedding_dim)  
        
        self.risk_predictor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2) 
        )
        self.surv_predictor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  
        )
            
    def forward(self, x_list, mask_list):
        # x_list = [x_pre_CT, x_post_CT]
        x_input_CT_pre = self.extractor_CT1(x_list[0], mask_list[0])
        x_input_CT_post = self.extractor_CT2(x_list[1], mask_list[1])
        
        b,c,t,h,w = x_input_CT_pre.shape
        # if self.args.model_CT == 'resnetMC3_18':
        pre2post, post2pre = self.TwoWayTransformer(
            x_input_CT_pre, 
            self.pe[:,:t].cuda(), 
            x_input_CT_post
        )  # [B, T, C], [B, T, C]
        
        
        # x_CT_pre_feat = self.aggregator_CT1(pre2post)[0][:,None,:]  # [B, 1, C]
        # x_CT_post_feat = self.aggregator_CT2(post2pre)[0][:,None,:]  # [B, 1, C]
        
        # 拼接特征
        x0 = torch.cat([
            # x_CT_pre_feat, 
            # x_CT_post_feat,
            pre2post,
            post2pre
        ], dim=1)  # [B, 2, C]
        x0 = self.aggregator(x0)  # [B, C]
        
        # 输出风险得分
        # print(x0.shape)
        out = self.risk_predictor(x0)  # [B, 1]
        out = out.view(out.size(0), -1) 

        risk_score = out[:, 0:1]
        risk_score = torch.sigmoid(risk_score)
        surv_score = out[:, 1:2]
        surv_score = surv_score.view(surv_score.size(0), -1)  

        # print(risk_score.shape)
        return risk_score, surv_score, x_input_CT_pre, x_input_CT_post


def cox_loss(risk_scores, survival_time, event_indicator, eps=1e-7):
    risk_scores = risk_scores.view(-1)
    survival_time = survival_time.view(-1)
    event_indicator = event_indicator.view(-1)
    
    n_samples = len(risk_scores)
    log_risk = 0
    
    for i in range(n_samples):
        if event_indicator[i] == 1:
            risk_set = survival_time >= survival_time[i]
            risk_set_scores = risk_scores[risk_set]
            log_sum = torch.log(torch.sum(torch.exp(risk_set_scores)) + eps)  
            log_risk += risk_scores[i] - log_sum
    
    loss = -log_risk / (torch.sum(event_indicator) + eps) if event_indicator.sum() > 0 else -log_risk
    return loss

def cox_loss_vectorized(risk_scores, survival_time, event_indicator, eps=1e-7):
    risk_scores = risk_scores.view(-1)
    survival_time = survival_time.view(-1)
    event_indicator = event_indicator.view(-1)
    
    risk_set = survival_time.unsqueeze(1) >= survival_time.unsqueeze(0)  
    exp_scores = torch.exp(risk_scores)
    sum_exp = torch.log(torch.matmul(risk_set.float(), exp_scores.unsqueeze(1)).squeeze() + eps)
    
    partial_likelihood = event_indicator * (risk_scores - sum_exp)
    loss = -partial_likelihood.sum() / (event_indicator.sum() + eps)
    return loss

class CoxLoss(nn.Module):
    def __init__(self, vectorized=True):
        super(CoxLoss, self).__init__()
        self.vectorized = vectorized
    
    def forward(self, risk_scores, survival_time, event_indicator):
        if self.vectorized:
            return cox_loss_vectorized(risk_scores, survival_time, event_indicator)
        else:
            return cox_loss(risk_scores, survival_time, event_indicator)