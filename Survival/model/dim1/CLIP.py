import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CLIP(nn.Module):
    def __init__(self, args):
        super(CLIP, self).__init__()
        
        self.args = args
        
        self.model, self.preprocess = clip.load("ViT-B/32")
        
        if self.args.learnablePrompt:
            self.dtype = self.model.dtype
            self.ctx_dim = self.model.ln_final.weight.shape[0]
            # self.ctx_vectors = torch.empty(self.args.n_prompts, self.args.n_ctx, self.ctx_dim, dtype=self.dtype)
            self.ctx_vectors = torch.empty(len(self.args.clinical_features)+1, self.args.n_ctx, self.ctx_dim, dtype=self.dtype)
            
            nn.init.normal_(self.ctx_vectors, std=0.02)
            
            self.ctx = nn.Parameter(self.ctx_vectors)
    
    def forward(self, x):
        # x_text : "a lung cancer patient photo of sex 1"
        # x = clip.tokenize(x_text) : [len(args.clinical_feature), 77]
        
        if self.args.learnablePrompt:
            # x : tokenized_prompts
            
            with torch.no_grad():
                embedding = self.model.token_embedding(x[0,:,:]).type(self.dtype)
                # embedding = embedding.expand(self.args.n_prompts, -1, -1)
            
            # prefix = embedding[:, :1, :].detach()
            prefix = embedding[:, :1, :]
            # suffix = embedding[:, 1+self.args.n_ctx:, :].detach()
            suffix = embedding[:, 1+self.args.n_ctx:, :]
            
            ctx = self.ctx
            # if ctx.dim() == 2:
            #     ctx = ctx.unsqueeze(0).expand(self.args.n_prompts, -1, -1)
            
            prompts = torch.cat(
                [
                    prefix,    # (n_prompts, 1, dim)
                    ctx,   # (n_prompts, n_ctx, dim)
                    suffix,    # (n_prompts, *, dim)
                ],
                dim = 1
            )
            
            x_ = prompts + self.model.positional_embedding.type(self.dtype)
            x_ = x_.permute(1, 0, 2)
            x_ = self.model.transformer(x_)
            x_ = x_.permute(1, 0, 2)
            x_ = self.model.ln_final(x_).type(self.dtype)
            
            x_ = x_[torch.arange(x_.shape[0]), x.argmax(dim=-1)] @ self.model.text_projection
            
            x_features = x_.float().cuda()
            # x_features = torch.zeros((x.shape[0],self.args.n_prompts,512), device=x.device)
            # for i in range(x.shape[0]):
            #     x_features[i,:,:] = x_[i,:,:]
            
            # x_features = torch.zeros((x.shape[0], suffix.shape[1], 512), device=x.device)
            # for i in range(x.shape[0]):
            #     x_features[i,:,:] = suffix[i,:,:]
        
        else:
            with torch.no_grad():
                x_features = torch.zeros((x.shape[0],x.shape[1],512), device=x.device)
                for i in range(x.shape[0]):
                    x_features[i,:,:] = self.model.encode_text(x[i,:,:]) # [len(args.clinical_feature), 512]
        
        return x_features