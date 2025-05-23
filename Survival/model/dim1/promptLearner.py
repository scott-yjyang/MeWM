import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.metrics import compute_accuracy
# from dassl.utils import load_pretrained_weights, load_checkpoint
# from dassl.optim import build_optimizer, build_lr_scheduler

# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# _tokenizer = _Tokenizer()

import clip


class PromptLearner(nn.Module):
    def __init__(self, args, text):
        
        self.args = args
        
        self.model, self.preprocess = clip.load("ViT-B/32")
        
        dtype = self.model.dtype
        ctx_dim = self.model.ln_final.weight.shape[0]
        ctx_vectors = torch.empty(self.args.n_ctx, ctx_dim, dtype=dtype)
        
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * self.args.n_ctx)
        
        self.ctx = nn.Parameter(ctx_vectors)
        
        text = text.replace(",", "")
        
        prompts = [prompt_prefix + " " + text + "."]
        tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in prompts])
        with torch.no_grad():
            embedding = self.model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :]) # SOS
        self.register_buffer("token_suffix", embedding[:, 1+self.args.n_ctx:, :]) # CLS, EOS
        
    def forward(self):
        if self.ctx.dim() == 2:
            self.ctx = self.ctx.unsqueeze(0).expand(self.args.n_prompts, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = torch.cat(
            [
                prefix,    # (n_prompts, 1, dim)
                self.ctx,  # (n_prompts, n_ctx, dim)
                suffix,    # (n_prompts, *, dim)
            ],
            dim = 1
        )
        
        return prompts