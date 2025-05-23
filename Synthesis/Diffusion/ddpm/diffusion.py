import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding

from ddpm.text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import Dataset, DataLoader
from vq_gan_3d.model.vqgan import VQGAN


import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer

# Initialize CLIP model for text embedding

# In Unet3D or GaussianDiffusion initialization, we would do:
# self.text_encoder = text_encoder
# self.tokenizer = tokenizer

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model


import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from transformers import CLIPTokenizer, CLIPTextModel
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

cache_dir = "/home/yyang303/project/Synthesis/"
import torch
import torch.nn as nn

class Attentiontext(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context=None):
        context = context if context is not None else x

        q = self.to_q(x)  # [batch, seq_length, dim_head * heads]
        k, v = self.to_kv(context).chunk(2, dim=-1)  # [batch, seq_length, dim_head * heads]

        q = q.view(x.size(0), x.size(1), self.heads, self.dim_head).transpose(1, 2) * self.scale  # [batch, heads, seq_length, dim_head]
        k = k.view(context.size(0), context.size(1), self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, seq_length, dim_head]
        v = v.view(context.size(0), context.size(1), self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, seq_length, dim_head]

        attn = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq_length, seq_length]
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # [batch, heads, seq_length, dim_head]
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.heads * self.dim_head)  # [batch, seq_length, heads * dim_head]

        return self.to_out(out)  # [batch, seq_length, dim]

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, text_dim=1024):
        super().__init__()
        self.dim = dim
        self.attn = Attentiontext(dim, heads=heads, dim_head=dim_head)
        self.text_projection = nn.Linear(text_dim, dim)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, img_features, text_features):
        batch_size, channels, depth, height, width = img_features.shape

        text_proj = self.text_projection(text_features) 
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        text_proj = text_proj.expand(-1, -1, depth, height, width)  

        if channels != self.dim:
            raise ValueError(f"img_features channels ({channels}) must match dim ({self.dim}) in CrossAttention")

        fused_features = img_features + text_proj 

        fused_features = fused_features.view(batch_size, self.dim, -1).transpose(1, 2) 

        attended = self.attn(fused_features) 

        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.dim, depth, height, width)  # [batch, dim, depth, height, width]
        out = self.to_out(attended) + img_features 

        return out

 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        use_cross_attention=True,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8,
        clip_model_name="openai/clip-vit-base-patch32"
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim 
        self.init_dim = default(init_dim, dim) 
        self.init_kernel_size = init_kernel_size
        self.use_cross_attention = use_cross_attention
        
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): 
            return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        self.init_conv = nn.Conv3d(49, init_dim, (1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding))

        if self.use_cross_attention:
            self.cross_attn = CrossAttention(dim=self.dim, heads=8, dim_head=64)
    
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)
        
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        prob_focus_present=0.,
        text=None
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'

        #print(text)
        if text is not None:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(x.device)
            text_embeddings = self.clip_model.get_text_features(**inputs)  # (batch_size, embedding_dim)
            
        #print("text",text_embeddings.shape)
        
        x = torch.cat([x, cond], dim=1)
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        input_channels = x.shape[1] 

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # Text embeddings passed through all layers
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                            self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []
        # print("x",x.shape)
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            #x = self.cross_attn(x, text_embeddings)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        #x = self.cross_attn(x, text_embeddings) 
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            #x = self.cross_attn(x, text_embeddings) 
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)
    
class MedicalTextEncoder(nn.Module):
    def __init__(self, tokenizer, clip_model, max_segments=8, clip_dim=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        
        # 定义药物词表
        self.drug_list = [
            "Raltitrexed", "Epirubicin", "Oxaliplatin", "Lobaplatin",
            "Idarubicin", "Cisplatin", "THP", "Mitomycin", "Doxorubicin",
            "Hydroxycamptothecin", "LC beads",
            "Nedaplatin", "Pirarubicin", "Lipiodol", "PVA",
            "KMG", "Absolute Alcohol", "NBCA", "Gelatin Sponge"
        ]
        
        # drug embedding
        self.drug_embedding = nn.Embedding(len(self.drug_list), 64)
        
        # text feature processing layer
        self.segment_encoder = nn.Linear(clip_dim, 256)
        
        # fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(256*max_segments + 64, 512),  
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        
        self.max_segments = max_segments

    def split_text(self, text):
        """Strictly split by semicolon and normalize"""
        segments = [s.strip() for s in text.split(';') if len(s.strip()) > 3]  # filter empty sentences
        
        if len(segments) > self.max_segments:
            # strategy 1: merge short paragraphs
            merged = [' '.join(segments[i:i+2]) for i in range(0, len(segments), 2)]
            return merged[:self.max_segments]
        else:
            # strategy 2: fill empty paragraphs
            return segments + ['']*(self.max_segments - len(segments))

    def extract_drug_features(self, text):
        """Extract drug features from text"""
        drug_indices = []
        for i, drug in enumerate(self.drug_list):
            if drug in text:
                drug_indices.append(i)
        
        if not drug_indices:
            return torch.zeros(64, device=self.drug_embedding.weight.device)
        
        drug_embeddings = self.drug_embedding(torch.tensor(drug_indices, 
                         device=self.drug_embedding.weight.device))
        return drug_embeddings.mean(dim=0)  

    def forward(self, text_list):
        batch_embs = []
        device = next(self.clip_model.parameters()).device
        
        for text in text_list:
            # split text
            segments = self.split_text(text)
            
            # CLIP text encoding
            seg_embs = []
            for seg in segments:
                inputs = self.tokenizer(seg, return_tensors="pt", 
                                      padding=True, truncation=True).to(device)
                emb = self.clip_model.get_text_features(**inputs)
                seg_embs.append(self.segment_encoder(emb))
            
            # extract drug features
            drug_emb = self.extract_drug_features(text)
            
            # concatenate and fuse features
            text_emb = torch.cat(seg_embs, dim=1)
            combined_emb = torch.cat([text_emb.squeeze(0), drug_emb])
            merged = self.fusion_layer(combined_emb)
            
            batch_embs.append(merged.unsqueeze(0))
        
        return torch.cat(batch_embs, dim=0)

import random
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=200,
        loss_type='l1',
        use_dynamic_thres=False, 
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
        contrastive_temp=0.07,
        contrastive_weight=0.1
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.med_text_encoder = MedicalTextEncoder(self.denoise_fn.tokenizer, self.denoise_fn.clip_model)
        self.contrastive_temp = contrastive_temp 
        self.contrastive_weight = contrastive_weight

        if vqgan_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
            self.vqgan.eval()
        else:
            self.vqgan = None

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)


        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


        register_buffer('posterior_variance', posterior_variance)


        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))


        self.text_use_bert_cls = text_use_bert_cls

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile
        self.text_proj = nn.Linear(256, 32)  # other suitable output dimension

    def get_another_report(self, text):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "con_report/surgery_action.txt")

        def process_file(file_path):
            text_lists = []
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    report = line.strip().replace("\n", "").replace("\r", "")
                    text_lists.append(report)
            return text_lists

        def contains_keywords(text, keywords):
            return any(keyword in text for keyword in keywords)

        keywords = [
            "Raltitrexed", "Epirubicin", "Oxaliplatin", "Lobaplatin",
            "Idarubicin", "Cisplatin", "THP", "Mitomycin", "Doxorubicin",
            "Hydroxycamptothecin", "LC beads",
            "Nedaplatin", "Pirarubicin", "Lipiodol", "PVA",
            "KMG", "Absolute Alcohol", "NBCA", "Gelatin Sponge"
        ]
        text_data = process_file(file_path)
        text1_keywords = [keyword for keyword in keywords if keyword in text]
        text2 = random.choice(text_data)
        while text2 == text or contains_keywords(text2, text1_keywords):
            text2 = random.choice(text_data)
        # print(text2)
        return text2
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1., text=None):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale, text=text)  # Include text
        )

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True, text=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale, text=text  # Pass text to p_mean_variance
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                    *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1., text=None):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        
        #print('text', text.shape)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(
                img, 
                torch.full((b,), i, device=device, dtype=torch.long), 
                cond=cond, 
                cond_scale=cond_scale, 
                text=text  # Pass the text argument here
            )

        return img


    @torch.inference_mode()
    def sample(self, cond=None, text=None, cond_scale=1., batch_size=16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = batch_size
        image_size = self.image_size
        channels = 8  # self.channels
        num_frames = self.num_frames


        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale, text=text)

        if isinstance(self.vqgan, VQGAN):
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

            _sample = self.vqgan.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    
    def contrastive_loss_3d(self, img1, img2, temperature=0.07):
        if img1.shape != img2.shape:
            img2 = F.interpolate(img2, size=img1.shape[2:], mode='trilinear', align_corners=False)
        
        assert img1.shape == img2.shape, "Shapes should now match after resizing"
        
        b, c, d, h, w = img1.shape  
        img1_flat = img1.view(b, -1)  
        img2_flat = img2.view(b, -1)  

        sim_matrix = F.cosine_similarity(img1_flat.unsqueeze(1), img2_flat.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / temperature
        sim_matrix_exp = torch.exp(sim_matrix)
        
        positive_sim = torch.diag(sim_matrix_exp)
        contrastive_loss = -torch.log(
            positive_sim / (sim_matrix_exp.sum(dim=-1) - positive_sim)
        ).mean()

        return contrastive_loss
    def get_cond(self, img, mask, post_img, post_mask, text, device):
        bs = img.shape[0]
        
        # select processing mode based on vascular angiography description
        def select_occlusion_mode(text):
            """Parse text description to determine processing intensity"""
            text_lower = text[0].lower()  # assume text is the same description in the batch
            if 'occlude' in text_lower:
                return 'partial'    
            elif 'disappear' in text_lower:
                return 'complete'   # completely disappear
            elif 'reduced' or 'improve' in text_lower:
                return 'reduced'    # significantly reduced
            else:
                return 'baseline'   # baseline processing

        # generate dynamic gaussian kernel (adjust parameters based on processing mode)
        def get_adaptive_gaussian_kernel3d(mask, mode):
            """Adjust blur parameters based on processing mode"""
            voxel_ratio = mask.sum() / mask.numel()
            
            # different mode parameters configuration
            params = {
                'complete': {'base_sigma': 8.0, 'max_sigma': 12.0, 'exponent': 0.3},
                'partial':  {'base_sigma': 4.0, 'max_sigma': 8.0,  'exponent': 0.5},
                'reduced':  {'base_sigma': 6.0, 'max_sigma': 10.0, 'exponent': 0.4},
                'baseline':{'base_sigma': 2.0, 'max_sigma': 5.0,  'exponent': 0.7}
            }[mode]
            
            sigma = params['base_sigma'] + (params['max_sigma'] - params['base_sigma']) * (1 - voxel_ratio)
            return sigma, params['exponent']

        # dynamic morphological preprocessing
        def preprocess_mask_3d(mask, mode):
            """
            3D morphological processing function
            Args:
                mask: 3D tumor mask [B, 1, D, H, W], value range {0, 1}
                mode: processing mode ('complete', 'partial', 'reduced', 'baseline')
            Returns:
                processed_mask: processed 3D mask [B, 1, D, H, W]
            """
            import cv2
            import numpy as np
            
            # get batch size
            batch_size = mask.shape[0]
            # create output tensor
            processed = torch.zeros_like(mask)
            
            # process each batch separately
            for b in range(batch_size):
                # get single sample and remove channel dimension
                mask_np = mask[b, 0].cpu().numpy()  # [D, H, W]
                
                # define 3D structure element
                def get_3d_kernel(size):
                    z, y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1, -size//2:size//2+1]
                    kernel = (x**2 + y**2 + z**2) <= (size//2)**2
                    return kernel.astype(np.uint8)
                
                # select processing mode
                kernel_size = max(3, int(np.cbrt(mask_np.sum()) * 0.1))  # dynamic kernel size
                kernel = get_3d_kernel(kernel_size)
                
                # process each depth slice
                processed_slices = np.zeros_like(mask_np)
                for z in range(mask_np.shape[0]):
                    slice_2d = mask_np[z]  # [H, W]
                    
                    if mode == 'complete':
                        processed_slices[z] = cv2.erode(slice_2d.astype(np.float32), kernel[kernel_size//2], iterations=2)
                    elif mode == 'partial':
                        processed_slices[z] = cv2.erode(slice_2d.astype(np.float32), kernel[kernel_size//2], iterations=1)
                    elif mode == 'reduced':
                        eroded = cv2.erode(slice_2d.astype(np.float32), kernel[kernel_size//2])
                        processed_slices[z] = cv2.dilate(eroded, kernel[kernel_size//2])
                    else:  # baseline
                        processed_slices[z] = slice_2d
                
                processed[b, 0] = torch.tensor(processed_slices, device=mask.device)
            
            return processed

        def get_gaussian_kernel3d(kernel_size=7, sigma=2.0, device=None):
            # create isotropic gaussian kernel (three-dimensional same sigma)
            kernel_size = [kernel_size] * 3
            sigma = [sigma] * 3  # keep isotropic
            
            x = torch.linspace(-kernel_size[0]//2, kernel_size[0]//2, kernel_size[0], device=device)
            y = torch.linspace(-kernel_size[1]//2, kernel_size[1]//2, kernel_size[1], device=device)
            z = torch.linspace(-kernel_size[2]//2, kernel_size[2]//2, kernel_size[2], device=device)
            
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
            
            gaussian = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2*(sigma[0]**2)))
            gaussian = gaussian / gaussian.sum()
            
            return gaussian.view(1, 1, *gaussian.shape)

        # step 1: determine processing mode
        mode = select_occlusion_mode(text)
        
        # step 2: morphological preprocessing
        processed_mask = preprocess_mask_3d(mask, mode)
        
        # step 3: dynamic blur processing
        sigma, exponent = get_adaptive_gaussian_kernel3d(processed_mask, mode)
        gaussian_kernel = get_gaussian_kernel3d(kernel_size=11, sigma=sigma, device=img.device)
        
        # keep spatial size padding
        padding = 11 // 2
        
        # apply gaussian blur
        mask_blur = F.conv3d(processed_mask.view(-1, 1, *processed_mask.shape[2:])+1e-10,
                            gaussian_kernel,
                            padding=padding,
                            groups=1).view_as(processed_mask)
        
        # enhance center retention effect
        mask_blur = torch.pow(mask_blur, exponent)
        
        # generate masked image (adjust decay intensity based on mode)
        # decay_factor = {'complete': 0.1, 'partial': 0.3, 'reduced': 0.5, 'baseline': 0.7}[mode]
        masked_img = img * (1 - mask_blur) + img * mask_blur

        # --- keep original VQGAN processing流程 ---
        masked_img = masked_img.permute(0, 1, -1, -3, -2)
        img = img.permute(0, 1, -1, -3, -2)
        mask = mask.permute(0, 1, -1, -3, -2)

        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                img_vq = self.vqgan.encode(img, quantize=False, include_embeddings=True)
                img_vq = ((img_vq - self.vqgan.codebook.embeddings.min()) / 
                    (self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0

                masked_img_vq = self.vqgan.encode(masked_img, quantize=False, include_embeddings=True)
                masked_img_vq = ((masked_img_vq - self.vqgan.codebook.embeddings.min()) / 
                            (self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            img_vq = normalize_img(img)
            masked_img_vq = normalize_img(masked_img)

        mask = mask * 2.0 - 1.0
        cc = torch.nn.functional.interpolate(mask, size=masked_img_vq.shape[-3:])
        cond = torch.cat((masked_img_vq, cc), dim=1)

        # ==== improved text processing ====
        if text is not None:
            # batch process all texts
            text_emb = self.med_text_encoder(text)  # [B, 512]
            
            # use fixed projection layer
            text_emb = self.text_proj(text_emb)  # [B, cond_dim]
            
            # dimension adjustment and concatenation
            text_emb = text_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            text_emb = text_emb.expand(-1, -1, *cond.shape[2:])
            cond = torch.cat((cond, text_emb), dim=1)

        return cond.detach()

    def p_losses(self, x_start, t, cond=None, text=None, noise=None, pre_img_sim=None, pre_mask_sim=None, post_img_sim=None, post_mask_sim=None, text_sim=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls).to(device)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond, text=text, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        cond_sim = None
        if pre_img_sim is not None and pre_mask_sim is not None and post_img_sim is not None and post_mask_sim is not None and text_sim is not None:
            cond_sim = self.get_cond(pre_img_sim, pre_mask_sim, post_img_sim, post_mask_sim, text_sim, device)

        text2 = self.get_another_report(text)
        if text2 is not None:
            # get text2 text embedding
            inputs2 = self.denoise_fn.tokenizer(text2, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeddings2 = self.denoise_fn.clip_model.get_text_features(**inputs2)
            text_proj = nn.Linear(text_embeddings2.shape[-1], cond.shape[1]).to(device)
            text_embeddings2 = text_proj(text_embeddings2)
            text_embeddings2 = text_embeddings2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            text_embeddings2 = text_embeddings2.expand(-1, -1, cond.shape[2], cond.shape[3], cond.shape[4])
            
            cond2 = cond.clone()
            cond2[:, -text_embeddings2.shape[1]:] = text_embeddings2  # replace the last text embedding part

            img1 = self.p_sample_loop(x_start.shape, cond=cond, text=text)
            img2 = self.p_sample_loop(x_start.shape, cond=cond2, text=text2)
            contrastive_loss_dissim = self.contrastive_loss_3d(img1, img2, temperature=self.contrastive_temp)
        else:
            contrastive_loss_dissim = 0

        contrastive_loss_sim = 0
        if cond_sim is not None:
            img3 = self.p_sample_loop(x_start.shape, cond=cond_sim, text=text_sim)
            contrastive_loss_sim = self.contrastive_loss_3d(img1, img3, temperature=self.contrastive_temp)

        # print(text_sim)
        # print(text2)
        # print(text)
        total_loss = loss + self.contrastive_weight * (contrastive_loss_dissim - contrastive_loss_sim)

        return total_loss

    def forward(self, x, cond=None, text=None, return_cond=False, *args, **kwargs):
        pre_img, pre_mask, post_img, post_mask = x
        bs = pre_img.shape[0]
        
        # select processing mode based on vascular angiography description
        def select_occlusion_mode(text):
            """Parse text description to determine processing intensity"""
            text_lower = text[0].lower()  # assume text is the same description in the batch
            if 'occlude' in text_lower:
                return 'partial'    # occlusion of the feeding artery - partial shrinkage
            elif 'disappear' in text_lower:
                return 'complete'   # completely disappear
            elif 'reduced' or 'improve' in text_lower:
                return 'reduced'    # significantly reduced
            else:
                return 'baseline'   # baseline processing

        # generate dynamic gaussian kernel (adjust parameters based on processing mode)
        def get_adaptive_gaussian_kernel3d(mask, mode):
            """Adjust blur parameters based on processing mode"""
            voxel_ratio = mask.sum() / mask.numel()
            
            # different mode parameters configuration
            params = {
                'complete': {'base_sigma': 8.0, 'max_sigma': 12.0, 'exponent': 0.3},
                'partial':  {'base_sigma': 4.0, 'max_sigma': 8.0,  'exponent': 0.5},
                'reduced':  {'base_sigma': 6.0, 'max_sigma': 10.0, 'exponent': 0.4},
                'baseline':{'base_sigma': 2.0, 'max_sigma': 5.0,  'exponent': 0.7}
            }[mode]
            
            sigma = params['base_sigma'] + (params['max_sigma'] - params['base_sigma']) * (1 - voxel_ratio)
            return sigma, params['exponent']

        # dynamic morphological preprocessing
        def preprocess_mask_3d(mask, mode):
            """
            3D morphological processing function
            Args:
                mask: 3D tumor mask [B, 1, D, H, W], value range {0, 1}
                mode: processing mode ('complete', 'partial', 'reduced', 'baseline')
            Returns:
                processed_mask: processed 3D mask [B, 1, D, H, W]
            """
            import cv2
            import numpy as np
            
            # get batch size
            batch_size = mask.shape[0]
            # create output tensor
            processed = torch.zeros_like(mask)
            
            # process each batch separately
            for b in range(batch_size):
                # get single sample and remove channel dimension
                mask_np = mask[b, 0].cpu().numpy()  # [D, H, W]
                
                # define 3D structure element
                def get_3d_kernel(size):
                    z, y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1, -size//2:size//2+1]
                    kernel = (x**2 + y**2 + z**2) <= (size//2)**2
                    return kernel.astype(np.uint8)
                
                # select processing mode
                kernel_size = max(3, int(np.cbrt(mask_np.sum()) * 0.1))  # dynamic kernel size
                kernel = get_3d_kernel(kernel_size)
                
                # process each depth slice
                processed_slices = np.zeros_like(mask_np)
                for z in range(mask_np.shape[0]):
                    slice_2d = mask_np[z]  # [H, W]
                    
                    if mode == 'complete':
                        processed_slices[z] = cv2.erode(slice_2d.astype(np.float32), kernel[kernel_size//2], iterations=2)
                    elif mode == 'partial':
                        processed_slices[z] = cv2.erode(slice_2d.astype(np.float32), kernel[kernel_size//2], iterations=1)
                    elif mode == 'reduced':
                        eroded = cv2.erode(slice_2d.astype(np.float32), kernel[kernel_size//2])
                        processed_slices[z] = cv2.dilate(eroded, kernel[kernel_size//2])
                    else:  # baseline
                        processed_slices[z] = slice_2d
                
                # put the processed result back to tensor
                processed[b, 0] = torch.tensor(processed_slices, device=mask.device)
            
            return processed

        # generate dynamic gaussian kernel
        def get_gaussian_kernel3d(kernel_size=7, sigma=2.0, device=None):
            # create isotropic gaussian kernel (three-dimensional same sigma)
            kernel_size = [kernel_size] * 3
            sigma = [sigma] * 3  # keep isotropic
            
            x = torch.linspace(-kernel_size[0]//2, kernel_size[0]//2, kernel_size[0], device=device)
            y = torch.linspace(-kernel_size[1]//2, kernel_size[1]//2, kernel_size[1], device=device)
            z = torch.linspace(-kernel_size[2]//2, kernel_size[2]//2, kernel_size[2], device=device)
            
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
            
            # calculate gaussian value (keep isotropic)
            gaussian = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2*(sigma[0]**2)))
            gaussian = gaussian / gaussian.sum()
            
            return gaussian.view(1, 1, *gaussian.shape)

        # step 1: determine processing mode
        mode = select_occlusion_mode(text)
        
        # step 2: morphological preprocessing
        processed_mask = preprocess_mask_3d(pre_mask, mode)
        
        # step 3: dynamic blur processing
        sigma, exponent = get_adaptive_gaussian_kernel3d(processed_mask, mode)
        gaussian_kernel = get_gaussian_kernel3d(kernel_size=11, sigma=sigma, device=pre_img.device)
        
        # keep spatial size padding
        padding = 11 // 2
        
        # apply gaussian blur
        mask_blur = F.conv3d(processed_mask.view(-1, 1, *processed_mask.shape[2:])+1e-10,
                            gaussian_kernel,
                            padding=padding,
                            groups=1).view_as(processed_mask)
        
        # enhance center retention effect
        mask_blur = torch.pow(mask_blur, exponent)
        
        # generate masked image (adjust decay intensity based on mode)
        # decay_factor = {'complete': 0.1, 'partial': 0.3, 'reduced': 0.5, 'baseline': 0.7}[mode]
        masked_img = pre_img * (1 - mask_blur) + pre_img * mask_blur
        
        # create reverse mask (bbox area is 0, other areas are 1)
        # mask_ = (1 - bbox_mask).detach()
        # masked_img = (pre_img * mask_).detach()
        masked_img = masked_img.permute(0, 1, -1, -3, -2)
        pre_img = pre_img.permute(0, 1, -1, -3, -2)
        pre_mask = pre_mask.permute(0, 1, -1, -3, -2)
        post_img = post_img.permute(0, 1, -1, -3, -2)
        post_mask = post_mask.permute(0, 1, -1, -3, -2)

        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                img = self.vqgan.encode(post_img, quantize=False, include_embeddings=True)
                img = ((img - self.vqgan.codebook.embeddings.min()) / 
                      (self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0

                masked_img = self.vqgan.encode(masked_img, quantize=False, include_embeddings=True)
                masked_img = ((masked_img - self.vqgan.codebook.embeddings.min()) / 
                             (self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            img = normalize_img(post_img)
            masked_img = normalize_img(masked_img)

        mask = pre_mask * 2.0 - 1.0
        cc = torch.nn.functional.interpolate(mask, size=masked_img.shape[-3:])
        cond = torch.cat((masked_img, cc), dim=1)

        # ==== improved text processing ====
        if text is not None:
            # batch process all texts
            text_emb = self.med_text_encoder(text)  # [B, 512]
            
            # use fixed projection layer
            text_emb = self.text_proj(text_emb)  # [B, cond_dim]
            
            # dimension adjustment and concatenation
            text_emb = text_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            text_emb = text_emb.expand(-1, -1, *cond.shape[2:])
            cond = torch.cat((cond, text_emb), dim=1)
        # print("img.shape", img.shape)
        b, device, img_size = img.shape[0], img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if return_cond:
            return cond

        return self.p_losses(img, t, cond=cond, text=text, *args, **kwargs)

def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5

# trainer clas

# trainer clas
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt





from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from itertools import cycle

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        folder=None,
        dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='/home/yyang303/project/Synthesis/outputs/',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.cfg = cfg
        dl = dataset

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0  # Ensure self.step is initialized here

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(str(self.results_folder) + '/logs'):
            os.makedirs(str(self.results_folder) + '/logs')
        self.writer = SummaryWriter(str(self.results_folder) + '/logs')

        # Add a list to store loss values
        self.losses = []

        self.reset_parameters()


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
        
    def step_ema(self):
        """ Update EMA, ensuring it starts at step_start_ema """
        if self.step == self.step_start_ema:
            model_dict = self.model.state_dict()
            ema_model_dict = self.ema_model.state_dict()

            missing_in_ema = [k for k in model_dict if k not in ema_model_dict]
            missing_in_model = [k for k in ema_model_dict if k not in model_dict]

            if missing_in_ema:
                print(f"Keys in model but not in ema_model: {missing_in_ema}")
            if missing_in_model:
                print(f"Keys in ema_model but not in model: {missing_in_model}")

            self.ema_model.load_state_dict(self.model.state_dict(), strict=False)
        elif self.step >= self.step_start_ema:

            self.ema.update_model_average(self.ema_model, self.model)



    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        torch.save(data, self.results_folder / f'{milestone}.pt')

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            milestones = [int(p.stem.split('-')[-1]) for p in self.results_folder.glob('**/*.pt')]
            milestone = max(milestones) if milestones else None

        if milestone is not None:
            data = torch.load(self.results_folder / f'{milestone}.pt', map_location=map_location)
            self.step = data['step']
            self.model.load_state_dict(data['model'], **kwargs)
            self.ema_model.load_state_dict(data['ema'], **kwargs)
            self.scaler.load_state_dict(data['scaler'])

    def get_similar(self, current_name, text):
        if not text:
            return None, None, None, None, None
            
        attempts = 0
        while attempts < self.len_dataloader:
            item = next(self.dl)
            attempts += 1
            
            # skip current case
            if item['name'][0] == current_name:
                continue
                
            # get text data
            text_data = item.get('text', [])
            if not isinstance(text_data, list):
                continue

            # check if there is an exact text match
            if text in text_data:
                # process image and mask data
                pre_image = item['image.pre'].cuda() if isinstance(item['image.pre'], torch.Tensor) else torch.tensor(item['image.pre']).float().cuda()
                pre_mask = item['label.pre'].cuda() if isinstance(item['label.pre'], torch.Tensor) else torch.tensor(item['label.pre']).float().cuda()
                pre_mask[pre_mask == 1] = 0
                pre_mask[pre_mask >= 2] = 1
                
                post_image = item['image.post'].cuda() if isinstance(item['image.post'], torch.Tensor) else torch.tensor(item['image.post']).float().cuda()
                post_mask = item['label.post'].cuda() if isinstance(item['label.post'], torch.Tensor) else torch.tensor(item['label.post']).float().cuda()
                post_mask[post_mask == 1] = 0
                post_mask[post_mask >= 2] = 1
                
                print(f"Found exact text match for {current_name}")
                return pre_image, pre_mask, post_image, post_mask, item['text']
        
        print(f"No exact text match found for {current_name}")
        return None, None, None, None, None

    def train(self, prob_focus_present=0., focus_present_mask=None, log_fn=lambda x: x):
        best_train_loss = float('inf')
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)
                
                # add checkpoint
                # print("\nDataloader output check:")
                # print(f"image.pre shape: {data['image.pre'].shape}")
                # print(f"label.pre shape: {data['label.pre'].shape}")
                # print(f"image.post shape: {data['image.post'].shape}")
                # print(f"label.post shape: {data['label.post'].shape}")
                
                image = data['image.pre'].cuda()
                mask = data['label.pre'].cuda()
                post_image = data['image.post'].cuda()
                post_mask = data['label.post'].cuda()
                text = data.get('text', None)  
                name = data.get('name', None) 
                print(name)
                name = name[0]
                mask[mask == 1] = 0
                mask[mask >= 2] = 1
                post_mask[post_mask == 1] = 0
                post_mask[post_mask >= 2] = 1

                # input_data = torch.cat([image, mask, post_image, post_mask], dim=0)
                input_data = [image, mask, post_image, post_mask]
                keyword = []
                keywords = [
                        "Raltitrexed", "Epirubicin", "Oxaliplatin", "Lobaplatin",
                        "Idarubicin", "Cisplatin", "THP", "Mitomycin", "Doxorubicin",
                        "Hydroxycamptothecin", "LC beads",
                        "Nedaplatin", "Pirarubicin", "Lipiodol", "PVA",
                        "KMG", "Absolute Alcohol", "NBCA", "Gelatin Sponge"
                    ]
                if text is not None and len(text) > 0:
                    # print(text)
                    chosen_text_line = random.choice(text)  
                    for kw in keywords:
                        if kw in chosen_text_line:
                            keyword.append(kw)
                            # break
                
                pre_img_sim, pre_mask_sim, post_img_sim, post_mask_sim, text_sim_line = self.get_similar(name, random.choice(text))

                with autocast(enabled=self.amp):
                    loss = self.model(
                        input_data,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                        text=text,
                        pre_img_sim=pre_img_sim,
                        pre_mask_sim=pre_mask_sim,
                        post_img_sim=post_img_sim,
                        post_mask_sim=post_mask_sim,
                        text_sim=text_sim_line,
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'Step {self.step}: Loss = {loss.item()}')

            log = {'loss': loss.item()}


            if self.max_grad_norm:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            self.writer.add_scalar('Train_Loss', loss.item(), self.step)
            self.writer.add_scalar('Learning_rate', self.opt.param_groups[0]['lr'], self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                self.save('best_model')

            if self.step % self.save_and_sample_every == 0:
                self.save(self.step // self.save_and_sample_every)

            log_fn(log)
            self.step += 1

        print('Training completed')

class Tester(object):
    def __init__(
        self,
        diffusion_model,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema_model = copy.deepcopy(self.model)
        self.step=0
        self.image_size = diffusion_model.image_size

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)

def get_bbox_from_mask(mask):
    """get the bbox coordinates of the mask"""
    # find the indices of the non-zero elements
    indices = torch.nonzero(mask)
    if indices.shape[0] == 0:
        return None
    
    # get the bbox coordinates
    min_d = indices[:, 2].min()
    max_d = indices[:, 2].max()
    min_h = indices[:, 3].min()
    max_h = indices[:, 3].max()
    min_w = indices[:, 4].min()
    max_w = indices[:, 4].max()
    
    return min_d, max_d, min_h, max_h, min_w, max_w

def create_bbox_mask(mask_shape, bbox_coords):
    """create bbox mask"""
    device = bbox_coords[0].device  # get device information from coordinates
    mask = torch.zeros(mask_shape, device=device)
    min_d, max_d, min_h, max_h, min_w, max_w = bbox_coords
    mask[:, :, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = 1
    return mask