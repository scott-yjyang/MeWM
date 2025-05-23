# import torch
# from torch import Tensor, nn

# import math
# from typing import Tuple, Type

# from .common import MLPBlock


# class TwoWayTransformer(nn.Module):
#     def __init__(
#         self,
#         args,
#         depth: int,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#     ) -> None:
#         """
#         A transformer decoder that attends to an input image using
#         queries whose positional embedding is supplied.

#         Args:
#           depth (int): number of layers in the transformer
#           embedding_dim (int): the channel dimension for the input embeddings
#           num_heads (int): the number of heads for multihead attention. Must
#             divide embedding_dim
#           mlp_dim (int): the channel dimension internal to the MLP block
#           activation (nn.Module): the activation to use in the MLP block
#         """
#         super().__init__()
#         self.args = args
#         self.depth = depth
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.mlp_dim = mlp_dim
#         self.layers = nn.ModuleList()

#         for i in range(depth):
#             self.layers.append(
#                 TwoWayAttentionBlock(
#                     embedding_dim=embedding_dim,
#                     num_heads=num_heads,
#                     mlp_dim=mlp_dim,
#                     activation=activation,
#                     attention_downsample_rate=attention_downsample_rate,
#                     skip_first_layer_pe=(i == 0),
#                 )
#             )

#         self.final_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm_final_attn = nn.LayerNorm(embedding_dim)

#     def forward(
#         self,
#         pre_ct: Tensor,    # pre-CT features
#         pe: Tensor,        # positional encoding
#         post_ct: Tensor,   # post-CT features
#     ) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#             pre_ct: pre-CT features (B, C, T, H, W)
#             pe: positional encoding
#             post_ct: post-CT features (B, C, T, H, W)
#         Returns:
#             Tuple[Tensor, Tensor]: pre2post features and post2pre features (B, T, C)
#         """
#         # 调整CT特征维度
#         if self.args.model_CT == 'resnetMC3_18' or self.args.model_CT == 'resnetMC3_18_wMask':
#             # (B, C, T, H, W) -> (B, T, C)
#             pre_ct = pre_ct.mean(dim=(3,4)).permute(0, 2, 1)
#             post_ct = post_ct.mean(dim=(3,4)).permute(0, 2, 1)
#         elif self.args.model_CT == 'medicalNet':
#             # (B, C, T, H, W) -> (B, T*H*W, C)
#             b, c, t, h, w = pre_ct.shape
#             pre_ct = pre_ct.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
#             post_ct = post_ct.permute(0, 2, 3, 4, 1).reshape(b, -1, c)

#         # 准备queries和keys
#         queries = pre_ct   # pre-CT作为queries
#         keys = post_ct     # post-CT作为keys

#         # 应用transformer层
#         for layer in self.layers:
#             queries, keys = layer(
#                 queries=queries,
#                 keys=keys,
#                 query_pe=pe[:,:queries.size(1)],  # 裁剪PE以匹配序列长度
#                 key_pe=pe[:,:keys.size(1)]
#             )

#         # 最终的注意力层
#         q = queries + pe[:,:queries.size(1)]
#         k = keys + pe[:,:keys.size(1)]
#         pre2post = self.final_attn_token_to_image(q=q, k=k, v=keys)
#         pre2post = pre2post + queries
#         pre2post = self.norm_final_attn(pre2post)

#         # 反向的注意力层
#         q = keys + pe[:,:keys.size(1)]
#         k = queries + pe[:,:queries.size(1)]
#         post2pre = self.final_attn_token_to_image(q=q, k=k, v=queries)
#         post2pre = post2pre + keys
#         post2pre = self.norm_final_attn(post2pre)

#         return pre2post, post2pre  # 返回双向融合特征


# class TwoWayTransformer2(nn.Module):
#     def __init__(
#         self,
#         args,
#         depth: int,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#     ) -> None:
#         """
#         A transformer decoder that attends to an input image using
#         queries whose positional embedding is supplied.

#         Args:
#           depth (int): number of layers in the transformer
#           embedding_dim (int): the channel dimension for the input embeddings
#           num_heads (int): the number of heads for multihead attention. Must
#             divide embedding_dim
#           mlp_dim (int): the channel dimension internal to the MLP block
#           activation (nn.Module): the activation to use in the MLP block
#         """
#         super().__init__()
#         self.args = args
#         self.depth = depth
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.mlp_dim = mlp_dim
#         self.layers = nn.ModuleList()

#         for i in range(depth):
#             self.layers.append(
#                 TwoWayAttentionBlock2(
#                     embedding_dim=embedding_dim,
#                     num_heads=num_heads,
#                     mlp_dim=mlp_dim,
#                     activation=activation,
#                     attention_downsample_rate=attention_downsample_rate,
#                     skip_first_layer_pe=(i == 0),
#                 )
#             )

#         self.final_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm_final_attn = nn.LayerNorm(embedding_dim)

#     def forward(
#         self,
#         image_embedding: Tensor,
#         image_pe: Tensor,
#         point_embedding: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#           image_embedding (torch.Tensor): image to attend to. Should be shape
#             B x embedding_dim x h x w for any h and w. 그런데 CtransPath의 결과가 B x p x embedding_dim 이기 때문에 이에 맞게 코드 수정.
#           image_pe (torch.Tensor): the positional encoding to add to the image. Must
#             have the same shape as image_embedding.
#           point_embedding (torch.Tensor): the embedding to add to the query points.
#             Must have shape B x N_points x embedding_dim for any N_points.

#         Returns:
#           torch.Tensor: the processed point_embedding
#           torch.Tensor: the processed image_embedding
#         """
        
#         if self.args.alignment_base == 'CT':
#             # point_embedding : CT
#             if len(point_embedding.shape) == 5:
#                 # bs, n, c, h, w = point_embedding.shape
#                 if self.args.model_CT == 'resnetMC3_18':
#                     # (1, 512, 160, 14, 14) --> (1, 512, 160)
#                     point_embedding = point_embedding.mean(dim=(3,4)).permute(0, 2, 1)
#                 elif self.args.model_CT == 'medicalNet':
#                     # (1, 512, 20, 64, 64) --> (1, 512, 20*64*64)
#                     point_embedding = point_embedding.flatten(2).permute(0, 2, 1)
#         else:
#             # image_embedding : CT
#             # BxCxHxW -> BxHWxC == B x N_image_tokens x C
#             if len(image_embedding.shape) == 5:
#                 # bs, n, c, h, w = image_embedding.shape
#                 if self.args.model_CT == 'resnetMC3_18':
#                     # (1, 512, 160, 14, 14) --> (1, 512, 160)
#                     image_embedding = image_embedding.mean(dim=(3,4)).permute(0, 2, 1)
#                 elif self.args.model_CT == 'medicalNet':
#                     # (1, 512, 20, 64, 64) --> (1, 512, 20*64*64)
#                     image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

#         # Prepare queries
#         queries = point_embedding
#         keys = image_embedding

#         # Apply transformer blocks and final layernorm
#         for layer in self.layers:
#             queries, keys = layer(
#                 queries=queries,
#                 keys=keys,
#                 query_pe=point_embedding,
#                 key_pe=image_pe,
#             )

#         # Apply the final attention layer from the points to the image
#         q = queries + point_embedding
#         k = keys + image_pe
#         attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
#         queries = queries + attn_out
#         queries = self.norm_final_attn(queries)

#         return queries, keys


# class TwoWayAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int = 2048,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#         skip_first_layer_pe: bool = False,
#     ) -> None:
#         """
#         A transformer block with four layers: (1) self-attention of sparse
#         inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
#         block on sparse inputs, and (4) cross attention of dense inputs to sparse
#         inputs.

#         Arguments:
#           embedding_dim (int): the channel dimension of the embeddings
#           num_heads (int): the number of heads in the attention layers
#           mlp_dim (int): the hidden dimension of the mlp block
#           activation (nn.Module): the activation of the mlp block
#           skip_first_layer_pe (bool): skip the PE on the first layer
#         """
#         super().__init__()
#         self.self_attn = Attention(embedding_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embedding_dim)

#         self.cross_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm2 = nn.LayerNorm(embedding_dim)

#         self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
#         self.norm3 = nn.LayerNorm(embedding_dim)

#         self.norm4 = nn.LayerNorm(embedding_dim)
#         self.cross_attn_image_to_token = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )

#         self.skip_first_layer_pe = skip_first_layer_pe

#     def forward(
#         self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
#     ) -> Tuple[Tensor, Tensor]:
#         # Self attention block
#         if self.skip_first_layer_pe:
#             queries = self.self_attn(q=queries, k=queries, v=queries)
#         else:
#             q = queries + query_pe
#             attn_out = self.self_attn(q=q, k=q, v=queries)
#             queries = queries + attn_out
#         queries = self.norm1(queries)

#         # Cross attention block, tokens attending to image embedding
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
#         queries = queries + attn_out
#         queries = self.norm2(queries)

#         # MLP block
#         mlp_out = self.mlp(queries)
#         queries = queries + mlp_out
#         queries = self.norm3(queries)

#         # Cross attention block, image embedding attending to tokens
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
#         keys = keys + attn_out
#         keys = self.norm4(keys)

#         return queries, keys


# class TwoWayAttentionBlock2(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int = 2048,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#         skip_first_layer_pe: bool = False,
#     ) -> None:
#         """
#         A transformer block with four layers: (1) self-attention of sparse
#         inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
#         block on sparse inputs, and (4) cross attention of dense inputs to sparse
#         inputs.

#         Arguments:
#           embedding_dim (int): the channel dimension of the embeddings
#           num_heads (int): the number of heads in the attention layers
#           mlp_dim (int): the hidden dimension of the mlp block
#           activation (nn.Module): the activation of the mlp block
#           skip_first_layer_pe (bool): skip the PE on the first layer
#         """
#         super().__init__()
#         self.self_attn_token = Attention(embedding_dim, num_heads)
#         self.self_attn_image = Attention(embedding_dim, num_heads)
#         self.norm1_token = nn.LayerNorm(embedding_dim)
#         self.norm1_image = nn.LayerNorm(embedding_dim)

#         self.cross_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm2 = nn.LayerNorm(embedding_dim)

#         self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
#         self.norm3 = nn.LayerNorm(embedding_dim)

#         self.norm4 = nn.LayerNorm(embedding_dim)
#         self.cross_attn_image_to_token = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )

#         self.skip_first_layer_pe = skip_first_layer_pe

#     def forward(
#         self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
#     ) -> Tuple[Tensor, Tensor]:
#         # Self attention block
#         if self.skip_first_layer_pe:
#             queries = self.self_attn_token(q=queries, k=queries, v=queries)
#             keys = self.self_attn_image(q=keys, k=keys, v=keys)
#         else:
#             q = queries + query_pe
#             attn_out_q = self.self_attn_token(q=q, k=q, v=queries)
#             queries = queries + attn_out_q
#             k = keys + key_pe
#             attn_out_k = self.self_attn_image(q=k, k=k, v=keys)
#             keys = keys + attn_out_k
#         queries = self.norm1_token(queries)
#         keys = self.norm1_image(keys)

#         # Cross attention block, tokens attending to image embedding
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
#         queries = queries + attn_out
#         queries = self.norm2(queries)

#         # MLP block
#         mlp_out = self.mlp(queries)
#         queries = queries + mlp_out
#         queries = self.norm3(queries)

#         # Cross attention block, image embedding attending to tokens
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
#         keys = keys + attn_out
#         keys = self.norm4(keys)

#         return queries, keys


# class Attention(nn.Module):
#     """
#     An attention layer that allows for downscaling the size of the embedding
#     after projection to queries, keys, and values.
#     """

#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         downsample_rate: int = 1,
#     ) -> None:
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.internal_dim = embedding_dim // downsample_rate
#         self.num_heads = num_heads
#         assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

#         self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

#     def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
#         b, n, c = x.shape
#         x = x.reshape(b, n, num_heads, c // num_heads)
#         return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

#     def _recombine_heads(self, x: Tensor) -> Tensor:
#         b, n_heads, n_tokens, c_per_head = x.shape
#         x = x.transpose(1, 2)
#         return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

#     def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
#         # Input projections
#         q = self.q_proj(q)
#         k = self.k_proj(k)
#         v = self.v_proj(v)

#         # Separate into heads
#         q = self._separate_heads(q, self.num_heads)
#         k = self._separate_heads(k, self.num_heads)
#         v = self._separate_heads(v, self.num_heads)

#         # Attention
#         _, _, _, c_per_head = q.shape
#         attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
#         attn = attn / math.sqrt(c_per_head)
#         attn = torch.softmax(attn, dim=-1)

#         # Get output
#         out = attn @ v
#         out = self._recombine_heads(out)
#         out = self.out_proj(out)

#         return out

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        args,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.args = args
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w. 그런데 CtransPath의 결과가 B x p x embedding_dim 이기 때문에 이에 맞게 코드 수정.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        
        # image_embedding : CT
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        if len(image_embedding.shape) == 5:
            # bs, n, c, h, w = image_embedding.shape
            if self.args.model_CT == 'resnetMC3_18' or self.args.model_CT == 'resnetMC3_18_wMask':

                # (1, 512, 160, 14, 14) --> (1, 512, 160)
                image_embedding = image_embedding.mean(dim=(3,4)).permute(0, 2, 1)
            elif self.args.model_CT == 'medicalNet':
                # (1, 512, 20, 64, 64) --> (1, 512, 20*64*64)
                image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayTransformer2(nn.Module):
    def __init__(
        self,
        args,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.args = args
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock2(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w. 그런데 CtransPath의 결과가 B x p x embedding_dim 이기 때문에 이에 맞게 코드 수정.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        
        if self.args.alignment_base == 'CT':
            # point_embedding : CT
            if len(point_embedding.shape) == 5:
                # bs, n, c, h, w = point_embedding.shape
                if self.args.model_CT == 'resnetMC3_18':
                    # (1, 512, 160, 14, 14) --> (1, 512, 160)
                    point_embedding = point_embedding.mean(dim=(3,4)).permute(0, 2, 1)
                elif self.args.model_CT == 'medicalNet':
                    # (1, 512, 20, 64, 64) --> (1, 512, 20*64*64)
                    point_embedding = point_embedding.flatten(2).permute(0, 2, 1)
        else:
            # image_embedding : CT
            # BxCxHxW -> BxHWxC == B x N_image_tokens x C
            if len(image_embedding.shape) == 5:
                # bs, n, c, h, w = image_embedding.shape
                if self.args.model_CT == 'resnetMC3_18':
                    # (1, 512, 160, 14, 14) --> (1, 512, 160)
                    image_embedding = image_embedding.mean(dim=(3,4)).permute(0, 2, 1)
                elif self.args.model_CT == 'medicalNet':
                    # (1, 512, 20, 64, 64) --> (1, 512, 20*64*64)
                    image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayAttentionBlock2(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn_token = Attention(embedding_dim, num_heads)
        self.self_attn_image = Attention(embedding_dim, num_heads)
        self.norm1_token = nn.LayerNorm(embedding_dim)
        self.norm1_image = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn_token(q=queries, k=queries, v=queries)
            keys = self.self_attn_image(q=keys, k=keys, v=keys)
        else:
            q = queries + query_pe
            attn_out_q = self.self_attn_token(q=q, k=q, v=queries)
            queries = queries + attn_out_q
            k = keys + key_pe
            attn_out_k = self.self_attn_image(q=k, k=k, v=keys)
            keys = keys + attn_out_k
        queries = self.norm1_token(queries)
        keys = self.norm1_image(keys)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out