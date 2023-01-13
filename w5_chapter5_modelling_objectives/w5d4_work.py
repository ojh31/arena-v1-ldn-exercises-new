#%%
import glob
import os
import sys
from typing import Callable, Union, cast
import pandas as pd
import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers.models.clip import modeling_clip
import sentence_transformers # You might need to pip install this
import w5d4_tests
from w5d4_utils import (
    CLIPConfig,
    CLIPOutput,
    CLIPTextConfig,
    CLIPVisionConfig,
    get_reference_model,
    get_reference_clip_model,
)
import importlib

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
def print_class_attrs(cls: type) -> None:
    print(f"\n\n{cls.__name__}\n---")
    for (k, v) in ((k, v) for (k, v) in vars(cls).items() if k[0] != "_"):
        print(f"{k}: {v}")


if MAIN:
    print_class_attrs(CLIPVisionConfig)
    print_class_attrs(CLIPTextConfig)
    print_class_attrs(CLIPConfig)
# %%
class CLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    patch_size: int
    image_size: int
    embed_dim: int
    num_patches: int
    class_embedding: nn.Parameter
    patch_embedding: nn.Conv2d
    position_embedding: nn.Embedding
    position_ids: t.Tensor

    def __init__(self, config: CLIPVisionConfig):
        '''Assign values from input config to class member variables as appropriate,
        e.g. self.patch_size = config.patch_size'''
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.embed_dim = config.hidden_size
        assert self.image_size % self.patch_size == 0
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.class_embedding = nn.Parameter(
            t.randn((self.embed_dim, ))
        )
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        position_ids = t.arange(0, self.num_patches + 1, dtype=t.long).unsqueeze(0)
        self.register_buffer(
            'position_ids',
            position_ids
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=position_ids.shape[-1],
            embedding_dim=self.embed_dim,
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the patch embeddings and the positional embeddings and return their sum.

        x: shape (batch, channels=3, height=224, width=224)
        out: shape (batch, sequence=257, hidden=1024)
        '''
        b, c, h, w = x.shape
        patch = self.patch_embedding(x)
        pos = self.position_embedding(self.position_ids)
        patch_flat = rearrange(patch, 'b e h w -> b (h w) e')
        class_reshaped = repeat(self.class_embedding, 'e -> b s e', b=b, s=1)
        patch_and_class = t.cat((class_reshaped, patch_flat), dim=1)
        return patch_and_class + pos
        

#%%
if MAIN:
    w5d4_tests.test_vision_embeddings(CLIPVisionEmbeddings)
#%%
def gelu_sigmoid_approximation(x: t.Tensor) -> t.Tensor:
    '''Return sigmoid approximation of GELU of input tensor x with same shape.'''
    return x * t.sigmoid(1.702 * x)


def plot_gelu_approximation(x: t.Tensor):
    (fig, (ax0, ax1)) = plt.subplots(nrows=2, figsize=(12, 12))
    actual = F.gelu(x)
    approx = gelu_sigmoid_approximation(x)
    diff = (actual - approx).abs()
    x_cpu = x.cpu()
    ax0.plot(x_cpu, diff.cpu(), label="absolute error")
    ax0.legend()
    ax1.plot(x_cpu, actual.cpu(), label="exact", alpha=0.5)
    ax1.plot(x_cpu, approx.cpu(), label="sigmoid", alpha=0.5)
    ax1.legend()
    ax1.set(xlabel=f"x ({x.dtype})")


if MAIN:
    x = t.linspace(-5, 5, 400)
    plot_gelu_approximation(x)
    if t.cuda.is_available():
        x16 = t.linspace(-5, 5, 400, dtype=t.float16, device=device)
        plot_gelu_approximation(x16)

#%%
class CLIPMLP(nn.Module):
    fc1: nn.Linear
    fc2: nn.Linear

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        '''Initialize parent class, then assign fully-connected layers based
        on shape in input config'''
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Run forward pass of MLP, including fully-connected layers and non-linear
        activations where appropriate'''
        x = self.fc1(x)
        x = gelu_sigmoid_approximation(x)
        x = self.fc2(x)
        return x


if MAIN:
    w5d4_tests.test_mlp(CLIPMLP)
# %%
class CLIPAttention(nn.Module):
    num_heads: int
    head_size: int
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        '''Assign values from input config to class member variables as appropriate'''
        super().__init__()
        self.head_size = config.hidden_size / config.num_attention_heads
        self.num_heads = config.num_attention_heads
        assert self.head_size % self.num_heads == 0
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        '''
        Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and 
        a key at sequence position k.
        '''
        Q = self.q_proj(x)
        K = self.k_proj(x)

        new_Q = rearrange(
            Q, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=self.num_heads
        )
        new_K = rearrange(
            K, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=self.num_heads
        )

        einsum_eq = (
            'batches nheads seq_Q head_size, '
            'batches nheads seq_K head_size -> '
            'batches nheads seq_Q seq_K'
        )
        attention_scores = einsum(einsum_eq, new_Q, new_K)
        attention_scores /= (self.head_size ** 0.5)
        return attention_scores

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Perform forward pass through attention layer, computing attention pattern and 
        value projections to combine into output. 
        Remember to apply dropout.
        '''
        attention_scores = self.attention_pattern_pre_softmax(x)
        attention_probabilities = nn.functional.softmax(
            attention_scores, dim=-1
        )
        dropped_probabilities = self.dropout(attention_probabilities)
        V = self.v_proj(x)
        new_V = rearrange(
            V, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=self.num_heads
        )
        attention_values = einsum(
            'batches nheads seq_Q seq_K, batches nheads seq_K head_size -> '
            'batches seq_Q nheads head_size', 
            dropped_probabilities, 
            new_V
        )
        attention_rearranged = rearrange(
            attention_values, 
            'batches seq_Q nheads head_size -> batches seq_Q (nheads head_size)'
        )
        attention_times_o = self.out_proj(attention_rearranged)
        return attention_times_o


if MAIN:
    w5d4_tests.test_vision_attention(CLIPAttention)
# %%
class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x
# %%
class CLIPEncoder(nn.Module):
    layers: nn.ModuleList # [CLIPEncoderLayer]

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: t.Tensor) -> t.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
# %%
class CLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    embeddings: CLIPVisionEmbeddings
    pre_layrnorm: nn.LayerNorm
    encoder: CLIPEncoder
    post_layernorm: nn.LayerNorm

    def __init__(self, config: CLIPVisionConfig):
        '''
        Assign values from input config to class member variables as appropriate
        '''
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Perform forward pass through vision transformer: 
            embedding, layer norm, encoder, layer norm

        Return output corresponding to prepended class_embedding
        '''
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        x = self.encoder(x)
        x = x[:, 0, :] # output is taken from "begin token"
        x = self.post_layernorm(x)
        return x


if MAIN:
    importlib.reload(w5d4_tests)
    w5d4_tests.test_vision_transformer(CLIPVisionTransformer)
# %%
