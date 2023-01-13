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
        patch_and_class = t.cat((patch_flat, class_reshaped), dim=1)
        return patch_and_class + pos
        

#%%
if MAIN:
    w5d4_tests.test_vision_embeddings(CLIPVisionEmbeddings)
#%%
