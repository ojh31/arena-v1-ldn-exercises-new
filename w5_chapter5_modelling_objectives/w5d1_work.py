#%%
import torch as t
from typing import Union, Optional, Tuple
from torch import nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
import os
import sys
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataclasses import dataclass
import wandb

# Add to your path here, so you can import the appropriate functions
sys.path.append('/home/oskar/projects/arena-v1-ldn-exercises-new')
import w5d1_utils
import w5d1_tests
from w0d2_chapter0_convolutions.solutions import (
    pad1d, pad2d, conv1d_minimal, conv2d_minimal, Conv2d, Linear, ReLU
)
from w0d3_chapter0_resnets.solutions import BatchNorm2d
# %%
def conv_transpose1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''
    Like torch's conv_transpose1d using bias=False and 
    all other keyword arguments left at their default values.
    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)
    Returns: shape (batch, out_channels, output_width)
    '''
    in_channels, out_channels, kernel_width = weights.shape
    x_padded = pad1d(x, kernel_width - 1, kernel_width - 1, 0)
    weights_flip = weights.flip([-1])
    weights_flip = rearrange(weights_flip, 'i o k -> o i k')
    print('w', weights, weights_flip)
    return conv1d_minimal(x_padded, weights_flip)

w5d1_tests.test_conv_transpose1d_minimal(conv_transpose1d_minimal)
# %%
