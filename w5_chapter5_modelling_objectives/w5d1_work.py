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
import numpy as np
import time

# Add to your path here, so you can import the appropriate functions
sys.path.append('/home/oskar/projects/arena-v1-ldn-exercises-new')
import w5d1_utils
import w5d1_tests
from w0d2_chapter0_convolutions.solutions import (
    pad1d, pad2d, conv1d_minimal, conv2d_minimal, Conv2d, Linear, ReLU
)
from w0d3_chapter0_resnets.solutions import BatchNorm2d
from w5_chapter5_modelling_objectives.w5d1_solutions import celeb_DCGAN
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
    return conv1d_minimal(x_padded, weights_flip)

w5d1_tests.test_conv_transpose1d_minimal(conv_transpose1d_minimal)
# %%
def fractional_stride_1d(x, stride: int = 1):
    '''
    Returns a version of x suitable for transposed convolutions, 
    i.e. "spaced out" with zeros between its values.
    This spacing only happens along the last dimension.

    x: shape (batch, in_channels, width)
    Example: 
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    '''
    batch, in_channels, width = x.shape
    x_rep = repeat(x, 'b i w -> b i (w stride)', stride=stride)
    width_idx = repeat(t.arange(width * stride), 'w -> b i w', b=batch, i=in_channels)
    x_spaced = x_rep.where(width_idx % stride == 0, t.tensor([0], device=x.device))
    x_trim = x_spaced[..., :1-stride] if stride > 1 else x_spaced
    return x_trim


w5d1_tests.test_fractional_stride_1d(fractional_stride_1d)

#%%
def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''
    Like torch's conv_transpose1d using bias=False and 
    all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)
    Returns: shape (batch, out_channels, output_width)
    '''
    in_channels, out_channels, kernel_width = weights.shape
    x_spaced = fractional_stride_1d(x, stride=stride)
    pad_amt = kernel_width - 1 - padding
    x_padded = pad1d(x_spaced, pad_amt, pad_amt, 0)
    weights_flip = weights.flip([-1])
    weights_flip = rearrange(weights_flip, 'i o k -> o i k')
    return conv1d_minimal(x_padded, weights_flip)

w5d1_tests.test_conv_transpose1d(conv_transpose1d)
# %%
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)
#%%
def fractional_stride_2d(x, stride_h: int, stride_w: int):
    '''
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    '''
    device = x.device
    batch, in_channels, height, width = x.shape
    x_rep = repeat(x, 'b i h w-> b i (h stride_h) (w stride_w)', stride_w=stride_w, stride_h=stride_h)
    width_idx = repeat(
        t.arange(width * stride_w, device=device), 
        'w -> b i h w', 
        b=batch, i=in_channels, h=height * stride_h
    )
    height_idx = repeat(
        t.arange(height * stride_h, device=device), 
        'h -> b i h w', b=batch, i=in_channels, w=width * stride_w
    )
    x_spaced = x_rep.where(
        (width_idx % stride_w == 0) & (height_idx % stride_h == 0), 
        t.tensor([0], device=device)
    )
    if stride_w > 1:
        x_spaced = x_spaced[:, :, :, : 1 - stride_w]
    if stride_h > 1:
        x_spaced = x_spaced[:, :, : 1 - stride_h, :]
    return x_spaced


def test_fractional_stride_2d(fractional_stride_2d):
    x = t.tensor([[[[1, 2, 3], [4, 5, 6]]]])
    
    actual = fractional_stride_2d(x, stride_h=1, stride_w=1)
    expected = x
    t.testing.assert_close(actual, expected)

    actual = fractional_stride_2d(x, stride_h=2, stride_w=2)
    expected = t.tensor([[[[1, 0, 2, 0, 3], [0, 0, 0, 0, 0], [4, 0, 5, 0, 6]]]])
    print(actual.shape, expected.shape)
    t.testing.assert_close(actual, expected)

    print("All tests in `test_fractional_stride_2d` passed!")

test_fractional_stride_2d(fractional_stride_2d=fractional_stride_2d)

#%%

def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''
    Like torch's conv_transpose2d using bias=False
    x: shape (batch, in_channels, height, width)
    weights: shape (in_channels, out_channels, kernel_height, kernel_width)
    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    in_channels, out_channels, kernel_height, kernel_width = weights.shape
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    x_spaced = fractional_stride_2d(x, stride_h=stride_h, stride_w=stride_w)
    pad_x = kernel_width - 1 - padding_w
    pad_y = kernel_height - 1 - padding_h
    x_padded = pad2d(x_spaced, pad_x, pad_x, pad_y, pad_y, 0)
    weights_flip = weights.flip([-2, -1])
    weights_flip = rearrange(weights_flip, 'i o kh kw -> o i kh kw')
    return conv2d_minimal(x_padded, weights_flip)

w5d1_tests.test_conv_transpose2d(conv_transpose2d)
# %%
class ConvTranspose2d(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, 
        stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.ConvTranspose2d with bias=False.
        Name your weight field `self.weight` for compatibility with the tests.
        '''
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        super().__init__()
        kernel_height, kernel_width = force_pair(kernel_size)
        max_weight = 1 / np.sqrt(out_channels * kernel_height * kernel_width)
        self.weight = nn.Parameter(
            t.rand((in_channels, out_channels, kernel_height, kernel_width)) * 
            2 * max_weight - max_weight
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        return conv_transpose2d(x, self.weight, self.stride, self.padding)

w5d1_tests.test_ConvTranspose2d(ConvTranspose2d)
#%%
class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return (t.exp(2 * x) - 1) / (t.exp(2 * x) + 1)


class LeakyReLU(nn.Module):

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.where(x > 0, x * self.negative_slope)

    def extra_repr(self) -> str:
        return f'LeakyReLU({self.negative_slope:.02f})'

class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return 1.0 / (1.0 + t.exp(-x))

w5d1_tests.test_Tanh(Tanh)
w5d1_tests.test_LeakyReLU(LeakyReLU)
w5d1_tests.test_Sigmoid(Sigmoid)
# %%
def initialize_weights(model: nn.Module) -> None:
    is_batch = 'running_mean' in dir(model)
    for name, param in model.named_parameters():
        is_bias = 'bias' in name.lower()
        if is_batch and not is_bias:
            nn.init.normal_(param, mean=1, std=np.sqrt(.02))
        elif not is_batch:
            nn.init.normal_(param, mean=0, std=np.sqrt(.02))
#%%
dcgan_config = dict(
    latent_dim_size=100,
    img_size=64,
    img_channels=3,
    generator_num_features=1024,
    n_layers=4,
)
# %%
class Generator(nn.Module):
    '''
    Use ReLU activation in generator for all layers except for the output, which uses Tanh.
    Use batchnorm in both the generator and the discriminator.
    not applying batchnorm to the generator output layer (i.e. after the conv blocks)
    4 BatchNorm layers in the generator
    '''

    def __init__(
        self,
        latent_dim_size: int,           # size of the random vector we use for generating outputs, e.g. 100
        img_size: int,                  # size of the images we're generating, e.g. 64x64
        img_channels: int,              # indicates RGB images, e.g. 3
        generator_num_features: int,    # number of channels after first projection and reshaping, e.g. 1024
        n_layers: int,                  # number of CONV_n layers, e.g. 4
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers
        assert img_size >= 2 ** n_layers
        self.smallest_size = img_size // (2 ** n_layers) # e.g. 64 / 2^4 = 4
        self.gen_dim_size = generator_num_features * (self.smallest_size ** 2)
        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim_size, self.gen_dim_size, bias=False),
            Rearrange('b (c h w) -> b c h w', c=self.generator_num_features, h=self.smallest_size),
            BatchNorm2d(self.generator_num_features),
            ReLU(),
        )
        blocks = []
        for i in range(self.n_layers):
            block = []
            is_output = i + 1 == self.n_layers
            in_channels = generator_num_features // (2 ** i)
            if is_output:
                out_channels = self.img_channels
            else:
                out_channels = generator_num_features // (2 ** (i + 1))
            conv = ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            block.append(conv)
            if is_output:
                block.append(Tanh())
            else:
                block.append(BatchNorm2d(out_channels))
                block.append(ReLU())
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: t.Tensor):
        '''
        x: 100d latent vector, sampled from N(0, I_100)
        -> 4 * 4 * 1024 linear layer
        -> 4 transposed convolutions k=4, s=2, p=1
        '''
        x = self.projection(x)
        x = self.blocks(x)
        return x


class Discriminator(nn.Module):
    '''
    Use LeakyReLU activation in the discriminator for all layers.
    Use batchnorm in both the generator and the discriminator.
    not applying batchnorm to the generator output layer and the discriminator input layer
    '''

    def __init__(
        self,
        img_size: int,
        img_channels: int,
        generator_num_features: int,
        n_layers: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers
        assert img_size >= 2 ** n_layers
        self.smallest_size = img_size // (2 ** n_layers) # e.g. 64 / 2^4 = 4
        self.gen_dim_size = generator_num_features * (self.smallest_size ** 2)
        blocks = []
        for i in range(self.n_layers):
            block = []
            is_input = i == 0
            to_go = self.n_layers - i
            if is_input:
                in_channels = img_channels
            else:
                in_channels = generator_num_features // (2 ** (to_go))
            out_channels = generator_num_features // (2 ** (to_go - 1))
            conv = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            block.append(conv)
            if not is_input:
                block.append(BatchNorm2d(out_channels))
                block.append(LeakyReLU())
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.gen_dim_size, 1, bias=False)
        self.sigmoid = Sigmoid()


    def forward(self, x: t.Tensor):
        '''
        x is an image and we have to decide real/fake
        '''
        x = self.blocks(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self, latent_dim_size: int, img_size: int, img_channels: int, generator_num_features: int, 
        n_layers: int, ) -> None:
        super().__init__()
        self.netG = Generator(
            latent_dim_size=latent_dim_size, img_size=img_size, img_channels=img_channels, 
            generator_num_features=generator_num_features, n_layers=n_layers,
        )
        self.netD = Discriminator(
            img_size=img_size, 
            img_channels=img_channels, 
            generator_num_features=generator_num_features,
            n_layers=n_layers,
        )
        initialize_weights(self.netG)
        initialize_weights(self.netD)

#%%
my_DCGAN = DCGAN(**dcgan_config)
w5d1_utils.print_param_count(my_DCGAN.netG, celeb_DCGAN.netG)
#%%
my_DCGAN = DCGAN(**dcgan_config)
w5d1_utils.print_param_count(my_DCGAN.netD, celeb_DCGAN.netD)
# %%
from torchvision import transforms, datasets

image_size = dcgan_config['img_size']
transform = transforms.Compose([
    transforms.Resize((image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = ImageFolder(
    root="/home/oskar/projects/arena-v1-ldn-exercises-new/w5_chapter5_modelling_objectives/celebA",
    transform=transform
)

w5d1_utils.show_images(trainset, rows=3, cols=5)
# %%
@dataclass
class DCGANargs():
    latent_dim_size: int
    img_size: int
    img_channels: int
    generator_num_features: int
    n_layers: int
    trainset: datasets.ImageFolder
    lr: float
    betas: Tuple[float]
    batch_size: int = 8
    epochs: int = 1
    track: bool = True
    cuda: bool = True
    seconds_between_image_logs: int = 30
    scale_factor: int = 8
    max_examples: int = 5_000

#%%

def train_DCGAN(args: DCGANargs) -> DCGAN:
    device = 'cuda' if args.cuda and t.cuda.is_available() else 'cpu'
    batch_size = args.batch_size // args.scale_factor
    lr = args.lr / np.sqrt(args.scale_factor)
    train_loader = DataLoader(args.trainset, batch_size=batch_size)
    net = DCGAN(
        latent_dim_size=args.latent_dim_size,
        img_size=args.img_size,
        img_channels=args.img_channels,
        generator_num_features=args.generator_num_features,
        n_layers=args.n_layers
    ).train().to(device=device)
    optD = t.optim.Adam(net.netD.parameters(), lr=lr, maximize=True)
    optG = t.optim.Adam(net.netG.parameters(), lr=lr, maximize=True)
    last_image_log = time.time()
    image_latent_vector = t.randn((batch_size, args.latent_dim_size), device=device)
    n_examples_seen = 0
    if args.track:
        wandb.init(
            project='w5d1_GAN',
        )
        wandb.watch(net)
    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch + 1}')
        bar = tqdm(
            train_loader, total=min(args.max_examples // batch_size, len(train_loader))
        )
        for x, y in bar:
            x = x.to(device=device)
            y = y.to(device=device)
            z = t.randn((batch_size, args.latent_dim_size), device=device)
            fakes = net.netG(z)
            assert (y == 0).all() # these are all real images so same class

            # Discriminator train step
            optD.zero_grad()
            true_negative_reward = t.log(1 - net.netD(fakes.detach())).mean()
            true_positive_reward = t.log(net.netD(x)).mean()
            rewardD = true_negative_reward + true_positive_reward
            rewardD.backward()
            optD.step()

            # Generator train step
            optG.zero_grad()
            rewardG = t.log(net.netD(fakes)).mean()
            rewardG.backward()
            optG.step()

            long_since_image_log = (
                time.time() - last_image_log > args.seconds_between_image_logs
            )
            if args.track and long_since_image_log:
                wandb.log(dict(rewardD=rewardD, rewardG=rewardG), step=n_examples_seen)
                net.netG.train()
                with t.inference_mode():
                    new_image_arr = net.netG(image_latent_vector)
                net.netG.eval()
                arrays = rearrange(
                    new_image_arr, "b c h w -> b h w c"
                ).detach().cpu().numpy()
                images = [wandb.Image(arr) for arr in arrays]
                wandb.log({"images": images}, step=n_examples_seen)
                last_image_log = time.time()
            n_examples_seen += x.shape[0]
            if n_examples_seen > args.max_examples:
                break
    return net
#%%
paper_args = DCGANargs(
    latent_dim_size=100,
    img_size=64,
    img_channels=3,
    generator_num_features=1024,
    n_layers=4,
    lr=0.0002,
    betas=(0.5, 0.999),
    trainset=trainset,
    batch_size=128,
    epochs=1,
    track=True,
    cuda=True,
    scale_factor=32,
    max_examples=50_000,
    seconds_between_image_logs=30,
)

#%%
model = train_DCGAN(paper_args)
#%%