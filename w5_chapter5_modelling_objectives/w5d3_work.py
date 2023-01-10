#%%
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
import torch as t
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import DataLoader
import wandb
from torchvision import transforms
import torchinfo
from torch import nn
import plotly.express as px
from einops.layers.torch import Rearrange
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torchvision import datasets
from pathlib import Path
from fancy_einsum import einsum
import numpy as np

MAIN = __name__ == "__main__"

device = "cuda" if t.cuda.is_available() else "cpu"

import sys, os
# CHANGE THIS TO YOUR PATH, TO IMPORT FROM WEEK 0
sys.path.append(r"/home/oskar/projects/arena-v1-ldn-exercises-new/")

from w0d2_chapter0_convolutions.solutions import Linear, conv2d, force_pair, IntOrPair
from w0d3_chapter0_resnets.solutions import Sequential
from w1d1_chapter1_transformer_reading.solutions import GELU, PositionalEncoding
from w5d1_solutions import ConvTranspose2d
import w5d3_tests
# %%
def gradient_images(n_images: int, img_size: tuple[int, int, int]) -> t.Tensor:
    '''Generate n_images of img_size, each a color gradient
    '''
    (C, H, W) = img_size
    corners = t.randint(0, 255, (2, n_images, C))
    xs = t.linspace(0, W / (W + H), W)
    ys = t.linspace(0, H / (W + H), H)
    (x, y) = t.meshgrid(xs, ys, indexing="xy")
    grid = x + y
    grid = grid / grid[-1, -1]
    grid = repeat(grid, "h w -> b c h w", b=n_images, c=C)
    base = repeat(corners[0], "n c -> n c h w", h=H, w=W)
    ranges = repeat(corners[1] - corners[0], "n c -> n c h w", h=H, w=W)
    gradients = base + grid * ranges
    assert gradients.shape == (n_images, C, H, W)
    return gradients / 255

def plot_img(img: t.Tensor, title: Optional[str] = None) -> None:
    '''Plots a single image, with optional title.
    '''
    img = rearrange(img, "c h w -> h w c").clip(0, 1)
    img = (255 * img).to(t.uint8)
    fig = px.imshow(img, title=title)
    fig.update_layout(margin=dict(t=70 if title else 40, l=40, r=40, b=40))
    fig.show()

def plot_img_grid(imgs: t.Tensor, title: Optional[str] = None, cols: Optional[int] = None) -> None:
    '''Plots a grid of images, with optional title.
    '''
    b = imgs.shape[0]
    imgs = (255 * imgs).to(t.uint8).squeeze()
    if imgs.ndim == 3:
        imgs = repeat(imgs, "b h w -> b 3 h w")
    imgs = rearrange(imgs, "b c h w -> b h w c")
    if cols is None: cols = int(b**0.5) + 1
    fig = px.imshow(imgs, facet_col=0, facet_col_wrap=cols, title=title)
    for annotation in fig.layout.annotations: annotation["text"] = ""
    fig.show()

def plot_img_slideshow(imgs: t.Tensor, title: Optional[str] = None) -> None:
    '''Plots slideshow of images.
    '''
    imgs = (255 * imgs).to(t.uint8).squeeze()
    if imgs.ndim == 3:
        imgs = repeat(imgs, "b h w -> b 3 h w")
    imgs = rearrange(imgs, "b c h w -> b h w c")
    fig = px.imshow(imgs, animation_frame=0, title=title)
    fig.show()

if MAIN:
    print("A few samples from the input distribution: ")
    image_shape = (3, 16, 16)
    n_images = 5
    imgs = gradient_images(n_images, image_shape)
    for i in range(n_images):
        plot_img(imgs[i])
# %%
def normalize_img(img: t.Tensor) -> t.Tensor:
    return img * 2 - 1

def denormalize_img(img: t.Tensor) -> t.Tensor:
    return ((img + 1) / 2).clamp(0, 1)

if MAIN:
    plot_img(imgs[0], "Original")
    plot_img(normalize_img(imgs[0]), "Normalized")
    plot_img(denormalize_img(normalize_img(imgs[0])), "Denormalized")
# %%
def linear_schedule(
    max_steps: int, min_noise: float = 0.0001, max_noise: float = 0.02
) -> t.Tensor:
    '''
    Return the forward process variances as in the paper.

    max_steps: total number of steps of noise addition
    out: shape (step=max_steps, ) the amount of noise at each step
    '''
    return t.linspace(min_noise, max_noise, max_steps)

if MAIN:
    betas = linear_schedule(max_steps=200)
# %%
def q_forward_slow(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    '''
    Return the input image with num_steps iterations of noise added according to schedule.
    x: shape (channels, height, width)
    betas: shape (T, ) with T >= num_steps

    out: shape (channels, height, width)
    '''
    for step in range(num_steps):
        beta = betas[step]
        x = np.sqrt(1 - beta) * x + np.sqrt(beta) * t.randn_like(x)
    return x

if MAIN:
    x0s = normalize_img(gradient_images(1, (3, 16, 16))[0])
    for n in [1, 10, 50, 200]:
        xt = q_forward_slow(x0s, n, betas)
        plot_img(denormalize_img(xt), f"Equation 2 after {n} step(s)")
    plot_img(denormalize_img(t.randn_like(xt)), "Random Gaussian noise")
# %%
def q_forward_fast(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    '''Equivalent to Equation 2 but without a for loop.'''
    alphas = 1 - betas[:num_steps]
    alpha_bar = alphas.prod()
    return np.sqrt(alpha_bar) * x + np.sqrt(1 - alpha_bar) * t.randn_like(x)

if MAIN:
    for n in [1, 10, 50, 200]:
        xt = q_forward_fast(x0s, n, betas)
        plot_img(denormalize_img(xt), f"Equation 4 after {n} steps")
# %%
class NoiseSchedule(nn.Module):
    betas: t.Tensor
    alphas: t.Tensor
    alpha_bars: t.Tensor

    def __init__(self, max_steps: int, device: Union[t.device, str]) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.device = device
        self.register_buffer('betas', linear_schedule(max_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bars', self.alphas.cumprod(dim=0))
        self.to(device)
        
    @t.inference_mode()
    def beta(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the beta(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        return self.betas[num_steps]

    @t.inference_mode()
    def alpha(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the alphas(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        return self.alphas[num_steps]

    @t.inference_mode()
    def alpha_bar(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the alpha_bar(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        return self.alpha_bars[num_steps]

    def alpha_bar_like(
        self, img: t.Tensor, num_steps: Union[int, t.Tensor],
    ) -> t.Tensor:
        assert isinstance(img, t.Tensor)
        c, h, w = img.shape[-3:]
        return repeat(
            self.alpha_bar(num_steps),
            'b -> b c h w',
            c=c,
            h=h,
            w=w,
        )

    def __len__(self) -> int:
        return self.max_steps

    def extra_repr(self) -> str:
        return f"max_steps={self.max_steps}"
# %%
def noise_img(
    img: t.Tensor, noise_schedule: NoiseSchedule, 
    max_steps: Optional[int] = None
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    '''
    Adds a uniform random number of steps of noise to each image in img.

    img: An image tensor of shape (B, C, H, W)
    noise_schedule: The NoiseSchedule to follow
    max_steps: if provided, only perform the first max_steps of the schedule

    Returns a tuple composed of:
    num_steps: an int tensor of shape (B,) of the number of steps of noise added to each image
    noise: the unscaled, standard Gaussian noise added to each image, a tensor of shape (B, C, H, W)
    noised: the final noised image, a tensor of shape (B, C, H, W)
    '''
    assert isinstance(img, t.Tensor)
    num_steps = t.randint_like(
        img[:, 0, 0, 0], 0, noise_schedule.max_steps, dtype=t.long
    )
    if max_steps is not None:
        num_steps = t.min(num_steps, t.ones_like(num_steps) * max_steps)
    noise = t.randn_like(img)
    alpha_bar = noise_schedule.alpha_bar_like(img, num_steps)
    noised = (
        t.sqrt(alpha_bar) * img + 
        t.sqrt(1 - alpha_bar) * noise
    )
    assert noise.shape == img.shape
    assert noised.shape == img.shape
    return num_steps, noise, noised

if MAIN:
    noise_schedule = NoiseSchedule(max_steps=200, device="cpu")
    img = gradient_images(1, (3, 16, 16))
    (num_steps, noise, noised) = noise_img(normalize_img(img), noise_schedule, max_steps=10)
    plot_img(img[0], "Gradient")
    plot_img(noise[0], "Applied Unscaled Noise")
    plot_img(denormalize_img(noised[0]), "Gradient with Noise Applied")
# %%
def reconstruct(
    noisy_img: t.Tensor, noise: t.Tensor, num_steps: t.Tensor, noise_schedule: NoiseSchedule
) -> t.Tensor:
    '''
    Subtract the scaled noise from noisy_img to recover the original image. 
    We'll later use this with the model's output to log reconstructions during training. 
    We'll use a different method to sample images once the model is trained.

    Returns img, a tensor with shape (B, C, H, W)
    '''
    alpha_bar = noise_schedule.alpha_bar_like(noisy_img, num_steps)
    return (
        noisy_img - t.sqrt(1 - alpha_bar) * noise
    ) / t.sqrt(alpha_bar)

if MAIN:
    reconstructed = reconstruct(noised, noise, num_steps, noise_schedule)
    denorm = denormalize_img(reconstructed)
    plot_img(img[0], "Original Gradient")
    plot_img(denorm[0], "Reconstruction")
    t.testing.assert_close(denorm, img)
# %%
@dataclass
class DiffusionArgs:
    project: str
    lr: float = 0.001
    image_shape: tuple = (3, 4, 5)
    hidden_size: int = 128
    epochs: int = 10
    max_steps: int = 100
    batch_size: int = 128
    seconds_between_image_logs: int = 10
    n_images_per_log: int = 3
    n_images: int = 50000
    n_eval_images: int = 1000
    cuda: bool = True
    track: bool = True
    seed: int = 0
    

class DiffusionModel(nn.Module, ABC):
    image_shape: tuple[int, ...]
    noise_schedule: Optional[NoiseSchedule]
    max_steps: int

    @abstractmethod
    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        ...

@dataclass(frozen=True)
class TinyDiffuserConfig:
    image_shape: Tuple[int, ...] = (3, 4, 5)
    hidden_size: int = 128
    max_steps: int = 100

class TinyDiffuser(DiffusionModel):
    def __init__(self, config: TinyDiffuserConfig):
        '''
        A toy diffusion model composed of an MLP (Linear, ReLU, Linear)
        '''
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_shape = config.image_shape
        self.noise_schedule = None
        self.max_steps = config.max_steps
        self.flatten_dim = np.prod(self.image_shape)
        c, h, w = self.image_shape
        self.flattener = Rearrange('b c h w -> b (c h w)')
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim + 1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.flatten_dim),
            Rearrange('b (c h w) -> b c h w', c=c, h=h, w=w)
        )

    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        '''
        Given a batch of images and noise steps applied, attempt to 
        predict the noise that was applied.
        images: tensor of shape (B, C, H, W)
        num_steps: tensor of shape (B,)

        Returns
        noise_pred: tensor of shape (B, C, H, W)
        '''
        flat = self.flattener(images)
        time = (num_steps / self.max_steps).unsqueeze(1)
        flat = t.cat((flat.T, time.T)).T
        return self.mlp(flat)
        

if MAIN:
    image_shape = (3, 4, 5)
    n_images = 5
    imgs = gradient_images(n_images, image_shape)
    n_steps = t.zeros(imgs.size(0))
    model_config = TinyDiffuserConfig(image_shape, hidden_size=16)
    model = TinyDiffuser(model_config)
    out = model(imgs, n_steps)
    plot_img(out[0].detach(), "Noise prediction of untrained model")
#%%
def train_tiny_diffuser(
    model: DiffusionModel, 
    args: DiffusionArgs, 
    trainset: TensorDataset,
    testset: Optional[TensorDataset] = None
) -> DiffusionModel:
    t.manual_seed(args.seed)
    device = 'cuda' if args.cuda and t.cuda.is_available() else 'cpu'
    model = model.to(device=device)
    batch_size = args.batch_size
    max_steps = args.max_steps
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    opt = t.optim.Adam(model.parameters())
    model.noise_schedule = noise_schedule = NoiseSchedule(max_steps, device)
    example_images, = next(iter(testloader))
    example_images = example_images.to(device=device)

    if args.track:
        wandb.init(project=args.project, config=args.__dict__)

    n_examples_seen = 0
    last_image_log = time.time()
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch + 1}')
        epoch_loss = 0
        model.train()
        for x0, in tqdm(trainloader):
            opt.zero_grad()
            x0 = x0.to(device=device)
            num_steps, noise, noised = noise_img(x0, noise_schedule)
            b, c, h, w = x0.shape
            alpha_bar = noise_schedule.alpha_bar_like(x0, num_steps)
            x0_scaled = t.sqrt(alpha_bar) * x0 + t.sqrt(1 - alpha_bar) * noise
            eps_model = model(x0_scaled, num_steps)
            loss = ((noise - eps_model) ** 2).sum()
            loss.backward()
            opt.step()
            n_examples_seen += batch_size
            epoch_loss += loss

            long_since_image_log = (
                time.time() - last_image_log > args.seconds_between_image_logs
            )
            if args.track and long_since_image_log:
                num_steps, noise, noised = noise_img(
                    example_images, noise_schedule
                )
                example_bar = noise_schedule.alpha_bar_like(example_images, num_steps)
                example_scaled = t.sqrt(example_bar) * example_images + t.sqrt(1 - example_bar) * noise
                model.eval()
                with t.inference_mode():
                    noise_model = model(example_scaled, num_steps)
                model.train()
                reconstructed = reconstruct(noised, noise_model, num_steps, noise_schedule)
                wandb_images = log_images(
                    example_images, noised, noise, noise_model, reconstructed,
                )
                wandb.log(
                    dict(
                        images=wandb_images, 
                    ), 
                    step=n_examples_seen
                )
                last_image_log = time.time()

        if args.track:
            wandb.log(dict(
                train_loss=epoch_loss / len(trainloader)
            ), step=n_examples_seen)

        model.eval()
        test_loss = 0
        for x0, in testloader:
            x0 = x0.to(device=device)
            num_steps, noise, noised = noise_img(x0, noise_schedule)
            alpha_bar = repeat(
                noise_schedule.alpha_bar(num_steps),
                'b -> b c h w',
                c=c,
                h=h,
                w=w,
            )
            x0_scaled = t.sqrt(alpha_bar) * x0 + t.sqrt(1 - alpha_bar) * noise
            with t.inference_mode():
                eps_model = model(x0_scaled, num_steps)
            loss = ((noise - eps_model) ** 2).sum()
            test_loss += loss
        if args.track:
            wandb.log(dict(
                train_loss=epoch_loss / len(trainloader)
            ), step=n_examples_seen)
            
    return model.eval().to(device='cpu')
# %%
def log_images(
    img: t.Tensor, noised: t.Tensor, noise: t.Tensor, noise_pred: t.Tensor, 
    reconstructed: t.Tensor, num_images: int = 3
) -> list[wandb.Image]:
    '''
    Convert tensors to a format suitable for logging to Weights and Biases. 
    Returns an image with the ground truth in the upper row, and 
    model reconstruction on the bottom row. 
    Left is the noised image, middle is noise, and reconstructed image is 
    in the rightmost column.
    '''
    actual = t.cat((noised, noise, img), dim=-1)
    pred = t.cat((noised, noise_pred, reconstructed), dim=-1)
    log_img = t.cat((actual, pred), dim=-2)
    images = [wandb.Image(i) for i in log_img[:num_images]]
    return images

#%%
if MAIN:
    args = DiffusionArgs(
        project='w5d3_tiny_diffuser',
        epochs=10,
        track=True,
        seconds_between_image_logs=1,
    ) # This shouldn't take long to train
    model_config = TinyDiffuserConfig(
        args.image_shape,
        args.hidden_size,
        args.max_steps,
    )
    model = TinyDiffuser(model_config).to(device).train()
    trainset = TensorDataset(normalize_img(gradient_images(
        args.n_images, args.image_shape
    )))
    testset = TensorDataset(normalize_img(gradient_images(
        args.n_eval_images, args.image_shape
    )))
    model = train_tiny_diffuser(model, args, trainset, testset)
#%%
def sample(
    model: DiffusionModel, n_samples: int, return_all_steps: bool = False
) -> Union[t.Tensor, list[t.Tensor]]:
    '''
    Sample, following Algorithm 2 in the DDPM paper

    model: The trained noise-predictor
    n_samples: The number of samples to generate
    return_all_steps: 
        if true, return a list of the reconstructed tensors generated at 
        each step, rather than just the final reconstructed image tensor.

    out: 
        shape (B, C, H, W), the denoised images
        or (T, B, C, H, W), if return_all_steps=True (where ith element is 
        batched result of (i+1) steps of sampling)
    '''
    schedule = model.noise_schedule
    device = 'cuda' if args.cuda and t.cuda.is_available() else 'cpu'
    model = model.to(device=device)
    assert schedule is not None
    xs = t.zeros(
        (model.max_steps, n_samples,) + model.image_shape, device=device
    )
    x = t.randn((n_samples, ) + model.image_shape).to(device=device)
    for to_go in range(model.max_steps):
        step = model.max_steps - to_go
        z = t.randn(model.image_shape, device=device) if step > 1 else 0
        alpha = schedule.alpha(step - 1).to(device=device)
        bar = schedule.alpha_bar(step - 1).to(device=device)
        beta = schedule.beta(step - 1).to(device=device)
        t_model = t.full((n_samples,), fill_value=step - 1, device=device)
        eps = model(x, t_model)
        x = (1.0 / t.sqrt(alpha)) * (
            x - ((1 - alpha) / t.sqrt(1 - bar) * eps)
        ) + t.sqrt(beta) * z
        xs[to_go] = x
    if return_all_steps:
        return xs
    else:
        return xs[0, ...].reshape((n_samples, ) + model.image_shape)


#%%
if MAIN:
    print("Generating multiple images")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 6)
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_grid(samples_denormalized, title="Sample denoised images", cols=3)
if MAIN:
    print("Printing sequential denoising")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 1, return_all_steps=True)[::10, 0, :]
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_slideshow(samples_denormalized, title="Sample denoised image slideshow")
#%% [markdown] 
#### The DDPM Architecture

#%%
class GroupNorm(nn.Module):

    def __init__(
        self, num_groups, num_channels, eps=1e-05, affine=True, device=None, 
        dtype=None
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.device = device
        assert num_channels % num_groups == 0, (
            f'num_groups={num_groups}, num_channels={num_channels} '
            'do not divide'
        )
        self.group_size = num_channels // num_groups
        self.groups_to_channels = [
            list(range(g * self.group_size, (g + 1) * self.group_size)) 
            for g in range(num_groups)
        ]
        if self.affine:
            self.weight = nn.Parameter(t.ones(
                self.num_channels, device=device, dtype=dtype
            ))
            self.bias = nn.Parameter(t.zeros(
                self.num_channels, device=device, dtype=dtype
            ))

    def forward(self, x: t.Tensor):
        '''
        x: shape b c h w
        '''
        b, c, h, w = x.shape
        assert c == self.num_channels
        mean = t.zeros_like(x)
        var = t.ones_like(x)
        for channels in self.groups_to_channels:
            channel_mean = x[:, channels, :, :].mean(dim=(1, 2, 3), keepdim=True)
            channel_var = x[:, channels, :, :].var(unbiased=False, dim=(1, 2, 3), keepdim=True)
            mean[:, channels, :, :] = channel_mean
            var[:, channels, :, :] = channel_var
        normalised = (x - mean) / t.sqrt(var + self.eps) 
        if self.affine:
            weight = repeat(self.weight, 'c -> b c h w', b=b, h=h, w=w)
            bias = repeat(self.bias, 'c -> b c h w', b=b, h=h, w=w)
            return normalised * weight + bias
        else:
            return normalised

if MAIN:
    w5d3_tests.test_groupnorm(GroupNorm, affine=False)
    w5d3_tests.test_groupnorm(GroupNorm, affine=True)


# %%
class PositionalEncodingOld(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len # same as base above
        self.embedding_dim = embedding_dim # same as D above

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq_len, embedding_dim)
        output: shape (batch, seq_len, embedding_dim)
        """
        batch, seq_len, embedding_dim = x.shape
        embedding_idx = repeat(
            t.arange(embedding_dim), 
            'e -> b s e', 
            s=seq_len, 
            b=batch
        )
        pos_idx = repeat(
            t.arange(seq_len), 
            's -> b s e', 
            e=embedding_idx, 
            b=batch,
        )
        return t.where(
            embedding_idx % 2 == 0,
            t.sin(pos_idx / t.pow(seq_len, embedding_idx / embedding_dim)),
            t.cos(pos_idx / t.pow(seq_len, (embedding_idx - 1) / embedding_dim))
        )
# %%
class PositionalEncoding(nn.Module):

    def __init__(self, max_steps: int, embedding_dim: int):
        super().__init__()
        self.max_steps = max_steps
        self.embedding_dim = embedding_dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch,) - for each batch element, the number of noise steps
        Out: shape (batch, embedding_dim)
        '''
        batch, = x.shape
        assert (x <= self.max_steps).all()
        embedding_idx = repeat(
            t.arange(self.embedding_dim), 
            'e -> b e', 
            b=batch
        )
        return t.where(
            embedding_idx % 2 == 0,
            t.sin(x / t.pow(10_000, embedding_idx / self.embedding_dim)),
            t.cos(x / t.pow(10_000, (embedding_idx - 1) / self.embedding_dim))
        )
# %%
def swish(x: t.Tensor) -> t.Tensor:
    return x / (1 + t.exp(-x))

class SiLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return swish(x)

if MAIN:
    "YOUR CODE HERE, TO PLOT FUNCTION"
    x_swish = t.linspace(-10, 10, 1_000)
    y_swish = swish(x_swish)
    fig = px.line(x=x_swish, y=y_swish)
    fig.update_layout(title='Sigmoid Linear Unit (swish)', title_x=0.5)
    fig.show()

# %%
def multihead_masked_attention(
    Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int
):
    '''
    Implements multihead masked attention on the matrices Q, K and V.
    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)
    returns: shape (batch, seq, nheads*headsize)
    '''
    new_Q = rearrange(Q, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
    new_K = rearrange(K, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
    new_V = rearrange(V, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)

    attention_scores = einsum('batches nheads seq_Q head_size, batches nheads seq_K head_size -> batches nheads seq_Q seq_K', new_Q, new_K)
    batches, _, seq_Q, head_size = new_Q.shape
    batches, _, seq_K, head_size = new_K.shape
    attention_probabilities = nn.functional.softmax(
        attention_scores / np.sqrt(head_size), dim=-1
    )
    attention_values = einsum(
        'batches nheads seq_Q seq_K, batches nheads seq_K head_size -> batches seq_Q nheads head_size', 
        attention_probabilities, 
        new_V
    )
    return rearrange(
        attention_values, 
        'batches seq_Q nheads head_size -> batches seq_Q (nheads head_size)'
    )

# %%
class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.W_O = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        Return: shape (batch, seq, hidden_size)
        '''
        QKV = self.W_QKV(x)
        Q = QKV[..., :self.hidden_size]
        K = QKV[..., self.hidden_size:-self.hidden_size]
        V = QKV[..., -self.hidden_size:]
        attention_values = multihead_masked_attention(Q, K, V, self.num_heads)
        attention_times_o = self.W_O(attention_values)
        return attention_times_o
# %%
class SelfAttention(nn.Module):
    W_QKV: Linear
    W_O: Linear

    def __init__(self, channels: int, num_heads: int = 4):
        '''
        Self-Attention with two spatial dimensions.

        channels: the number of channels. 
        num_heads: number of attention heads
        Channels should be divisible by the number of heads.
        '''
        super().__init__()
        assert channels % num_heads == 0
        self.W_QKV = nn.Linear(channels, 3 * channels, bias=True)
        self.W_O = nn.Linear(channels, channels, bias=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        '''
        b, c, h, w = x.shape
        assert c == self.channels
        QKV = self.W_QKV(x)
        Q = QKV[..., :self.hidden_size]
        K = QKV[..., self.hidden_size:-self.hidden_size]
        V = QKV[..., -self.hidden_size:]
        attention_values = multihead_masked_attention(Q, K, V, self.num_heads)
        attention_times_o = self.W_O(attention_values)
        return attention_times_o

if MAIN:
    w5d3_tests.test_self_attention(SelfAttention)