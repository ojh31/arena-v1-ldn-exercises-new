#%%
import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from einops import rearrange
from tqdm.auto import tqdm
import wandb
import numpy as np
import plotly.express as px
from typing import Tuple, Callable
#%%
trainset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
testset = datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)

#%%
class Autoencoder(nn.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, img_channels, img_size, out_features, latent_dim_size):
        super().__init__()
        flatten_size = img_channels * (img_size ** 2)
        self.encoder = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(flatten_size, out_features),
            nn.ReLU(),
            nn.Linear(out_features, latent_dim_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, out_features),
            nn.ReLU(),
            nn.Linear(out_features, flatten_size),
            Rearrange('b (c h w) -> b c h w', c=img_channels, h=img_size, w=img_size)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed
# %%
def train_autoencoder(config):
    device = 'cuda' if config['cuda'] and t.cuda.is_available() else 'cpu'
    autoencoder = Autoencoder(
        img_channels=config['img_channels'],
        img_size=config['img_size'],
        out_features=config['out_features'],
        latent_dim_size=config['latent_dim_size'],
    ).train().to(device=device)
    batch_size = config['batch_size']
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    example_images, example_labels = next(iter(testloader))
    opt = t.optim.Adam(autoencoder.parameters())

    if config['track']:
        wandb.init(project=config['project'], config=config)

    n_examples_seen = 0
    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')
        epoch_loss = 0
        epoch_batches = 0
        for x, y in tqdm(trainloader):
            x = x.to(device=device)
            y = y.to(device=device)
            opt.zero_grad()
            reconstructed = autoencoder(x)
            loss = ((reconstructed - x) ** 2).sum()
            loss.backward()
            opt.step()
            n_examples_seen += x.shape[0]
            epoch_loss += loss
            epoch_batches += 1

        if config['track']:
            autoencoder.eval()
            with t.inference_mode():
                new_image_arr = autoencoder(example_images)
            autoencoder.train()
            arrays = rearrange(
                new_image_arr, "b c h w -> b h w c"
            ).detach().cpu().numpy()
            images = [
                wandb.Image(arr, caption=example_labels[i]) for i, arr in enumerate(arrays)
            ]
            wandb.log(
                dict(
                    train_loss=epoch_loss / epoch_batches, 
                    images=images, 
                ), 
                step=n_examples_seen
            )

    return autoencoder.eval().to(device='cpu')

#%%
autoencoder_config = dict(
    project='w5d2_autoencoder',
    track = True,
    cuda = False,
    epochs = 10,
    batch_size = 64,
    img_channels=1,
    img_size=28,
    out_features=100,
    latent_dim_size=5,
)
#%%
trained_autoencoder = train_autoencoder(autoencoder_config)
#%% [markdown]
#### Decoding latent space dimensions (Autoencoder)
#%%
def visualise_decoder(model: nn.Module, config: dict):
    # Choose number of interpolation points, and 
    # interpolation range (you might need to adjust these)
    n_points = 11
    interpolation_range = (-10, 10)

    # Constructing latent dim data by making two of 
    # the dimensions vary independently between 0 and 1
    latent_dim_data = t.zeros(
        (n_points, n_points, config['latent_dim_size']), 
        device='cpu'
    )
    x = t.linspace(*interpolation_range, n_points)
    latent_dim_data[:, :, 0] = x.unsqueeze(0)
    latent_dim_data[:, :, 1] = x.unsqueeze(1)
    # Rearranging so we have a single batch dimension
    latent_dim_data = rearrange(latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim")

    # Getting model output, and normalising & truncating it in the range [0, 1]
    output = model.decoder(latent_dim_data).detach().cpu().numpy()
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = rearrange(
        output_truncated, 
        "(b1 b2) 1 height width -> (b1 height) (b2 width)", 
        b1=n_points
    )

    # Plotting results
    fig = px.imshow(output_single_image, color_continuous_scale="greys_r")
    fig.update_layout(
        title_text="Decoder output from varying first two latent space dims ({})".format(config['project']), 
        title_x=0.5,
        coloraxis_showscale=False, 
        xaxis=dict(
            tickmode="array", 
            tickvals=list(range(14, 14 + 28 * n_points, 28)), 
            ticktext=[f"{i:.2f}" for i in x]
        ),
        yaxis=dict(
            tickmode="array", 
            tickvals=list(range(14, 14 + 28 * n_points, 28)), 
            ticktext=[f"{i:.2f}" for i in x]
        )
    )
    fig.show()
#%%
visualise_decoder(trained_autoencoder, autoencoder_config)
# %%
def visualise_encoder(encoder_fn: Callable, config: dict):
    testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=True)
    example_images, example_labels = next(iter(testloader))
    example_codes = encoder_fn(example_images)
    encoder_x = example_codes[:, 0].detach().numpy()
    encoder_y = example_codes[:, 1].detach().numpy()
    encoder_color = example_labels.detach().numpy().astype(np.uint8)
    fig = px.scatter(
        x=encoder_x, y=encoder_y, color=encoder_color, 
        title='First 2 dimensions of latent space for different classes ({})'.format(config['project']),
    )
    fig.update_layout(title_x=0.5)
    fig.show()
#%%
visualise_encoder(trained_autoencoder.encoder, autoencoder_config)

# %% [markdown]
#### Variational Autoencoder
#%%
# %%
class VariationalEncoder(nn.Module):

    def __init__(self, img_channels, img_size, out_features, latent_dim_size):
        super().__init__()
        flatten_size = img_channels * (img_size ** 2)
        self.linear_encoder = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(flatten_size, out_features),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(out_features, latent_dim_size)
        self.std_layer = nn.Sequential(
            nn.Linear(out_features, latent_dim_size),
        )

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        '''
        Returns a tuple of (mu, logsigma, z), where:
            mu and logsigma are the outputs of your encoder module
            z is the sampled latent vector taken from distribution N(mu, sigma**2)
        '''
        x_common = self.linear_encoder(x)
        mu = self.mean_layer(x_common)
        logsigma = self.std_layer(x_common)
        eps = t.randn_like(mu)
        z = mu + t.exp(logsigma) * eps
        return mu, logsigma, z
# %%
class VAE(nn.Module):
    encoder: VariationalEncoder
    decoder: nn.Sequential

    def __init__(self, img_channels, img_size, out_features, latent_dim_size):
        super().__init__()
        flatten_size = img_channels * (img_size ** 2)
        self.encoder = VariationalEncoder(
            img_channels, img_size, out_features, latent_dim_size
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, out_features),
            nn.ReLU(),
            nn.Linear(out_features, flatten_size),
            Rearrange('b (c h w) -> b c h w', c=img_channels, h=img_size, w=img_size)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed
#%%
def train_vae(config):
    t.manual_seed(config['seed'])
    device = 'cuda' if config['cuda'] and t.cuda.is_available() else 'cpu'
    vae = VAE(
        img_channels=config['img_channels'],
        img_size=config['img_size'],
        out_features=config['out_features'],
        latent_dim_size=config['latent_dim_size'],
    ).train().to(device=device)
    batch_size = config['batch_size']
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    example_images, example_labels = next(iter(testloader))
    opt = t.optim.Adam(vae.parameters())

    if config['track']:
        wandb.init(project=config['project'], config=config)

    n_examples_seen = 0
    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_loss = 0
        epoch_batches = 0
        for x, y in tqdm(trainloader):
            x = x.to(device=device)
            y = y.to(device=device)
            opt.zero_grad()
            mu, logsigma, z = vae.encoder(x)
            sigma = t.exp(logsigma)
            reconstructed = vae.decoder(z)
            recon_loss = ((reconstructed - x) ** 2).sum()
            kl_loss = (0.5 * (mu ** 2 + sigma ** 2 - 1) - logsigma).sum()
            loss = recon_loss + config['kl_coef'] * kl_loss
            loss.backward()
            opt.step()
            n_examples_seen += x.shape[0]
            epoch_loss += loss
            epoch_recon_loss += recon_loss
            epoch_kl_loss += kl_loss
            epoch_batches += 1

        if config['track']:
            vae.eval()
            with t.inference_mode():
                _, _, example_z = vae.encoder(example_images)
                new_image_arr = vae.decoder(example_z)
            vae.train()
            arrays = rearrange(
                new_image_arr, "b c h w -> b h w c"
            ).detach().cpu().numpy()
            images = [
                wandb.Image(arr, caption=example_labels[i]) for i, arr in enumerate(arrays)
            ]
            wandb.log(
                dict(
                    total_loss=epoch_loss / epoch_batches, 
                    recon_loss=epoch_recon_loss / epoch_batches, 
                    kl_loss=epoch_kl_loss / epoch_batches, 
                    images=images, 
                ), 
                step=n_examples_seen
            )

    return vae.eval().to(device='cpu')
#%%
vae_config = dict(
    project='w5d2_vae',
    track = True,
    cuda = False,
    epochs = 10,
    batch_size = 64,
    img_channels=1,
    img_size=28,
    out_features=100,
    latent_dim_size=5,
    kl_coef=0.1,
    seed=0,
)
# %%
trained_vae = train_vae(vae_config)
# %%
visualise_decoder(trained_vae, vae_config)
# %%
visualise_encoder(lambda x: trained_vae.encoder(x)[-1], vae_config)
# %%
