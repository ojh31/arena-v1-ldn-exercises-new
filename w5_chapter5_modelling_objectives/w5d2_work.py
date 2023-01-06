#%%
import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from einops import rearrange
from tqdm.auto import tqdm
import wandb
import time
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
    opt = t.optim.Adam(autoencoder.parameters())
    example_images = next(iter(testloader))
    last_image_log = time.time()

    if config['track']:
        wandb.init(project='w5d2_autoencoder', config=config)

    n_examples_seen = 0
    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for x, y in tqdm(trainloader):
            x = x.to(device=device)
            y = y.to(device=device)
            opt.zero_grad()
            reconstructed = autoencoder(x)
            loss = ((reconstructed - x) ** 2).sum()
            loss.backward()
            opt.step()

            long_since_image_log = (
                time.time() - last_image_log > config['seconds_between_image_logs']
            )
            if config['track'] and long_since_image_log:
                autoencoder.eval()
                with t.inference_mode():
                    new_image_arr = autoencoder(example_images)
                autoencoder.train()
                arrays = rearrange(
                    new_image_arr, "b c h w -> b h w c"
                ).detach().cpu().numpy()
                images = [wandb.Image(arr) for arr in arrays]
                wandb.log(dict(train_loss=loss, images=images), step=n_examples_seen)
                last_image_log = time.time()

            n_examples_seen += x.shape[0]

    return autoencoder

#%%
autoencoder_config = dict(
    track = False,
    cuda = False,
    epochs = 1,
    device = 'cpu',
    batch_size = 64,
    img_channels=1,
    img_size=28,
    out_features=100,
    latent_dim_size=5,
    seconds_between_image_logs=10,
)
#%%
trained_autoencoder = train_autoencoder(autoencoder_config)
#%%
