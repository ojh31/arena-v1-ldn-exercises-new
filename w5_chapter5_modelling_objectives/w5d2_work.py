#%%
import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
#%%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
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
        flatten_size = img_channels * (img_size ** 2)
        self.encoder = nn.Sequential(
            Rearrange('b c h w -> (b c h w)'),
            nn.Linear(flatten_size, out_features),
            nn.ReLU(),
            nn.Linear(out_features, latent_dim_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, out_features),
            nn.ReLU(),
            nn.Linear(out_features, flatten_size),
            Rearrange('(b c h w) -> b c h w', c=img_channels, h=img_size, w=img_size)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed
# %%
def train_autoencoder():
    track = False
    cuda = False
    epochs = 1
    device = 'cpu'
    batch_size = 64
    autoencoder = Autoencoder(1, 28, 100, 5).train().to(device=device)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    opt = t.optim.Adam(autoencoder.params())

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')

        for x, y in tqdm(trainloader):
            x = x.to(device=device)
            y = y.to(device=device)
            opt.zero_grad()
            reconstructed = autoencoder(x)
            loss = ((reconstructed - x) ** 2).sum()
            loss.backward()
            opt.step()

    return autoencoder