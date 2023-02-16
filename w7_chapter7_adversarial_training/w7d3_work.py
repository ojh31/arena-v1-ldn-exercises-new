#%%
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import models
import sys
from einops import rearrange
import torchvision
import time
from typing import Callable, Dict
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import wandb

sys.path.append('/home/oskar/projects/arena-v1-ldn-exercises-new')

from w7_chapter7_adversarial_training import w7d3_utils, w7d3_tests
# %%
def untargeted_FGSM(
    x_batch, true_labels, network, normalize, eps=8/255., step_size=None, num_steps=None, **kwargs
):
    '''
    Generates a batch of untargeted FGSM adversarial examples

    x_batch (torch.Tensor): the batch of unnormalized input examples.
    true_labels (torch.Tensor): the batch of true labels of the example.
    network (nn.Module): the network to attack.
    normalize (function): 
        a function which normalizes a batch of images 
        according to standard imagenet normalization.
    eps (float): the bound on the perturbations.
    '''
    loss_fn = nn.CrossEntropyLoss(reduce="mean")
    x_batch.requires_grad = True
    pred = network(normalize(x_batch), **kwargs)
    loss = loss_fn(pred, true_labels)
    loss.backward()
    with torch.inference_mode():
        return x_batch + eps * x_batch.grad.sign()

w7d3_tests.test_untargeted_attack(untargeted_FGSM, eps=8/255.)
#%%
def targeted_FGSM(
    x_batch, target_labels, network, normalize, eps=8/255., **kwargs
):
    '''
    Generates a batch of targeted FGSM adversarial examples

    x_batch (torch.Tensor): the unnormalized input example.
    target_labels (torch.Tensor): the labels the model will predict after the attack.
    network (nn.Module): the network to attack.
    normalize (function): a function which normalizes a batch of images 
        according to standard imagenet normalization.
    eps (float): the bound on the perturbations.
    '''
    loss_fn = nn.CrossEntropyLoss(reduce="mean")
    x_batch.requires_grad = True
    pred = network(normalize(x_batch), **kwargs)
    loss = loss_fn(pred, target_labels)
    loss.backward()
    with torch.inference_mode():
        return x_batch - eps * x_batch.grad.sign()

w7d3_tests.test_targeted_attack(targeted_FGSM, target_idx=8, eps=8/255.)
# FIXME: target label does not match prediction
#%%
def normalize_l2(x_batch):
    '''
    Expects x_batch.shape == [N, C, H, W]
    where N is the batch size, 
    C is the channels (or colors in our case),
    H, W are height and width respectively.

    Note: To take the l2 norm of an image, you will want to flatten its 
    dimensions (be careful to preserve the batch dimension of x_batch).
    '''
    x_flat = rearrange(x_batch, 'b c h w -> b (c h w)')
    return x_batch / x_flat.norm()

def tensor_clamp_l2(x_batch, center, radius):
    '''
    Batched clamp of x into l2 ball around center of given radius.
    '''
    from_center = rearrange(x_batch - center, 'b c h w -> b (c h w)').norm()
    clamp = center + normalize_l2(x_batch - center) * radius
    return x_batch.where(from_center <= radius, clamp)

def PGD_l2(
        x_batch, true_labels, network, normalize, num_steps=20, step_size=3./255, eps=128/255., 
        **kwargs
    ):
        '''
        Returns perturbed batch of images
        '''
        # Initialize our adversial image
        x_adv = x_batch.detach().clone()
        x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)

        for _ in range(num_steps):
            x_adv.requires_grad_()

            # Calculate gradients
            with torch.enable_grad():
              logits = network(normalize(x_adv))
              loss = F.cross_entropy(logits, true_labels, reduction='sum')

            # Normalize the gradients with your L2
            grad = normalize_l2(torch.autograd.grad(loss, x_adv, only_inputs=True)[0])

            # Take a step in the gradient direction.
            x_adv = x_adv.detach() + step_size * grad
            # Project (by clamping) the adversarial image back onto the hypersphere
            # around the image.
            x_adv = tensor_clamp_l2(x_adv, x_batch, eps).clamp(0, 1)

        return x_adv
# %%
w7d3_tests.test_untargeted_attack(PGD_l2, eps=128/255.)
# %%
def untargeted_PGD(x_batch, true_labels, network, normalize, num_steps=10, step_size=0.01, eps=8/255., **kwargs):
    '''Generates a batch of untargeted PGD adversarial examples

    x_batch (torch.Tensor): the batch of unnormalized input examples.
    true_labels (torch.Tensor): the batch of true labels of the example.
    network (nn.Module): the network to attack.
    normalize (function): a function which normalizes a batch of images 
        according to standard imagenet normalization.
    num_steps (int): the number of steps to run PGD.
    step_size (float): the size of each PGD step.
    eps (float): the bound on the perturbations.
    '''
    x_adv = x_batch.detach().clone()
    x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)

    for i in range(num_steps):
        x_adv.requires_grad_()

    # Calculate gradients
    with torch.enable_grad():
        logits = network(normalize(x_adv))
        loss = F.cross_entropy(logits, true_labels, reduction='sum')
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    # Perform one gradient step
    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())

    # Project the image to the ball.
    x_adv = torch.maximum(x_adv, x_batch - eps)
    x_adv = torch.minimum(x_adv, x_batch + eps)

    return x_adv

w7d3_tests.test_untargeted_attack(untargeted_PGD, eps=8/255.)
# %%
def targeted_PGD(x_batch, target_labels, network, normalize, num_steps=100, step_size=0.01, eps=8/255., **kwargs):
    '''Generates a batch of untargeted PGD adversarial examples

    Args:
    x_batch (torch.Tensor): the batch of preprocessed input examples.
    target_labels (torch.Tensor): the labels the model will predict after the attack.
    network (nn.Module): the network to attack.
    normalize (function): a function which normalizes a batch of images 
        according to standard imagenet normalization.
    num_steps (int): the number of steps to run PGD.
    step_size (float): the size of each PGD step.
    eps (float): the bound on the perturbations.
    '''
    x_adv = x_batch.detach().clone()
    x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)

    for i in range(num_steps):
        x_adv.requires_grad_()

    # Calculate gradients
    with torch.enable_grad():
        logits = network(normalize(x_adv))
        loss = F.cross_entropy(logits, target_labels, reduction='sum')
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    # Perform one gradient step
    # Note that this time we use gradient descent instead of gradient ascent
    x_adv = x_adv.detach() - step_size * torch.sign(grad.detach())

    # Project the image to the ball
    x_adv = torch.maximum(x_adv, x_batch - eps)
    x_adv = torch.minimum(x_adv, x_batch + eps)

    return x_adv

# Try changing the target_idx around!
w7d3_tests.test_targeted_attack(targeted_PGD, target_idx=3, eps=8/255.)
# %%
# Attack a normal model (we only support targeted methods)
w7d3_utils.attack_normal_model(
    targeted_PGD, 
    target_idx=10, 
    eps=8/255., 
    num_steps=10, 
    step_size=0.01
)
# %%
# Attack an adversarially trained model (we only support targeted methods)
w7d3_utils.attack_adversarially_trained_model(
    targeted_PGD, 
    target_idx=10, 
    eps=8/255., 
    num_steps=10, 
    step_size=0.01
) # FIXME: Original image is the same as the adversarial image here?
# %%
# Train your own adversarially trained model
# Use a relatively simple and small model, 
# e.g. your MNIST from week 0, or your resnet with CIFAR10 data.
# Use untargeted PGD  
    # eps=8/255., 
    # num_steps=10, 
    # step_size=0.01

#%%
# from w0d3 conv nets

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
mnist_normalise = transforms.Normalize((0.1307,), (0.3081,))
trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

#%%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.max2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
convnet_config = dict(
    epochs = 1,
    batch_size = 128,
    lambda_adv = 1.0,
    epsilon = 0.3, 
    num_steps = 10, 
    step_size = 0.01,
    attacker = untargeted_FGSM,
    log_image_freq = 50,
)
#%%
def train_convnet(
    config: Dict = None,
) -> ConvNet:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    '''
    wandb.init(project='w7d3_robust_mnist', config=config)
    batch_size = config['batch_size']
    epochs = config['epochs']
    attacker = config['attacker']
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device),
        )
    model = ConvNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loss_list = []
    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)
        for i, (x, y) in enumerate(progress_bar):

            x = x.to(device)
            y = y.to(device)
            x_norm = mnist_normalise(x)

            y_hat = model(x_norm)

            x_adv = attacker(
                x,
                y,
                model,
                mnist_normalise,
                eps=config['epsilon'], 
                num_steps=config['num_steps'], 
                step_size=config['step_size']
            )
            x_adv_norm = mnist_normalise(x_adv)
            y_adv_hat = model(x_adv_norm)

            ce_loss = loss_fn(y_hat, y)
            adv_loss = config['lambda_adv'] * loss_fn(y_adv_hat, y)
            loss = ce_loss + adv_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            progress_bar.set_description(
                f"Epoch = {epoch}, CE Loss = {ce_loss.item():.4f}, "
                f"ADV Loss = {adv_loss.item():.4f}"
            )
            wandb.log({
                'ce_loss': ce_loss.item(), 
                'adv_loss': adv_loss.item(), 
                'total_loss': adv_loss.item() + ce_loss.item(),
            })

            if i % config['log_image_freq'] == 0:
                pred = F.softmax(y_hat[0], dim=0)
                pred_index = pred.argmax().item()
                pred_conf = pred[pred_index]

                adv_pred = F.softmax(y_adv_hat[0], dim=0)
                adv_pred_index = adv_pred.argmax().item()
                adv_pred_conf = adv_pred[adv_pred_index]
                wandb.log({
                    'real_examples': wandb.Image(
                        x[0], 
                        caption=f'true={y[0]}, pred={pred_index}, conf={pred_conf:.1%}'
                    ),
                    'adversarial_examples': wandb.Image(
                        x_adv[0], 
                        caption=f'true={y[0]}, pred={adv_pred_index}, conf={adv_pred_conf:.1%}'
                    ),
                })

    model_path = os.path.join(wandb.run.dir, "model.h5")
    print(f"Saving model to: {model_path}")
    wandb.save(model_path)
    wandb.finish()
    return model.eval()
 #%%
def test_untargeted_attack_mnist(model, untargeted_adv_attack, eps):
    # Load the models
    model = model.eval()
    
    # Load the preprocessed image
    image, true_label = testset[0]
    image = image.to(device=device)
    true_index = int(true_label)
    norm_image = mnist_normalise(image)

    # Generate predictions
    _, label, confidence = w7d3_utils.make_single_prediction(model, norm_image)

    # Generate Adversarial Example
    true_index = torch.Tensor([true_index]).type(torch.long)
    adv_image = untargeted_adv_attack(
        image.unsqueeze(0), 
        true_index, 
        model, 
        mnist_normalise, 
        eps=eps
    ).squeeze(0)
    norm_adv_image = mnist_normalise(adv_image)

    # Display Results
    _, adv_label, adv_confidence = w7d3_utils.make_single_prediction(model, norm_adv_image)

    # Display Images
    w7d3_utils.display_adv_images(
        image, 
        adv_image,
        (label, confidence),
        (adv_label, adv_confidence),
        channels_first=True,
        denormalize=False
    )
#%%
convnet_config['lambda_adv'] = 0.0
convnet_model = train_convnet(convnet_config)
#%%
convnet_config['lambda_adv'] = 1.0
robust_model = train_convnet(convnet_config)
#%% [markdown]
#### Vanilla model diagnostics
# %%
test_untargeted_attack_mnist(convnet_model, untargeted_FGSM, eps=0.3)
# %%
test_untargeted_attack_mnist(convnet_model, untargeted_FGSM, eps=0.6)
# %%
test_untargeted_attack_mnist(convnet_model, untargeted_FGSM, eps=0.8)
#%% [markdown]
#### Robust model diagnostics
# %%
test_untargeted_attack_mnist(robust_model, untargeted_FGSM, eps=0.3)
# %%
test_untargeted_attack_mnist(robust_model, untargeted_FGSM, eps=0.6)
# %%
test_untargeted_attack_mnist(robust_model, untargeted_FGSM, eps=0.8)
# #%%
# test_untargeted_attack_mnist(convnet_model, untargeted_PGD, eps=0.95)
# # %%
# test_untargeted_attack_mnist(convnet_model, untargeted_PGD, eps=0.9)
# %%
