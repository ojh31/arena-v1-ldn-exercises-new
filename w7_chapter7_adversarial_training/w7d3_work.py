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

sys.path.append('/home/oskar/projects/arena-v1-ldn-exercises-new')

from w7_chapter7_adversarial_training import w7d3_utils, w7d3_tests
# %%
def untargeted_FGSM(
    x_batch, true_labels, network, normalize, eps=8/255., **kwargs
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
    torch.norm()
    x_flat.norm

def tensor_clamp_l2(x_batch, center, radius):
    '''Batched clamp of x into l2 ball around center of given radius.'''

    pass

def PGD_l2(x_batch, true_labels, network, normalize, num_steps=20, step_size=3./255, eps=128/255., **kwargs):
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