#%%
import argparse
import os
import random
import time
import sys
sys.path.append('/home/oskar/projects/arena-v1-ldn-exercises-new')
from distutils.util import strtobool
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from typing import Any, List, Optional, Union, Tuple, Iterable
from einops import rearrange
from utils import ppo_parse_args, make_env
import importlib
import tests

importlib.reload(tests)
MAIN = __name__ == "__main__"
RUNNING_FROM_FILE = "ipykernel_launcher" in os.path.basename(sys.argv[0])
# %%
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        obs_shape = np.array(envs.single_action_space.shape).prod()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1),
        )
# %%
@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.

    next_value: shape (1, env) - 
        represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)

    Return: shape (t, env)
    '''
    t_max, n_env = values.shape
    next_values = t.concat((values[1:, ], next_value))
    next_dones = t.concat((dones[1:, ], next_done.unsqueeze(0)))
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values  
    adv = deltas.clone().to(device)
    for to_go in range(1, t_max):
        t_idx = t_max - to_go - 1
        t.testing.assert_allclose(adv[t_idx], deltas[t_idx])
        adv[t_idx] += (
            gamma * gae_lambda * adv[t_idx + 1] * (1.0 - next_dones[t_idx]) 
        )
    return adv

if MAIN and RUNNING_FROM_FILE:
    tests.test_compute_advantages(compute_advantages)
    print('Passed test_compute_advantages')
# %%
@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(
    batch_size: int, minibatch_size: int
) -> list[np.ndarray]:
    '''
    Return a list of length (batch_size // minibatch_size) where 
    each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    n = batch_size // minibatch_size
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    return [indices[i::n] for i in range(n)]

if MAIN and RUNNING_FROM_FILE:
    tests.test_minibatch_indexes(minibatch_indexes)

def make_minibatches(
    obs: t.Tensor,
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> list[Minibatch]:
    '''
    Flatten the environment and steps dimension into one batch dimension, 
    then shuffle and split into minibatches.
    '''
    n_steps, n_env = values.shape
    n_dim = n_steps * n_env
    indexes = minibatch_indexes(batch_size=batch_size, minibatch_size=minibatch_size)
    obs_flat = obs.reshape(batch_size, obs_shape)
    act_flat = actions.reshape(batch_size, action_shape)
    probs_flat = logprobs.reshape(batch_size, action_shape)
    adv_flat = advantages.reshape(n_dim)
    val_flat = values.reshape(n_dim)
    return [
        Minibatch(
            obs_flat[idx], probs_flat[idx], act_flat[idx], adv_flat[idx], 
            adv_flat[idx] + val_flat[idx], val_flat[idx]
        )
        for idx in indexes
    ]

# %%
def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, 
    clip_coef: float
) -> t.Tensor:
    '''
    Return the policy loss, suitable for maximisation with gradient ascent.

    probs: 
        a distribution containing the actor's unnormalized logits of 
        shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.

    normalize: if true, normalize mb_advantages to have mean 0, variance 1
    '''
    adv_norm = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()
    ratio = t.exp(probs.log_prob(mb_action)) / t.exp(mb_logprobs)
    min_left = ratio * adv_norm
    min_right = t.clip(ratio, 1 - clip_coef, 1 + clip_coef) * adv_norm
    return t.minimum(min_left, min_right).mean()


if MAIN and RUNNING_FROM_FILE:
    tests.test_calc_policy_loss(calc_policy_loss)
# %%
def calc_value_function_loss(
    critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, v_coef: float
) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    v_coef: 
        the coefficient for the value loss, which weights its contribution to 
        the overall loss. Denoted by c_1 in the paper.
    '''
    output = critic(mb_obs)
    return v_coef * (output - mb_returns).pow(2).mean() / 2

if MAIN and RUNNING_FROM_FILE:
    tests.test_calc_value_function_loss(calc_value_function_loss)
# %%
def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.

    ent_coef: 
        The coefficient for the entropy loss, which weights its contribution to the overall loss. 
        Denoted by c_2 in the paper.
    '''
    return probs.entropy().mean() * ent_coef

if MAIN and RUNNING_FROM_FILE:
    tests.test_calc_entropy_loss(calc_entropy_loss)
# %%
