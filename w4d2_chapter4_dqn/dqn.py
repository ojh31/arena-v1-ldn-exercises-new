#%%
import argparse
import os
import sys
import random
import time
import re
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, List, Optional, Union, Tuple, Iterable
import gym
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from gym.spaces import Discrete, Box
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from numpy.random import Generator
import gym.envs.registration
import pandas as pd
import random
from w3d5_chapter4_tabular.utils import make_env
from w4d2_chapter4_dqn import utils

MAIN = __name__ == "__main__"
os.environ["SDL_VIDEODRIVER"] = "dummy"
# %%
class QNetwork(nn.Module):
    def __init__(
        self, dim_observation: int, num_actions: int, hidden_sizes: list[int] = [120, 84]
    ):
        super().__init__()
        h1, h2 = hidden_sizes
        self.linear1 = nn.Linear(dim_observation, h1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(h2, num_actions)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

if MAIN:
    net = QNetwork(dim_observation=4, num_actions=2)
    n_params = sum((p.nelement() for p in net.parameters()))
    print(net)
    print(f"Total number of parameters: {n_params}")
    print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
    assert n_params == 10934
# %%
@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    obs: shape (sample_size, *observation_shape), dtype t.float
    actions: shape (sample_size, ) dtype t.int
    rewards: shape (sample_size, ), dtype t.float
    dones: shape (sample_size, ), dtype t.bool
    next_observations: shape (sample_size, *observation_shape), dtype t.float
    '''

    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor



class ReplayBuffer:
    rng: Generator
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

    def __init__(
        self, buffer_size: int, num_actions: int, observation_shape: tuple, 
        num_environments: int, seed: int
    ):
        assert num_environments == 1, (
            "This buffer only supports SyncVectorEnv with 1 environment inside."
        )
        self.observations = t.tensor([])
        self.actions = t.tensor([], dtype=t.int64)
        self.rewards = t.tensor([])
        self.dones = t.tensor([], dtype=t.bool)
        self.next_observations = t.tensor([])
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.num_environments = num_environments
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
        dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
        '''
        obs: shape (num_environments, *observation_shape) 
            Observation before the action
        actions: shape (num_environments, ) 
            Action chosen by the agent
        rewards: shape (num_environments, ) 
            Reward after the action
        dones: shape (num_environments, ) 
            If True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape) 
            Observation after the action
            If done is True, this should be the terminal observation, NOT the 
            first observation of the next episode.
        '''
        self.observations = t.cat((
            self.observations, t.tensor(obs)
        ))[-self.buffer_size:, ...]
        self.actions = t.cat((
            self.actions, t.tensor(actions)
        ))[-self.buffer_size:, ...]
        self.rewards = t.cat((
            self.rewards, t.tensor(rewards)
        ))[-self.buffer_size:, ...]
        self.dones = t.cat((
            self.dones, t.tensor(dones)
        ))[-self.buffer_size:, ...]
        self.next_observations = t.cat((
            self.next_observations, t.tensor(next_obs)
        ))[-self.buffer_size:, ...]


    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''
        Uniformly sample sample_size entries from the buffer and convert them to 
        PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        choices = self.rng.integers(
            0, self.observations.shape[0], sample_size
        )
        return ReplayBufferSamples(
            observations=self.observations[choices, ...].to(device),
            actions=self.actions[choices, ...].to(device),
            rewards=self.rewards[choices, ...].to(device),
            dones=self.dones[choices, ...].to(device),
            next_observations=self.next_observations[choices, ...].to(device),
        )

if MAIN:
    utils.test_replay_buffer_single(ReplayBuffer)
    utils.test_replay_buffer_deterministic(ReplayBuffer)
    utils.test_replay_buffer_wraparound(ReplayBuffer)
# %%
if MAIN:
    rb = ReplayBuffer(
        buffer_size=256, num_actions=2, observation_shape=(4,), num_environments=1, seed=0
    )
    envs = gym.vector.SyncVectorEnv([utils.make_env("CartPole-v1", 0, 0, False, "test")])
    obs = envs.reset()
    for i in range(512):
        actions = np.array([1])
        (next_obs, rewards, dones, infos) = envs.step(actions)
        # real_next_obs = next_obs.copy()
        # for (i, done) in enumerate(dones):
        #     if done:
        #         real_next_obs[i] = infos[i]["terminal_observation"]
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs
    sample = rb.sample(128, t.device("cpu"))
    columns = ["cart_pos", "cart_v", "pole_angle", "pole_v"]
    df = pd.DataFrame(rb.observations, columns=columns)
    df.plot(subplots=True, title="Replay Buffer")
    df2 = pd.DataFrame(sample.observations, columns=columns)
    df2.plot(subplots=True, title="Shuffled Replay Buffer")
# %%
def linear_schedule(
    current_step: int, start_e: float, end_e: float, 
    exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at 
    step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    end_step = exploration_fraction * total_timesteps
    return max(start_e + current_step * (end_e - start_e) / end_step, end_e)

if MAIN:
    epsilons = [
        linear_schedule(
            step, start_e=1.0, end_e=0.05, 
            exploration_fraction=0.5, total_timesteps=500
        )
        for step in range(500)
    ]
    utils.test_linear_schedule(linear_schedule)
# %%
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, 
    obs: t.Tensor, epsilon: float
) -> np.ndarray:
    '''
    With probability epsilon, take a random action. Otherwise, take a greedy action 
    according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    '''
    if rng.random() < epsilon:
        pi = rng.integers(0, envs.single_action_space.n, envs.num_envs)
    else:
        action_values = q_network(obs)
        pi = action_values.argmax(dim=1).detach().numpy()
    return pi

if MAIN:
    utils.test_epsilon_greedy_policy(epsilon_greedy_policy)
# %%
