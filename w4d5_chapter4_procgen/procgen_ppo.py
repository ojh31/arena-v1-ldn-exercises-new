import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from distutils.util import strtobool
from itertools import chain
from typing import Any, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange
from gym.spaces import Discrete
from tests import (test_agent, test_calc_entropy_loss, test_calc_policy_loss,
                   test_calc_value_function_loss, test_compute_advantages)
from utils import make_env, ppo_parse_args, PPOArgs
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


import math

MAIN = __name__ == "__main__"


from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import math
import wandb

class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, rew, done, truncated, info) = super().step(action)
        done = np.abs(obs[0]) > 2.4
        rew = np.sin(obs[0]*obs[3]) # weight towards spinnging
        # reward = 1 - np.abs(next_obs[:,2]/(12 * 2 * math.pi / 360)) # reward shaping angel
        # reward = 1 - (next_obs[:, 0]/2.4)**2 # reward shaping x-location
        return obs, rew, done, truncated, info

gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=1000)

class EasyMountainCart(MountainCarEnv):
    def step(self, action):
        (obs, rew, done, truncated, info) = super().step(action)
        

        position, velocity = obs

        height = self._height(position)

        #rew = self.gravity*height + 0.05*velocity/math.cos(theta)# reward h
        coeff_velocity = 100
        rew = rew +  (coeff_velocity*0.5*(velocity**2)*(1+(1.35*math.cos(3*position))**2)+self.gravity*height)/0.01
        
        # weight towards spinnging
        # reward = 1 - np.abs(next_obs[:,2]/(12 * 2 * math.pi / 360)) # reward shaping angel
        # reward = 1 - (next_obs[:, 0]/2.4)**2 # reward shaping x-location
        return obs, rew, done, truncated, info

gym.envs.registration.register(id="EasyMountainCart-v0", entry_point=EasyMountainCart, max_episode_steps=200)

class EasyMountainCartContinuous(Continuous_MountainCarEnv):
    def __init__(self):
        super().__init__()
        self.gravity = 0.0025
    def step(self, action):
        (obs, rew, done, truncated, info) = super().step(action)
        

        position, velocity = obs

        height = self._height(position)

        #rew = self.gravity*height + 0.05*velocity/math.cos(theta)# reward h
        coeff_velocity = 100
        rew = rew +  (coeff_velocity*0.5*(velocity**2)*(1+(1.35*math.cos(3*position))**2)+self.gravity*height)/0.01
        
        # weight towards spinnging
        # reward = 1 - np.abs(next_obs[:,2]/(12 * 2 * math.pi / 360)) # reward shaping angel
        # reward = 1 - (next_obs[:, 0]/2.4)**2 # reward shaping x-location
        return obs, rew, done, truncated, info

gym.envs.registration.register(id="EasyMountainCartContinuous-v0", entry_point=EasyMountainCartContinuous, max_episode_steps=200)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_conv(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


class CnnBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = layer_init(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4))
        self.c2 = layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2))
        self.c3 = layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1))
        self.flatten = nn.Flatten()
        self.fc1 = layer_init(nn.Linear(1024, 512))

    def forward(self, x):
        x = rearrange(x, 'e h w c -> e c h w')
        x = x / 255.0
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x


class CnnAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv, shared=True):
        super().__init__()
        action_space_size = int(np.prod(envs.single_action_space.shape))
        assert envs.single_observation_space.shape == (64, 64, 3)

        if shared:
            self.shared = CnnBase()
            self.actor = nn.Sequential(
                self.shared,
                nn.Linear(512, action_space_size)
            )
            self.critic = nn.Sequential(
                self.shared,
                nn.Linear(512, 1)
            )
        else:
            self.shared = None
            self.actor = nn.Sequential(
                CnnBase(),
                nn.Linear(512, action_space_size)
            )
            self.critic = nn.Sequential(
                CnnBase(),
                nn.Linear(512, 1)
            )


# test_agent(Agent)

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

    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)

    Return: shape (t, env)
    '''
    T, _ = dones.shape

    # offset next values so we don't need to adjust indices
    next_values = torch.concat([values[1:], next_value])
    next_dones = torch.concat([dones[1:], next_done.unsqueeze(0)])

    # if done, then no next values/future rewards
    deltas = rewards + gamma * next_values * (1 - next_dones) - values

    advantage_t = deltas.clone().to(device)

    for t in reversed(range(1, T)):
        advantage_t[t - 1] = deltas[
            t - 1] + gamma * gae_lambda * (1.0 - dones[t]) * advantage_t[t]

    return advantage_t

test_compute_advantages(compute_advantages)

@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(batch_size: int,
                      minibatch_size: int) -> list[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    return indices.reshape(-1, minibatch_size).tolist()

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
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''
    obs = obs.reshape(-1, *obs_shape)
    logprobs = logprobs.reshape(-1)
    actions = actions.reshape(-1, *action_shape)
    advantages = advantages.reshape(-1)
    values = values.reshape(-1)
    returns = values + advantages

    list_of_indices = minibatch_indexes(batch_size=batch_size,
                                        minibatch_size=minibatch_size)

    minibatches = []
    for indices in list_of_indices:
        minibatches.append(
            Minibatch(
                obs=obs[indices],
                logprobs=logprobs[indices],
                actions=actions[indices],
                advantages=advantages[indices],
                returns=returns[indices],
                values=values[indices],
            ))
    return minibatches

def calc_policy_loss(probs: Union[Categorical, Normal], mb_action: t.Tensor,
                     mb_advantages: t.Tensor, mb_logprobs: t.Tensor,
                     clip_coef: float) -> t.Tensor:
    '''Return the policy loss, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.

    normalize: if true, normalize mb_advantages to have mean 0, variance 1

    '''

    # normalize advantages
    mb_advantages = (mb_advantages -
                     mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # get log probs of old model
    if isinstance(probs, Categorical):
        logprobs_old = probs.log_prob(mb_action)
    else:
        logprobs_old = probs.log_prob(mb_action).sum(axis=1)

    # calculated likelihood ratio
    ratio = torch.exp(logprobs_old - mb_logprobs)

    # calculate clipped likelihood ratio weighted mb_advantages
    clip_adv = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)

    # calculate policy loss (whatever is smaller, grad itself or clipped grad)
    loss = torch.min(ratio * mb_advantages, clip_adv * mb_advantages).mean()

    return loss

test_calc_policy_loss(calc_policy_loss)

def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor,
                             mb_returns: t.Tensor, v_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    v_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    loss = (critic(mb_obs) - mb_returns).pow(2).mean() / 2
    return loss * v_coef

test_calc_value_function_loss(calc_value_function_loss)

def calc_entropy_loss(probs: Union[Categorical, Normal], ent_coef: float):
    '''Return the entropy loss term.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return probs.entropy().mean() * ent_coef

test_calc_entropy_loss(calc_entropy_loss)

class PPOScheduler:

    def __init__(self, optimizer, initial_lr: float, end_lr: float,
                 num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.'''
        if self.n_step_calls <= self.num_updates:
            self.optimizer.param_groups[0][
                'lr'] = self.initial_lr + self.n_step_calls * (
                    self.end_lr - self.initial_lr) / self.num_updates
            self.n_step_calls += 1

def make_optimizer(agent: CnnAgent, num_updates: int, initial_lr: float,
                   end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, maximize=True)
    return optimizer, PPOScheduler(optimizer, initial_lr, end_lr, num_updates)


def train_ppo(args: PPOArgs):
    make_kwargs = dict(
        num_levels = 1,
        start_level = 0,
        render_mode = 'rgb_array',
    )
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %
        "\n".join([f"|{key}|{value}|" for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([
        make_env(
            args.env_id, args.seed + i, i, args.capture_video, run_name, **make_kwargs
        )
        for i in range(args.num_envs)
    ])
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    # assert isinstance(envs.single_action_space,
    #                   Discrete), "only discrete action space is supported"
    if isinstance(envs.single_action_space, gym.spaces.box.Box):
        action_space_type = 'Continuous'
        print('Continuous Action Space Type!')
    else:
        action_space_type = 'Discrete'
        print('Discrete Action Space')


    agent = CnnAgent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates,
                                            args.learning_rate, 0) 
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device, dtype=t.float32)
    next_done = torch.zeros(args.num_envs).to(device)
    for _ in range(num_updates):
        for i in range(0, args.num_steps):
            "YOUR CODE: Rollout phase (see detail #1)"

            global_step += args.num_envs

            obs[i] = next_obs  # we defined obs as 0's so we fill it in
            dones[i] = next_done  # we defined dones as 0's so we fill it in

            # critic provides values of states,
            # actor provides dist over actions
            with t.inference_mode():
                next_values = agent.critic(next_obs).flatten()
                
                
                if action_space_type == "Continuous":
                    means = agent.actor(next_obs)
                    probs = Normal(means, agent.logstd.exp().expand_as(means))
                    action = probs.sample()  # wait
                    logprobs[i] = probs.log_prob(action).sum(1)

                else:
                    logits = agent.actor(next_obs)
                    # pick an actual action though
                    probs = Categorical(logits=logits)
                    action = probs.sample()  # wait
                    logprobs[i] = probs.log_prob(action)
            
           

            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            # store all the results, casting from numpy where appropriate.
            rewards[i] = t.from_numpy(reward)
            
            actions[i] = action
            values[i] = next_values

            next_obs = t.from_numpy(next_obs).to(device, dtype=t.float32)
            next_done = t.from_numpy(done).to(device, dtype=t.float32)

            if "episode" in info.keys():
                for env in range(len(info["episode"])):
                    item = info["episode"][env]
                    if item is not None:
                        print(
                            f"global_step={global_step}, episodic_return={item['r']}, reward = {reward[0]}"
                        )
                        writer.add_scalar("charts/episodic_return",
                                            item["r"], global_step)
                        writer.add_scalar("charts/episodic_length",
                                            item["l"], global_step)
                        break

        next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(next_value, next_done, rewards, values,
                                        dones, device, args.gamma,
                                        args.gae_lambda)
        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:
                "YOUR CODE: compute loss on the minibatch and step the optimizer (not the scheduler). Do detail #11 (global gradient clipping) here using nn.utils.clip_grad_norm_."

                # ok so here we need to conpute the losses, let's do that.

                # for starter, let's get the "old probs"
                logits = agent.actor(mb.obs)
                # probs = Categorical(logits=logits)
                if action_space_type == "Continuous":
                    means = agent.actor(mb.obs)
                    probs = Normal(means, agent.logstd.exp().expand_as(means))
                else:
                    probs = Categorical(logits=logits)

                value_loss = calc_value_function_loss(agent.critic, mb.obs,
                                                      mb.returns, args.vf_coef)
                policy_loss = calc_policy_loss(probs, mb.actions,
                                               mb.advantages, mb.logprobs,
                                               args.clip_coef)
                entropy_loss = calc_entropy_loss(probs, args.ent_coef)

                # now get ready to do standard backward pass with losses
                total_loss = policy_loss - value_loss + entropy_loss
                #total_loss = policy_loss + entropy_loss # (experiment with no value loss)
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true -
                                                             y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [
                ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
            ]
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(),
                          global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var,
                          global_step)
        writer.add_scalar("charts/SPS",
                          int(global_step / (time.time() - start_time)),
                          global_step)
        if global_step % 10 == 0:
            print("steps per second (SPS):",
                  int(global_step / (time.time() - start_time)))
    envs.close()
    writer.close()

if MAIN:
    args = ppo_parse_args()
    train_ppo(args)