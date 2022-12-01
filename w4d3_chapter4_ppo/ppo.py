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
import wandb

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
        super().__init__()
        obs_shape = np.array(
            (envs.num_envs, ) + envs.single_action_space.shape
        ).prod().astype(int)
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
    assert isinstance(next_value, t.Tensor)
    assert isinstance(next_done, t.Tensor)
    assert isinstance(rewards, t.Tensor)
    assert isinstance(values, t.Tensor)
    assert isinstance(dones, t.Tensor)
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
    obs_flat = obs.reshape((batch_size,) + obs_shape)
    act_flat = actions.reshape((batch_size,) + action_shape)
    probs_flat = logprobs.reshape((batch_size,) + action_shape)
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
    Need to minimise this

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
    Need to maximise this

    ent_coef: 
        The coefficient for the entropy loss, which weights its contribution to the overall loss. 
        Denoted by c_2 in the paper.
    '''
    return probs.entropy().mean() * ent_coef

if MAIN and RUNNING_FROM_FILE:
    tests.test_calc_entropy_loss(calc_entropy_loss)
# %%
class PPOScheduler:
    def __init__(self, optimizer: optim.Adam, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''
        Implement linear learning rate decay so that after num_updates calls to step, 
        the learning rate is end_lr.
        '''
        lr = (
            self.initial_lr + 
            (self.end_lr - self.initial_lr) * self.n_step_calls / self.num_updates
        )
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.n_step_calls += 1

def make_optimizer(
    agent: Agent, num_updates: int, initial_lr: float, end_lr: float
) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, maximize=True)
    scheduler = PPOScheduler(
        optimizer=optimizer, initial_lr=initial_lr, end_lr=end_lr, num_updates=num_updates
    )
    return optimizer, scheduler
# %%
@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

def train_ppo(args: PPOArgs):
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
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" 
        for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) 
        for i in range(args.num_envs)
    ])
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
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
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    for _ in range(num_updates):
        for i in range(0, args.num_steps):
            "YOUR CODE: Rollout phase (see detail #1)"
            global_step += 1
            curr_obs = next_obs
            done = next_done
            with t.inference_mode():
                logits = agent.actor(curr_obs).detach()
                q_values = agent.critic(curr_obs).detach().squeeze(-1)
            prob = Categorical(logits=logits)
            action = prob.sample()
            logprob = prob.log_prob(action)
            next_obs, reward, next_done, info = envs.step(action.numpy())
            next_obs = t.tensor(next_obs, device=device)
            next_done = t.tensor(next_done, device=device)
            obs[i] = curr_obs
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = t.tensor(reward, device=device)
            dones[i] = t.tensor(done, device=device)
            values[i] = q_values

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
        with t.inference_mode():
            next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda
        )
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
                probs = Categorical(logits=agent.actor(mb.obs))
                value_loss = calc_value_function_loss(agent.critic, mb.obs, mb.returns, args.vf_coef)
                policy_loss = calc_policy_loss(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                entropy_loss = calc_entropy_loss(probs, args.ent_coef)
                loss = policy_loss + entropy_loss - value_loss
                loss.backward()
                nn.utils.clip_grad_norm(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if global_step % 10 == 0:
            print("steps per second (SPS):", int(global_step / (time.time() - start_time)))
    envs.close()
    writer.close()

if MAIN:
    if RUNNING_FROM_FILE:
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = ppo_parse_args()
    train_ppo(args)