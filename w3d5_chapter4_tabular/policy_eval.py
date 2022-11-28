#%%
# from gettext import find
from typing import Optional, Union
import numpy as np
import gym
import gym.spaces
import gym.envs.registration
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import importlib
import utils
importlib.reload(utils)

MAIN = __name__ == "__main__"
Arr = np.ndarray
# %%
class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        if terminal is None:
            self.terminal = np.array([], dtype=int)
        else:
            self.terminal = terminal
        (self.T, self.R) = self.build()

    def build(self):
        '''
        Constructs the T and R tensors from the dynamics of the environment.
        Outputs:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        '''
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        '''
        Computes the distribution over possible outcomes for a given state
        and action.
        Inputs:
            state : int (index of state)
            action : int (index of action)
        Outputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        '''
        raise NotImplementedError

    def render(pi: Arr):
        '''
        Takes a policy pi, and draws an image of the behavior of that policy,
        if applicable.
        Inputs:
            pi : (num_actions,) a policy
        Outputs:
            None
        '''
        raise NotImplementedError

    def out_pad(self, states: Arr, rewards: Arr, probs: Arr):
        '''
        Inputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        Outputs:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including
                           probability zero outcomes.)
        '''
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return (out_s, out_r, out_p)
#%%[markdown]
#### Environments
# %%
class Toy(Environment):
    def dynamics(self, state: int, action: int):
        (S0, SL, SR) = (0, 1, 2)
        LEFT = 0
        num_states = 3
        num_actions = 2
        assert 0 <= state < self.num_states and 0 <= action < self.num_actions
        if state == S0:
            if action == LEFT:
                (next_state, reward) = (SL, 1)
            else:
                (next_state, reward) = (SR, 0)
        elif state == SL:
            (next_state, reward) = (0, 0)
        elif state == SR:
            (next_state, reward) = (0, 2)
        return (np.array([next_state]), np.array([reward]), np.array([1]))

    def __init__(self):
        super().__init__(3, 2)
# %%
if MAIN:
    toy = Toy()
    print(toy.T)
    print(toy.R)
# %%
toy.T[0, :, :] # Starting from 0 you can go left or right deterministically
# %%
class Norvig(Environment):
    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        move = self.actions[action]
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))
        out_probs = np.zeros(self.num_actions) + 0.1
        out_probs[action] = 0.7
        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]
        for (i, s_new) in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue
            new_state = state_index(s_new)
            if new_state in self.walls:
                out_states[i] = state
            else:
                out_states[i] = new_state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]
        return (out_states, out_rewards, out_probs)

    def render(self, pi: Arr):
        assert len(pi) == self.num_states
        emoji = ["⬆️", "➡️", "⬇️", "⬅️"]
        grid = [emoji[act] for act in pi]
        grid[3] = "🟩"
        grid[7] = "🟥"
        grid[5] = "⬛"
        print(str(grid[0:4]) + "\n" + str(grid[4:8]) + "\n" + str(grid[8:]))

    def __init__(self, penalty=-0.04):
        self.height = 3
        self.width = 4
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.dim = (self.height, self.width)
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1])
        super().__init__(num_states, num_actions, start=8, terminal=terminal)
#%%[markdown]
#### Policies
# %%
def policy_eval_numerical(
    env: Environment, pi: Arr, gamma=0.99, eps=1e-08
) -> Arr:
    '''
    Numerically evaluates the value of a deterministic policy by iterating the 
    Bellman equation
    Inputs:
        env: Environment
        pi : shape (num_states,) - The deterministic policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
    Outputs:
        value : float (num_states,) - The value function for policy pi
    '''
    num_states = env.num_states
    num_actions = env.num_actions
    values = np.zeros(num_states)
    pi_padded = np.zeros((num_states, num_actions))
    pi_padded[np.arange(num_states), pi] = 1
    pi_broadcast = np.broadcast_to(
        pi_padded.T, 
        (num_states, num_actions, num_states)
    ).T
    error = np.inf
    while error >= eps:
        new_values = (
            pi_broadcast * env.T * (env.R + gamma * values)
        ).sum(axis=(1, 2))
        error = np.max(np.abs(new_values - values))
        values = new_values
    return values


if MAIN:
    utils.test_policy_eval(policy_eval_numerical, exact=False)
# %%
def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    num_states = env.num_states
    num_actions = env.num_actions
    pi_padded = np.zeros((num_states, num_actions), dtype='int')
    arange = np.arange(num_states, dtype='int')
    pi_padded[arange, pi] = 1
    pi_broadcast = np.broadcast_to(
        pi_padded.T, 
        (num_states, num_actions, num_states)
    ).T
    trans_mat = (pi_broadcast * env.T).sum(axis=1)
    rew_mat = (pi_broadcast * env.R).sum(axis=1)
    diag = np.diag((trans_mat @ rew_mat.T))
    id_mat = np.identity(num_states)
    i_minus_trans = id_mat - gamma * trans_mat
    return np.linalg.inv(i_minus_trans) @ diag

if MAIN:
    utils.test_policy_eval(policy_eval_exact, exact=True)
# %%
def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    '''
    Inputs:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : 
        vector (num_states,) of actions representing a new policy obtained via policy iteration
    '''
    return (env.T * (env.R + gamma * V)).sum(axis=-1).argmax(axis=1)

if MAIN:
    utils.test_policy_improvement(policy_improvement)
# %%
def find_optimal_policy(env: Environment, gamma=0.99):
    '''
    Inputs:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    '''
    pi = np.zeros(env.num_states, dtype='int')
    while True:
        v_pi = policy_eval_exact(env, pi, gamma)
        pi_new = policy_improvement(env, v_pi, gamma)
        if (pi == pi_new).all():
            break
        pi = pi_new
    return pi


if MAIN:
    utils.test_find_optimal_policy(find_optimal_policy)
    penalty = -0.04
    norvig = Norvig(penalty)
    pi_opt = find_optimal_policy(norvig, gamma=0.96)
    norvig.render(pi_opt)
#%%
if MAIN:
    minority_discount = find_optimal_policy(toy, gamma=0.6)
    majority_discount = find_optimal_policy(toy, gamma=0.4)
    print(minority_discount, majority_discount)
# %%
