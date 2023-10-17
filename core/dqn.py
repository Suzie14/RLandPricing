# -*- coding: utf-8 -*-
import torch
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque

import core.prices as pr


class ReplayMemory:

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DeepAgent:
    def __init__(self, nb_players=2, alpha=1e-4, eps_start=0.9, eps_end=0.05, eps_decay=1000, gamma=0.99, pN=None, pC=None, binary_demand=False, m=15):
        # Hyperparameters
        self.alpha = alpha
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = 128
        self.gamma = gamma
        self.tau = 0.005

        # Size of action and state spaces
        self.m = m
        self.n_state = nb_players
        self.n = nb_players

        # Demand function
        self.binary_demand = binary_demand

        # Action space's definition
        if pN == None or pC == None:
            self.pC, self.pN = self._get_prices()
        else:
            self.pC, self.pN = pC, pN
        self.Xi = 0.1
        self.p1 = self.pN-self.Xi*(self.pC-self.pN)
        self.pm = self.pC+self.Xi*(self.pC-self.pN)
        self.A = np.zeros(self.m)
        for i in range(self.m):
            self.A[i] = self.p1 + i*(self.pm-self.p1)/(self.m-1)

        # Selected action and associated index
        self.p = None
        self.a_ind = None

        # State and reward, given by the environment
        self.state = None
        self.reward = None

        # Device and Transition
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))

        # Neural networks
        self.policy_net = DQN(self.n_state, self.m).to(self.device)
        self.target_net = DQN(self.n_state, self.m).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer and memory
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.alpha, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def _get_prices(self):
        prices = pr.PriceOptimizer(
            nb_players=self.n, binary_demand=self.binary_demand)
        collusion_price, nash_price = prices()
        return collusion_price, nash_price

    def get_next_action(self, state, t):
        '''Return the index of the selected action'''
        sample = random.random()
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * t / self.eps_decay)
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(15)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_network(self):
        self.target_net_state_dict = self.target_net.state_dict()
        self.policy_net_state_dict = self.policy_net.state_dict()
        for key in self.policy_net_state_dict:
            self.target_net_state_dict[key] = self.policy_net_state_dict[key]*self.tau
            + self.target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(self.target_net_state_dict)


class Env:
    def __init__(self, nb_players=2, a_value=2, mu_value=1/4, c_value=1, binary_demand=False):
        self.nb_players = nb_players
        self.a_value = a_value
        self.mu_value = mu_value
        self.c_value = c_value

        self.a = np.concatenate(([0], np.full(nb_players, a_value)))
        self.c = np.full(nb_players, c_value)

        self.binary_demand = binary_demand

    def f(self, p):
        prime_p = np.array([0]+p)
        return np.exp((self.a-prime_p)/self.mu_value)

    def quantity(self, p):
        quant = self.f(p)
        q = np.zeros(len(quant) - 1)
        for i in range(len(quant) - 1):
            q[i] = quant[i + 1] / sum(quant)
        return q

    def binary_quantity(self, p):
        q = np.zeros(len(p))
        # Find indices of minimum prices
        min_indices = np.where(p == np.min(p))[0]

        # Distribute demand equally among players with the lowest prices
        num_min_prices = len(min_indices)
        if num_min_prices > 0:
            q[min_indices] = (self.a[min_indices+1] -
                              np.min(p)) / num_min_prices

        return q

    def __call__(self, p):
        if self.binary_demand:
            return [self.binary_quantity(p), p, self.c]
        else:
            return [self.quantity(p), p, self.c]
