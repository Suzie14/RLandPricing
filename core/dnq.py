# -*- coding: utf-8 -*-
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import core.prices as pr


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.memory)), batch_size)
        return [self.memory[idx] for idx in indices]

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_states, n_actions, fct1_dim=128, fct2_dim=128):
        super(DQN, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.fct1_dim = fct1_dim
        self.fct2_dim = fct2_dim

        self.layer1 = nn.Linear(self.n_states, self.fct1_dim)
        self.layer2 = nn.Linear(self.fct1_dim, self.fct2_dim)
        self.layer3 = nn.Linear(self.fct2_dim, self.n_actions)

        self.loss = nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        action = self.layer3(x)
        return action


class Agent:
    def __init__(self, nb_players=2, alpha=0.125, beta=10**(-5), delta=0.95, pN=None, pC=None, binary_demand=False, max_mem_size=1000, batch_size=32):
        self.m = 15
        self.n = nb_players

        # memory k : we can now change it
        self.mem_size = max_mem_size
        self.mem_count = 0
        self.batch_size = batch_size

        # state doesn't depend directly of the past states
        self.n_states = (self.m**(self.n*self.mem_size))

        # hyperparameters
        self.epsilon = 1
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.binary_demand = binary_demand
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

        self.Q_eval = DQN(n_states=self.n_states,
                          n_actions=self.m, alpha=self.alpha)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.states = np.zeros((self.mem_size, *self.n_states), np.float32)
        self.a_ind = None
        self.reward = None
        self.s_t1 = None
        self.p = None

    def _get_prices(self):
        prices = pr.PriceOptimizer(
            nb_players=self.n, binary_demand=self.binary_demand)
        collusion_price, nash_price = prices()
        return collusion_price, nash_price

    def get_next_action(self):
        if np.random.random() < 1 - self.epsilon:
            state = torch.tensor(self.s_t, dtype=torch.float32)
            q_values = self.Q_eval(state)
            return q_values.argmax().item()
        else:
            return np.random.choice(self.A)

    def get_reward(self, q, p, c):
        return (p-c)*q

    def transition(self, st_1):
        index = self.mem_count % self.k
        self.states[index] = st_1

        self.mem_count += 1

    def updateQ(self, p, q, c, t):
        reward = self.get_reward(q, p, c)

        state = torch.tensor(self.s_t, dtype=torch.float32)
        q_values = self.Q_eval(state)

        target = reward + self.delta * q_values.max()
        loss = nn.MSELoss()(q_values[self.a_ind], target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.p = self.A[self.a_ind]
        self.s_t = self.s_t1
        self.epsilon = np.exp(-self.beta * t)


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

    # prendre action et  renvoie la demande
    # renvoie q, q', p et p'
