# -*- coding: utf-8 -*-
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque

import core.prices as pr


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

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


class Agent:
    def __init__(self, nb_players=2, alpha=0.125, beta=10**(-5), delta=0.95, pN=None, pC=None, binary_demand=False, max_mem_size=1000, batch_size=32):
        self.m = 15
        self.n = nb_players

        # memory k : we can now change it
        self.mem_size = max_mem_size
        # self.mem_count = 0
        # self.batch_size = batch_size

        # state doesn't depend directly of the past states
        self.n_states = (self.m**(self.n*self.mem_size))

        # hyperparameters
        self.epsilon = 1
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.tau = 0.005

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

    def _get_reward(self, q, p, c):
        return (p-c)*q

    def transition(self, st_1):
        index = self.mem_count % self.k
        self.states[index] = st_1

        self.mem_count += 1

    def updateQ(self, p, q, c, t):
        reward = self._get_reward(q, p, c)

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


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


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
