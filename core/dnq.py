# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import core.prices as pr


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, fct1_dim=128, fct2_dim=128, alpha):
        super(DQN, self).__init__()
        
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.fct1_dim = fct1_dim
        self.fct2_dim = fct2_dim
        
        self.layer1 = nn.Linear(self.n_observations, self.fct1_dim)
        self.layer2 = nn.Linear(self.fct1_dim, self.fct2_dim)
        self.layer3 = nn.Linear(self.fct2_dim, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), alpha)
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
    def __init__(self, nb_players=2, alpha=0.125, beta=10**(-5), delta=0.95, pN=None, pC=None, binary_demand=False, k=1000, batch_size=32):
        self.m = 15
        self.n = nb_players
        ##memory k : we can now change it
        self.k = k
        self.mem_count = 0
        self.batch_size = batch_size
        
        n_observations=(self.m**(self.k*self.n))
        
        self.binary_demand = binary_demand
        if pN == None or pC == None:
            self.pC, self.pN = self.getPrices()
        else:
            self.pC, self.pN = pC, pN
        self.Xi = 0.1
        self.p1 = self.pN-self.Xi*(self.pC-self.pN)
        self.pm = self.pC+self.Xi*(self.pC-self.pN)
        self.A = np.zeros(self.m)
        for i in range(self.m):
            self.A[i] = self.p1 + i*(self.pm-self.p1)/(self.m-1)

        # combinations = itertools.product(self.A, repeat=self.n)
        # self.S = [list(combination) for combination in combinations]

        


# A CHANGER
        self.Q_eval = DQN(n_observations=n_observations,n_actions=self.m,alpha=self.alpha)

#####################################

        self.epsilon = 1
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.find_ind = {self.A[i]: chr(i + 65) for i in range(len(self.A))}

        self.s_t = np.zeros((self.mem_size, *n_observations), np.float32)
        self.a_t = np.zeros((self.mem_size, *n_observations), np.float32)
        self.reward = np.zeros((self.mem_size, *n_observations), np.float32)
        self.s_t1 = np.zeros((self.mem_size, *n_observations), np.float32)

    def getPrices(self):
        prices = pr.PriceOptimizer(
            nb_players=self.n, binary_demand=self.binary_demand)
        collusion_price, nash_price = prices()
        return collusion_price, nash_price

    def get_next_action(self, observation):

        if np.random.random() < 1-self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.randint(self.m)
            
        return action
        ################

    def get_reward(self, q, p, c):
        return (p-c)*q
    
    def transition(self, state, action, reward, state_1):
        index = self.mem_count % self.k
        self.s_t[index] = state
        self.s_t1[index] = state_1
        self.a_t[index] = action
        self.reward_memory = reward
        
        self.mem_count += 1
        

    def updateQ(self, t):  # upateQ
        if self.mem_count < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_count, self.k)
        batch = np.random.choice(self.k,self.batch_size,replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        action_batch = torch.tensor(self.a_t[batch]).to(self.Q_eval.device)
        state_batch = torch.tensor(self.s_t[batch]).to(self.Q_eval.device)
        state_batch_1 = torch.tensor(self.s_t1[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward[batch]).to(self.Q_eval.device)
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(state_batch_1)
        
        q_target = reward_batch + self.delta*torch.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = np.exp(-self.beta*t)



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
