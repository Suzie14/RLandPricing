# -*- coding: utf-8 -*-
import numpy as np

import core.prices as pr


class Agent:

    def __init__(self, nb_players=2, alpha=0.125, beta=10**(-5), delta=0.95, pN=None, pC=None, binary_demand=False, doubleQ=False):
        self.m = 15
        self.n = nb_players
        self.k = 1
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

        self.doubleQ = doubleQ

        if self.doubleQ:
            self.Q1 = np.random.uniform(size=(self.m, self.m**(self.n*self.k)),
                                        low=-0.5,
                                        high=0.5)

            self.Q2 = np.random.uniform(size=(self.m, self.m**(self.n*self.k)),
                                        low=-0.5,
                                        high=0.5)

        else:
            self.Q = np.random.uniform(size=(self.m, self.m**(self.n*self.k)),
                                       low=-0.5,
                                       high=0.5)

        self.epsilon = 1
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.t = 0

        self.find_ind = {self.A[i]: chr(i + 65) for i in range(len(self.A))}

        # On veut dans un certain sens que l'index soit complémentaire pour tous les joueurs
        self.s_ind = None
        self.s_t = None
        self.a_ind = None
        self.p = None
        self.s_ind1 = None
        self.s_t1 = None

    def getPrices(self):
        prices = pr.PriceOptimizer(
            nb_players=self.n, binary_demand=self.binary_demand)
        collusion_price, nash_price = prices()
        return collusion_price, nash_price

    def get_next_action(self):
        if self.doubleQ:
            Q = self.Q1 + self.Q2
        else:
            Q = self.Q

        if np.random.random() < 1-self.epsilon:
            return Q[:, self.s_ind].argmax()
        else:
            return np.random.randint(self.m)

    def get_reward(self, q, p, c):
        return (p-c)*q

    def updateQ(self, p, q, c, t):  # upateQ
        reward = self.get_reward(q, p, c)

        if self.doubleQ:
            if np.random.random() < 0.5:
                self.Q1[self.a_ind, self.s_ind] = (1-self.alpha)*self.Q1[self.a_ind, self.s_ind] + self.alpha*(
                    reward + self.delta*self.Q2[self.Q1[:, self.s_ind1].argmax(), self.s_ind1])
            else:
                self.Q2[self.a_ind, self.s_ind]
        else:
            self.Q[self.a_ind, self.s_ind] = (
                1-self.alpha)*self.Q[self.a_ind, self.s_ind] + self.alpha*(reward + self.delta*self.Q[:, self.s_ind1].max())
        # print("Start")
        # print(reward)
        # print(self.Q[self.a_ind, self.s_ind1].max())
        # print(self.delta)
        # print(self.alpha)

        self.p = self.A[self.a_ind]
        # self.s_t = self.s_t1
        self.s_ind = self.s_ind1
        self.epsilon = np.exp(-self.beta*t)

    def find_index(self, s_t):
        strongstring = ''.join(
            [str(self.find_ind.get(price, '')) for price in s_t])
        s_ind = sum((ord(char) - 65) * (15 ** (self.n - 1 - i))
                    for i, char in enumerate(strongstring))
        return s_ind


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
