import numpy as np
from itertools import product

import core.prices as pr


class StateMapper:
    def __init__(self, nb_players, num_bins,
                 lower, upper, xi):

        self.nb_players = nb_players

        self.bins = np.round(np.linspace(lower - xi*(upper-lower),
                                         upper + xi*(upper-lower),
                                         num_bins), 4)

        self.bin_idx = [str(k).zfill(2) for k in np.arange(0, num_bins)]

        self.mapping_dict = dict(zip(self.bins, self.bin_idx))

        self.all_state_dict = self._all_states_mapper()

    def _all_states_mapper(self):
        all_combs = list(product(self.bin_idx, repeat=self.nb_players))
        joined_combs = [''.join(tu) for tu in all_combs]

        return dict(zip(joined_combs, np.arange(0, len(joined_combs))))

    def __call__(self, state):
        mapped_bin = {}
        sum_mapped_bin = ''
        for i, p in enumerate(state):
            mapped_bin[i+1] = self.mapping_dict[p]
            sum_mapped_bin += mapped_bin[i+1]

        return self.all_state_dict[sum_mapped_bin]


class Agent:

    def __init__(self, nb_players=2, alpha=0.05, beta=10**(-5), delta=0.95, pN=None, pC=None, binary_demand=False, doubleQ=False):
        self.m = 15
        self.n = nb_players
        self.k = 1
        self.binary_demand = binary_demand
        if pN == None or pC == None:
            self.pC, self.pN = self._get_prices()
        else:
            self.pC, self.pN = pC, pN
        self.Xi = 0.1
        self.p1 = self.pN-self.Xi*(self.pC-self.pN)
        self.pm = self.pC+self.Xi*(self.pC-self.pN)
        self.A = np.round(np.linspace(self.p1, self.pm, self.m), 4)

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
        self.Q_count = np.zeros((self.m, self.m**(self.n*self.k)))

        self.epsilon = 1
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.find_ind = StateMapper(nb_players=self.n, num_bins=self.m,
                                    lower=self.pN, upper=self.pC, xi=self.Xi)

        self.s_ind = None
        self.s_t = None
        self.a_ind = None
        self.p = None
        self.s_ind1 = None
        self.s_t1 = None

        # When we want to stop when it convergs
        self.past_pairs = {}
        self.done = False
        self.cnt = 0

    def _get_prices(self):
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

    def updateQ(self, reward):  # upateQ

        if self.doubleQ:
            if np.random.random() < 0.5:
                self.Q1[self.a_ind, self.s_ind] = (1-self.alpha)*self.Q1[self.a_ind, self.s_ind] + self.alpha*(
                    reward + self.delta*self.Q2[self.Q1[:, self.s_ind1].argmax(), self.s_ind1])
            else:
                self.Q2[self.a_ind, self.s_ind]
        else:
            q_value = self.Q[self.a_ind, self.s_ind]
            # Create a memory view of the relevant column
            q_slice = self.Q[:, self.s_ind1]
            max_value = q_slice.max()  # Compute the max value
            diff = reward + self.delta*max_value - q_value
            q_value += self.alpha*diff
            self.Q[self.a_ind, self.s_ind] = q_value
        self.Q_count[self.a_ind, self.s_ind] += 1

    def transition(self, t):
        self.p = self.A[self.a_ind]
        self.s_ind = self.s_ind1
        self.epsilon = np.exp(-self.beta*t)

    def find_index(self, s_t):
        s_ind = self.find_ind(s_t)
        return s_ind

    def final_state(self, iterat=10**5):
        if self.done == False:
            if self.s_ind in self.past_pairs and self.past_pairs[self.s_ind] == self.a_ind:
                self.cnt += 1

            else:
                self.cnt = 0
                self.past_pairs[self.s_ind] = self.a_ind

            if self.cnt >= iterat:
                self.done = True


class Env:
    def __init__(self, nb_players=2, a_values=[2, 2], mu_value=1/4, c_values=[1, 1], binary_demand=False):
        self.nb_players = nb_players
        self.a_values = a_values
        self.mu_value = mu_value
        self.c_values = c_values
        self.binary_demand = binary_demand

        self.a = np.array([0] + a_values)
        self.dict_quant = {}

    def f(self, p):
        prime_p = np.array([0]+p)
        return np.exp((self.a-prime_p)/self.mu_value)

    def quantity(self, p):
        if tuple(p) not in self.dict_quant.keys():
            quant = self.f(p)
            q = quant/np.sum(quant)
            self.dict_quant[tuple(p)] = q[1:]
        return self.dict_quant[tuple(p)]

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
            return [self.binary_quantity(p), p, self.c_values]
        else:
            return [self.quantity(p), p, self.c_values]
