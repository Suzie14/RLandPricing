import core.qlearning2 as q
import numpy as np
import time
import pandas as pd


class Interaction:
    def __init__(self, nb_players=2, alpha=0.125, beta=10**(-5), delta=0.95, pN=None, pC=None, binary_demand=False, doubleQ=False, a_values=[2, 2], mu_value=1/4, c_values=[1, 1]):
        self.nb_players = nb_players
        # Hyperparameters can be different for every agent
        if not isinstance(alpha, float):
            if len(alpha) != nb_players:
                raise ValueError(
                    "alpha must be a float or a liste of lenghth of the number of players")
            else:
                self.alpha = alpha
        else:
            self.alpha = [alpha]*nb_players

        if not isinstance(beta, float):
            if len(beta) != nb_players:
                raise ValueError(
                    "beta must be a float or a liste of lenghth of the number of players")
            else:
                self.beta = beta
        else:
            self.beta = [beta]*nb_players

        if not isinstance(delta, float):
            if len(delta) != nb_players:
                raise ValueError(
                    "delta must be a float or a liste of lenghth of the number of players")
            else:
                self.delta = delta
        else:
            self.delta = [delta]*nb_players

        self.pN = pN
        self.pC = pC
        self.binary_demand = binary_demand
        self.doubleQ = doubleQ
        self.a_values = a_values
        self.c_values = c_values
        self.mu_value = mu_value

    def _init_agents(self):
        self.agents = [q.Agent(nb_players=self.nb_players, alpha=self.alpha[i], beta=self.beta[i], delta=self.delta[i], pN=self.pN, pC=self.pC, binary_demand=self.binary_demand, doubleQ=self.doubleQ)
                       for i in range(self.nb_players)]

    def _init_env(self):
        self.env = q.Env(nb_players=self.nb_players, binary_demand=self.binary_demand,
                         a_values=self.a_values, mu_value=self.mu_value, c_values=self.c_values)

    def _random_init(self):
        # Initialization of prices p0 (done directly in each agent)
        for agent in self.agents:
            agent.p = np.random.choice(agent.A)

        # Initialization of state
        s_t = self.env([agent.p for agent in self.agents])[1]
        for agent in self.agents:
            agent.s_t = s_t

        s_ind = self.agents[0].find_index(self.agents[0].s_t)
        for agent in self.agents:
            agent.s_ind = s_ind

    def _single_game(self, t):
        # Actions and state at t+1
        for agent in self.agents:
            agent.a_ind = agent.get_next_action()

        s_t1 = [agent.A[agent.a_ind] for agent in self.agents]
        for agent in self.agents:
            agent.s_t1 = s_t1

        s_ind1 = self.agents[0].find_index(self.agents[0].s_t1)
        for agent in self.agents:
            agent.s_ind1 = s_ind1

        ret = self.env(s_t1)
        quant, price, cost = ret

        rewards = quant*price-quant*cost
        prices = price
        epsilon_values = [agent.epsilon for agent in self.agents]

        for i, agent in enumerate(self.agents):
            agent.updateQ(reward=rewards[i])
            agent.transition(t)

        return rewards, prices, epsilon_values

    def __call__(self, nb_iterations):
        self._init_agents()
        self._init_env()

        iterations_values = []
        rewards_values = []
        prices_values = []
        epsilon_values = []

        time_values = []

        last_100_rewards = []
        last_100_prices = []
        last_100_epsilon = []

        self._random_init()

        for t in range(nb_iterations+1):

            # to print 10 iterations at the end
            prints = round(nb_iterations/10)

            # to have 1000 points on iterations at the end
            add_data = round(nb_iterations/1000)

            inter_start = time.time()
            if t % (prints) == 0:
                print("t:", t)

            rewards, prices, epsilon = self._single_game(t)

            if t % (add_data) == 0:
                iterations_values.append(t)
                rewards_values.append(rewards)
                prices_values.append(prices)
                epsilon_values.append(epsilon)

            if t > nb_iterations - 100:
                last_100_rewards.append(rewards)
                last_100_prices.append(prices)
                last_100_epsilon.append(epsilon)

            inter_end = time.time()
            time_values.append(inter_end-inter_start)
            if t % (prints) == 0:
                print('average CPU', np.mean(time_values))
                print('averages profits', np.mean(rewards_values, axis=0))
                print('averages prices', np.mean(prices_values, axis=0))
                print('epsilon', epsilon_values[-1])

        mean_last_rewards = np.mean(last_100_rewards, axis=0)
        mean_last_prices = np.mean(last_100_prices, axis=0)
        mean_last_epsilons = np.mean(last_100_epsilon, axis=0)

        iterations_values.append('last 100 iterations mean')
        rewards_values.append(mean_last_rewards)
        prices_values.append(mean_last_prices)
        epsilon_values.append(mean_last_epsilons)

        vals = {'Iteration': iterations_values, 'Rewards': rewards_values,
                'Prices': prices_values, 'Epsilon': epsilon_values}
        df = pd.DataFrame(vals)

        return df
