import numpy as np
from scipy.optimize import minimize

class PriceOptimizer:
    def __init__(self, nb_players=5, a_value=2, mu_value=1/4, c_value=1):
        self.nb_players = nb_players
        self.a_value = a_value
        self.mu_value = mu_value
        self.c_value = c_value
        
        self.a = np.concatenate(([0], np.full(nb_players, a_value)))
        self.c = np.full(nb_players, c_value)

    def f(self, p):
        prime_p = np.concatenate(([0], p))
        return np.exp((self.a - prime_p) / self.mu_value)

    def quantity(self, p):
        quant = self.f(p)
        q = np.zeros(len(quant) - 1)
        for i in range(len(quant) - 1):
            q[i] = quant[i + 1] / sum(quant)
        return q

    def profit(self, p):
        return np.multiply(self.quantity(p), np.subtract(p, self.c))

    def negative_profit(self, p):
        return -self.profit(p)[0]

    def price_constraint(self, p):
        return np.mean(p) - p[0]
    
    def individual_profit(self, player_price, other_prices):
        all_prices = np.insert(other_prices, 0, player_price)
        return self.profit(all_prices)[0]

    def CollusionPrice(self):
        initial_p = np.ones(self.nb_players)

        constraints = [{'type': 'eq', 'fun': self.price_constraint}]

        result = minimize(self.negative_profit, initial_p, bounds=[(0, None)] * self.nb_players, constraints=constraints, method='SLSQP')

        collusion_profit = -result.fun
        collusion_prices = result.x

        return collusion_profit, collusion_prices

    def NashPrice(self):
        # Initialize all players with initial prices
        current_prices = np.ones(self.nb_players)

        # Calculate initial profits
        current_profits = np.array([self.individual_profit(current_prices[i], np.delete(current_prices, i)) for i in range(self.nb_players)])

        # Initialize a flag to track if any player's profit can be increased
        profit_increased = True

        # Define a small step for adjusting prices
        price_step = 0.01

        while profit_increased:
            profit_increased = False
    
            for i in range(self.nb_players):
                # Try increasing the price for player i
                new_prices = np.copy(current_prices)
                new_prices[i] += price_step
        
                # Calculate profits with the new prices
                new_profits = np.array([self.individual_profit(new_prices[j], np.delete(new_prices, j)) for j in range(self.nb_players)])
        
                # Check if any player's profit can be increased
                if new_profits[i] > current_profits[i]:
                    current_prices = new_prices
                    current_profits = new_profits
                    profit_increased = True
                    break

        nash_profit = current_profits[0]
        nash_prices = current_prices

        return nash_profit, nash_prices
    
    def __call__(self):
        collusion_prices = self.CollusionPrice()[1][0]
        nash_prices = self.NashPrice()[1][0]
        return collusion_prices, nash_prices


