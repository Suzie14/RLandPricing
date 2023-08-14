import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify
from scipy.optimize import minimize

def multiply_lists(lst1, lst2):
    if len(lst1) != len(lst2):
        raise ValueError("Les listes doivent être de même taille pour effectuer la multiplication élément par élément.")
    
    result = [elem1 * elem2 for elem1, elem2 in zip(lst1, lst2)]
    return result

def subtract_lists(lst1, lst2):
    if len(lst1) != len(lst2):
        raise ValueError("Les listes doivent être de même taille pour effectuer la soustraction élément par élément.")
    
    result = [elem1 - elem2 for elem1, elem2 in zip(lst1, lst2)]
    return result


def f(p,a,mu): 
    prime_p = np.concatenate(([0], p))
    return np.exp((a-prime_p)/mu)
    
def quantity(p,a,mu):  
    quant = f(p,a,mu)
    q = np.zeros(len(quant)-1)
    for i in range (len(quant)-1):
        q[i] = quant[i+1]/sum(quant) 
        #such indexation because a0 exist in quant
    return q

def profit(p,a,mu,c): 
    return multiply_lists(quantity(p,a,mu), subtract_lists(p, c))





def CollusionPrice(nb_players, a_value=2, mu_value=1/4, c_value=1):
    # Create the repeated value arrays
    a = np.concatenate(([0], np.full(nb_players, a_value)))
    c = np.full(nb_players, c_value)

    # Define the profit function to be maximized
    def negative_profit(p):
        return - profit(p, a, mu_value, c)[0]

    # Initial guess for the values of p
    initial_p = np.ones(nb_players)

    def price_constraint(p):
        return np.mean(p) - p[0]

    constraints = [
        {'type': 'eq', 'fun': price_constraint}
    ]

    # Use optimization to find the maximum profit
    result = minimize(negative_profit, initial_p, bounds=[(0, None)] * nb_players, constraints=constraints, method='SLSQP')

    max_profit = -result.fun
    optimal_p = result.x

    print("Collusion Profit:", max_profit)
    print("Collusion Prices:", optimal_p)
    return max_profit, optimal_p



def NashPrice(nb_players, a_value=2, mu_value=1/4, c_value=1):

    # Create the repeated value arrays
    a = np.concatenate(([0], np.full(nb_players, a_value)))
    c = np.full(nb_players, c_value)

    # Define the individual profit function for a player i
    def individual_profit(player_price, other_prices):
        return (player_price-c[0]) * np.exp(4 * (2 - player_price)) / (np.exp(4 * (2 - player_price)) + np.sum(np.exp(4 * (2 - other_prices))) + 1)

    # Initialize all players with initial prices
    current_prices = np.ones(nb_players)

    # Calculate initial profits
    current_profits = np.array([individual_profit(current_prices[i], np.delete(current_prices, i)) for i in range(nb_players)])

    # Initialize a flag to track if any player's profit can be increased
    profit_increased = True

    # Define a small step for adjusting prices
    price_step = 0.01

    while profit_increased:
        profit_increased = False
    
        for i in range(nb_players):
            # Try increasing the price for player i
            new_prices = np.copy(current_prices)
            new_prices[i] += price_step
        
            # Calculate profits with the new prices
            new_profits = np.array([individual_profit(new_prices[j], np.delete(new_prices, j)) for j in range(nb_players)])
        
            # Check if any player's profit can be increased
            if new_profits[i] > current_profits[i]:
                current_prices = new_prices
                current_profits = new_profits
                profit_increased = True
                break

    nash_profits = np.sum(current_profits)
    nash_prices = current_prices


    print("Nash Profits:", nash_profits)
    print("Nash Equilibrium Prices:", nash_prices)
    return nash_profits, nash_prices
