import numpy as np
import matplotlib.pyplot as plt

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
    prime_p = np.array([0]+p)
    return np.exp((a-prime_p)/mu)
    
def quantity(p,a,mu):
    q = np.zeros(2)  
    quant = f(p,a,mu)
    q[0] = quant[1]/sum(quant)
    q[1] = quant[2]/sum(quant)
    return q

def profit(p,a,mu,c): 
    return multiply_lists(quantity(p,a,mu), subtract_lists(p, c))

