import numpy as np
import matplotlib.pyplot as plt

def Balayage(f,k,init,epsilon):
    #Résoudre f(x) = k
    sol = init
    while f(sol)>= k:
        sol=sol+epsilon
    return sol

def f_exp(x):
    return np.exp(4*(2-x))

def A(x,y):
    return (-4*f_exp(x)*(f_exp(y)+1))/((f_exp(x)+f_exp(y)+1)**2)

def B(x,y):
    return f_exp(x)/(f_exp(x)+f_exp(y)+1)

def f_true(x,y): 
    return (x-1)*A(x,y) + B(x,y)

def profit_Nash(x,y):
    return (x-1)*f_exp(x)/(f_exp(x)+f_exp(y)+1)

def f_Nash(x):
    return f_true(x,x)

def C(x):
    return -4*f_exp(x)/((2*f_exp(x)+1)**2)

def D(x): 
    return f_exp(x)/(2*f_exp(x)+1)

def profit_Mon(x): 
    return (x-1)*D(x)


def f_Mon(x):
    return (x-1)*C(x)+D(x)