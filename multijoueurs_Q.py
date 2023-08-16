import numpy as np 
import itertools

import resultsPriceandProfits as res 

class Agent: 
    
    def __init__(self,nb_player=2,alpha=0.125,beta=10**(-5),delta=0.95, pN=None, pC=None):
        self.m = 15
        self.n = nb_player
        self.k = 1
        if pN == None or pC == None:
            self.pC, self.pN = self.getPrices()
        else:
            self.pC, self.pN = pC, pN
        self.Xi = 0.1
        self.p1 = self.pN-self.Xi*(self.pC-self.pN)
        self.pm = self.pC+self.Xi*(self.pC-self.pN)
        self.A = np.zeros(self.m)
        for i in range (self.m):
            self.A[i] = self.p1 + i*(self.pm-self.p1)/(self.m-1)
        
        combinations = itertools.product(self.A, repeat=self.n)
        self.S = [list(combination) for combination in combinations]
        
                    
        self.Q = np.random.uniform(size = (self.m,self.m**(self.n*self.k)),
                                   low=-0.5,
                                   high =0.5)
        
        self.epsilon = 1
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.t = 0 
        
        self.s_ind = None # On veut dans un certain sens que l'index soit complémentaire pour tous les joueurs
        self.s_t = None
        self.a_ind = None
        self.p = None
        self.s_ind1 = None
        self.s_t1 = None
    
    def getPrices(self):
        prices = res.PriceOptimizer(nb_players=self.n)
        collusion_price, nash_price = prices()
        return collusion_price, nash_price
    
    def get_next_action(self):
        if np.random.random() < 1-self.epsilon: 
            return self.Q[:,self.s_ind].argmax()
        else: 
            return np.random.randint(self.m)
        
    def get_reward(self, q, p,c): 
        return (p-c)*q
        
    def updateQ(self, p, q, c, t):#upateQ
        reward = self.get_reward(q,p,c)
    
        self.Q[self.a_ind, self.s_ind] = (1-self.alpha)*self.Q[self.a_ind, self.s_ind] + self.alpha*( reward + self.delta*self.Q[:,self.s_ind1].max())
        #print("Start")
        #print(reward)
        #print(self.Q[self.a_ind, self.s_ind1].max())
        #print(self.delta)
        #print(self.alpha)
        
        
        self.p = self.A[self.a_ind]
        #self.s_t = self.s_t1
        self.s_ind = self.s_ind1
        self.epsilon = np.exp(-self.beta*t)
        
    
    
    def find_index(self, S, s_t):
        index = -1
        for test in range(len(S)):
            match = True
            for player in range(len(s_t)):
                if S[test][player] != s_t[player]:
                    match = False
                    break
            if match:
                index = test
                break
        return index

    
class Env: 
    def __init__(self, nb_players=2,a_value=2,mu_value=1/4,c_value=1,binary_demand=False): 
        self.nb_players = nb_players
        self.a_value = a_value
        self.mu_value = mu_value
        self.c_value = c_value
        
        self.a = np.concatenate(([0], np.full(nb_players, a_value)))
        self.c = np.full(nb_players, c_value)
    
    def f(self,p): 
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
        min_indices = np.where(p == np.min(p))[0]  # Find indices of minimum prices
    
        # Distribute demand equally among players with the lowest prices
        num_min_prices = len(min_indices)
        if num_min_prices > 0:
            q[min_indices] = 1 / num_min_prices
    
        return q

    def __call__(self,p):
        return [self.quantity(p), p, self.c]
    #prendre action et  renvoie la demande
    #renvoie q, q', p et p'


