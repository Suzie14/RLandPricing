import numpy as np
import pandas as pd

class GridBR: 

    def __init__(self,m=15,Xi=0.1,pN = 1.47,pC=1.92,a=[0,2,2],mu=1/4,c=[1,1]):
        self.m = m
        self.pN = pN
        self.pC = pC
        self.Xi = Xi
        self.p1 = self.pN-self.Xi*(self.pC-self.pN)
        self.pm = self.pC+self.Xi*(self.pC-self.pN)
        self.A = np.zeros(self.m)
        for i in range (self.m):
            self.A[i] = self.p1 + i*(self.pm-self.p1)/(self.m-1)
        self.a = a
        self.mu = mu
        self.c = c
        
    
    def profit_i(self,p):
        if len(self.quantity(p)) != len(self.margin(p)):
            raise ValueError("Les listes doivent être de même taille pour effectuer la multiplication élément par élément.")
    
        result = [elem1 * elem2 for elem1, elem2 in zip(self.quantity(p), self.margin(p))]
        return result[0]
    
    def margin(self, p):
        if len(p) != len(self.c):
            raise ValueError("Les listes doivent être de même taille pour effectuer la soustraction élément par élément.")
    
        result = [elem1 - elem2 for elem1, elem2 in zip(p, self.c)]
        return result
    


    
    def f(self,p): 
        prime_p = np.array([0]+p)
        return np.exp((self.a-prime_p)/self.mu)
    
    def quantity(self,p):
        q = np.zeros(2)  
        quant = self.f(p)
        q[0] = quant[1]/sum(quant)
        q[1] = quant[2]/sum(quant)
        return q
    
    
        

    def grid(self): 
        # Vecteurs discrets p_i et p_j
        choice_p_i = self.A
        choice_p_j = self.A

        # Créer un tableau à double entrée
        result_table = [[self.profit_i([p_i, p_j]) for p_j in choice_p_j] for p_i in choice_p_i]
        for row in result_table:
            print(row)
    
        return result_table
    

    def __call__(self):
        return self.A 
    


