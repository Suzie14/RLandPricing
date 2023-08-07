import numpy as np 

class Agent: 
    
    def __init__(self,alpha=0.125,beta=10**(-5),delta=0.95):
        self.m = 15
        self.n = 2
        self.k = 1
        self.pN = 1.47
        self.pM = 1.92
        self.Xi = 0.1
        self.p1 = self.pN-self.Xi*(self.pM-self.pN)
        self.pm = self.pM+self.Xi*(self.pM-self.pN)
        self.A = np.zeros(self.m)
        for i in range (self.m):
            self.A[i] = self.p1 + i*(self.pm-self.p1)/(self.m-1)
        #self.S = np.zeros((self.m**(self.n*self.k),2))
        
        self.S = []
        for i in range (len(self.A)):
            for j in range (len(self.A)): 
                self.S.append([self.A[i],self.A[j]])
        
        #for i in range (len(self.A)):
         #   for j in range (len(self.A)): 
             #   for k in range (len(self.S)):
              #      self.S[k][0] = self.A[i]
                #    self.S[k][1] = self.A[j]
                    
        self.Q1 = np.random.uniform(size = (self.m,self.m**(self.n*self.k)),
                                   low=-0.5,
                                   high =0.5)
        
        self.Q2 = np.random.uniform(size = (self.m,self.m**(self.n*self.k)),
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
       
            
    def get_next_action(self):
        Q = self.Q1 + self.Q2
        if np.random.random() < 1-self.epsilon: 
            return Q[:,self.s_ind].argmax()
        else: 
            return np.random.randint(self.m)
        
    def get_reward(self, q, p,c): 
        return (p-c)*q
        
    def updateQ(self, p, q, c, t):#upateQ 
        reward = self.get_reward(q,p,c)

        if np.random.random() < 0.5: 
            self.Q1[self.a_ind, self.s_ind] = (1-self.alpha)*self.Q1[self.a_ind, self.s_ind] + self.alpha*( reward + self.delta*self.Q2[self.Q1[:,self.s_ind1].argmax(),self.s_ind1])
        else : 
            self.Q2[self.a_ind, self.s_ind] = (1-self.alpha)*self.Q2[self.a_ind, self.s_ind] + self.alpha*( reward + self.delta*self.Q1[self.Q2[:,self.s_ind1].argmax(),self.s_ind1])
        
        #print("Start")
        #print(reward)
        #print(self.Q[self.a_ind, self.s_ind1].max())
        #print(self.delta)
        #print(self.alpha)
        
        
        self.p = self.A[self.a_ind]
        #self.s_t = self.s_t1
        self.s_ind = self.s_ind1
        self.epsilon = np.exp(-self.beta*t)
        
    
    
    def find_index(self,S,s_t): 
        index = -1
        for test in range(len(S)):
            if S[test][0] == s_t[0] and S[test][1] == s_t[1]:
               index = test 
        return index

    
class Env: 
    def __init__(self): 
        self.mu = 1/4
        self.a = np.array([0,2,2])
        self.c = [1,1]
    
    def f(self,p): 
        prime_p = np.array([0]+p)
        return np.exp((self.a-prime_p)/self.mu)
    
    def quantity(self, p):
        q = np.zeros(2)  
        quant = self.f(p)
        q[0] = quant[1]/sum(quant)
        q[1] = quant[2]/sum(quant)
        return q

    def __call__(self,p):
        return [self.quantity(p), p, self.c]
    #prendre action et  renvoie la demande
    #renvoie q, q', p et p'
