import core.interactions as interact
import pandas as pd
import pickle
import numpy as np
import multiprocessing

from joblib import Parallel, delayed


def process(beta, nb_games):
    results = []
    for i in range(10):
        np.random.seed(i)
        game = interact.Interaction(beta=beta)
        result = game(nb_iterations=nb_games)
        result['beta'] = beta
        result['index'] = i+1
        results.append(result)
    results_df = pd.concat(results, ignore_index=True)
    with open(f'data/simulation_beta_{beta}.pkl', 'wb') as f:
        pickle.dump(results_df, f)


betas = [7.5*10**(-3), 5*10**(-3), 2.5*10**(-3), 10**(-3), 7.5*10**(-4), 5*10**(-4), 2.5*10**(-4), 10**(-4), 7.5*10**(-5), 5 *
         10**(-5), 2.5*10**(-5), 10**(-5), 7.5*10**(-6), 5*10**(-6), 2.5*10**(-6), 10**(-5), 7.5*10**(-7), 5*10**(-7), 2.5*10**(-7), 10**(-7)]


no_process = multiprocessing.cpu_count()
print(f'number of cores detected :{no_process}')

Parallel(n_jobs=no_process)(delayed(process)(beta, 10*10**(6))
                            for beta in betas)
