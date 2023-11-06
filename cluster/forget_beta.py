import core.interactions as interact
import pandas as pd
import pickle
import numpy as np
import multiprocessing

from joblib import Parallel, delayed


def process(beta, nb_games, path1, path2):
    results = []
    get_all_Qs = []
    for i in range(10):
        np.random.seed(i)
        game = interact.Interaction(beta=beta)
        result = game(nb_iterations=nb_games)
        result['beta'] = beta
        result['index'] = i+1
        results.append(result)
        getQs = game.getQs()
        if not isinstance(beta, float): 
            getQs['beta'] = beta[0]
        else: 
            getQs['beta'] = beta
        getQs['index'] = i+1
        get_all_Qs.append(getQs)
    results_df = pd.concat(results, ignore_index=True)
    Qs_df = pd.concat(get_all_Qs, ignore_index=True)

    with open(path1, 'wb') as f:
        pickle.dump(results_df, f)
    with open(path2, 'wb') as l:
        pickle.dump(Qs_df, l)
    


betas, paths = [10**(-6), [10**(-6),10**(-5)]]
paths1 = ['data/data_sim/simulation_beta_10e-06.pkl','data/data_simbis/simulation_beta_10e-06.pkl']
paths2 = ['data/data_Q/Q_values_beta_10e-06.pkl','data/data_Qbis/Q_values_beta_10e-06.pkl']


no_process = multiprocessing.cpu_count()
print(f'number of cores detected :{no_process}')

Parallel(n_jobs=no_process)(delayed(process)(10**(-6), 10**(7), 'data/data_sim/simulation_beta_10e-06.pkl', 'data/data_Q/Q_values_beta_10e-06.pkl'))
Parallel(n_jobs=no_process)(delayed(process)([10**(-6),10**(-5)], 10**(7), 'data/data_simbis/simulation_beta_10e-06.pkl', 'data/data_Qbis/Q_values_beta_10e-06.pkl'))

