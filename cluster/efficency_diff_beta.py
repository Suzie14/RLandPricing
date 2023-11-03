import core.interactions as interact
import pandas as pd
import pickle
import numpy as np
import multiprocessing

from joblib import Parallel, delayed


def process(beta, nb_games):
    results = []
    get_all_Qs = []
    for i in range(10):
        np.random.seed(i)
        game = interact.Interaction(beta=[beta,10**(-5)])
        result = game(nb_iterations=nb_games)
        result['beta'] = beta
        result['index'] = i+1
        results.append(result)
        getQs = game.getQs()
        getQs['beta'] = beta
        getQs['index'] = i+1
        get_all_Qs.append(getQs)
    results_df = pd.concat(results, ignore_index=True)
    Qs_df = pd.concat(get_all_Qs, ignore_index=True)

    with open(f'data/data_simbis/simulation_beta_{beta}.pkl', 'wb') as f:
        pickle.dump(results_df, f)
    with open(f'data/data_Qbis/Q_values_beta_{beta}.pkl', 'wb') as l:
        pickle.dump(Qs_df, l)
    


betas = [7.5*10**(-3), 5*10**(-3), 2.5*10**(-3), 10**(-3), 7.5*10**(-4), 5*10**(-4), 2.5*10**(-4), 10**(-4), 7.5*10**(-5), 5 *
         10**(-5), 2.5*10**(-5), 10**(-5), 7.5*10**(-6), 5*10**(-6), 2.5*10**(-6), 10**(-5), 7.5*10**(-7), 5*10**(-7), 2.5*10**(-7), 10**(-7)]


no_process = multiprocessing.cpu_count()
print(f'number of cores detected :{no_process}')

Parallel(n_jobs=no_process)(delayed(process)(beta, 10**(7))
                            for beta in betas)
