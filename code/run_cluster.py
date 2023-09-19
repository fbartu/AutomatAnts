import Model
import json
from multiprocessing import Pool, cpu_count
import os, time
import numpy as np
import pandas as pd
from functions import argparser

parameters = argparser()   

food_condition = parameters.pop('food_condition')
results_path = parameters.pop('results_path')
filename = parameters.pop('filename')
runs = parameters.pop('runs')
alpha, beta, gamma = parameters.pop('alpha'), parameters.pop('beta'), parameters.pop('gamma')

def run_model(i):
    np.random.seed(int((os.getpid() * (i/np.random.random())* time.time()) % 123456789))
    m = Model.Model(alpha=alpha, beta=beta, gamma=gamma,
                    food_condition= food_condition, **parameters)

    m.run()
    # result = {'T': m.T, 'N': m.N, 'I': m.I, 'SiOut': m.o, 'pos': m.position_history}
    result = pd.DataFrame({'T': m.T, 'N': m.N, 'I': m.I, 'SiOut': m.o})
    result['Frame'] = (result['T'] // 0.5) * 0.5
    df = result.groupby('Frame').mean().reset_index()
    df = df.drop(columns = ['T'])
    path = '%s%s_%s.json' % (results_path, filename, i)
    df.to_csv(path)
    # with open(path, 'w') as f:
    #     json.dump(result, f)
        
        
if __name__ == '__main__':

    num_processes = cpu_count()
    pool = Pool(processes=num_processes)

    pool.map(run_model, range(runs))

    pool.close()
    pool.join()