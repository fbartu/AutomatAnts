from Model import *
import json

''' PARAMETERS '''
runs = 100
results_path = "../results/STO_SIMS/"
    
m = Model()

for i in range(runs):
    m.run()
    # result = {'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut}
    result = {'T': m.T, 'N': m.N, 'I': m.I, 'SiIn': m.n, 'SiOut': m.o}
    path = '%sfood_%s.json' % (results_path, i)
    with open (path, 'w') as f:
        json.dump(result, f)
    del m
    m = Model()