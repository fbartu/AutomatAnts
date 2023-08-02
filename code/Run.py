import Model
import json
import sys, getopt

argv = sys.argv[1:]

opts, args = getopt.getopt(argv, 'n:d:x:f:m:j:p:',
                           ['nruns=', 'directory=', 'filename=', 
                            'food=', 'movement=', 'memory=', 'parameters='])

parameters = {}

for opt, arg in opts:
    if opt in ('-n', '--nruns'):
        runs = int(arg)
        
    elif opt in ('-d', '--directory'):
        results_path = arg
        
    elif opt in ('-x', '--filename'):
        filename = arg
        
    elif opt in ('-f', '--food'):
        parameters['food_condition'] = arg
        
    elif opt in ('-m', '--movement'):
        parameters['default_movement'] = arg
        
    elif opt in ('-p', '--parameters'):
        plist = arg.split(',')
        for p in plist:
            x = p.split('=')
            if x[0] == 'alpha':
                Model.alpha = float(x[1])
            elif x[0] == 'beta':
                Model.beta = float(x[1])
            elif x[0] == 'gamma':
                Model.gamma = float(x[1])
            else:
                print('Unknown parameter', x[0])
        
    elif opt in ('-j', '--states'):
        parameters['memory'] = arg

''' PARAMETERS '''

if not 'filename' in globals():
    filename = 'food'
if not 'runs' in globals():
    runs = 100  
if not 'results_path' in globals():
    results_path = "../results/STO_SIMS/"
if not 'food_condition' in parameters:
    food_condition = 'sto_1'
else:
    food_condition = parameters.pop('food_condition')
    
if __name__ == '__main__':
    m = Model.Model(alpha = Model.alpha, beta = Model.beta, gamma = Model.gamma,
                    food_condition = food_condition, **parameters)
    print(m.rates)

    for i in range(runs):
        m.run()
        # result = {'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut}
        # result = {'T': m.T, 'N': m.N, 'I': m.I, 'SiIn': m.n, 'SiOut': m.o, 'pos': m.position_history}
        result = {'T': m.T, 'N': m.N, 'I': m.I, 'SiOut': m.o, 'pos': m.position_history}
        path = '%s%s_%s.json' % (results_path, filename, i)
        with open (path, 'w') as f:
            json.dump(result, f)
        del m
        m = Model.Model(food_condition = food_condition, **parameters)