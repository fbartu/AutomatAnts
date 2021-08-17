import sys
import os
sys.path.append('~/research/AutomatAnts/code/')

#import numpy as np
#import time

from model import Model
from lattice import Lattice
from copy import deepcopy

import params

path = params.path
filename = params.file_name

def create_instance():
	environment = Lattice(params.n_agents, params.width, params.height, params.nest_node, params.food)
	return Model(params.n_agents, params.recruitment, environment, params.n_steps, path, filename)
	
if params.n_runs > 1:
	if params.run_parallel:
		os.mkdir(path + 'results/' + filename)
		def run_parallel():
			model = create_instance()
			model.run()
			return deepcopy(model)
		
		def average_runs(runs):
			pass


		import multiprocessing as mp
		pool = mp.Pool(mp.cpu_count())
		models = [pool.apply_async(run_parallel, ()) for i in range(params.n_runs)]
		models = [i.get() for i in models]
		results = [i.results for i in models]
		pool.close()

		for i in range(len(models)):
			models[i].data2json(folder = filename + '/', filename = filename + '_' + str(i))

		
	else:
		for run in range(params.n_runs):
			model = create_instance()
			model.run()

else:
	model = create_instance()
	model.run()