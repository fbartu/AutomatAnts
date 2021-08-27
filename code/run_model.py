import sys
import os

from pandas.core.reshape.merge import merge
sys.path.append('~/research/AutomatAnts/code/')

import numpy as np
import math
#import time
import pandas as pd

from model import Model
from lattice import Lattice
from copy import deepcopy

import params

'''
NEED FOR AN ARGUMENT PARSER FUNCTION !!!
'''


path = params.path
filename = params.file_name

def create_instance():
	environment = Lattice(params.n_agents, params.width, params.height, params.nest_node, params.food)
	return Model(params.n_agents, params.recruitment, environment, params.n_steps, path, filename)
	
if params.n_runs > 1:

	def merge_runs(runs):
		results = pd.DataFrame.from_dict(runs[0][0].results)
		for i in range(1, len(runs)):
			results.append(pd.DataFrame.from_dict(runs[i][0].results))

		return results.sort_values('Time (s)')

	def average_runs(merged_result):
		sq = list(range(1, math.ceil(max(merged_result['Time (s)']))))

		idx = [np.where(np.logical_and(merged_result['Time (s)'] >= (i - 1),
		 merged_result['Time (s)'] < i))[0].tolist() for i in sq]

		nonempty = [i for i, x in enumerate(idx) if x != []]

		data = np.zeros((len(sq), merged_result.shape[1]))
		sd = deepcopy(data)

		for i in nonempty:
			data[i][:] = merged_result.iloc[idx[i], :].mean(axis = 0)
			sd[i][:] = merged_result.iloc[idx[i], :].std(axis = 0)
		
		data = pd.DataFrame(data, columns= merged_result.columns)
		data['Time (s)'] = sq
		sd = pd.DataFrame(sd, columns= merged_result.columns)
		sd['Time (s)'] = sq

		return data, sd

	if params.run_parallel:
		os.mkdir(path + 'results/' + filename)
		def run_parallel():
			model = create_instance()
			model.run()
			return deepcopy(model)
		
		import multiprocessing as mp
		pool = mp.Pool(mp.cpu_count())
		models = [pool.apply_async(run_parallel, ()) for i in range(params.n_runs)]
		models = [i.get() for i in models]
		# results = [i.results for i in models]
		pool.close()

		for i in range(len(models)):
			models[i].data2json(folder = filename + '/', filename = filename + '_' + str(i))

	else:
		models = []
		for run in range(params.n_runs):
			model = create_instance()
			model.run()
			model.data2json(folder = filename + '/', filename = filename + '_' + str(run))
			models.append(model)

	# AVERAGE DATAFRAME FOR THE RESULTS
	mrg = merge_runs(models) # single data frame for all results
	avg, sd = average_runs(mrg)
	avg.to_csv(path + filename + '_average.csv')
	sd.to_csv(path + filename + '_sd.csv')

else:
	model = create_instance()
	model.run()
	model.data2json(filename = filename)