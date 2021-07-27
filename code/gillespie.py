import numpy as np
import random
from scipy.stats import rv_discrete
from agent import *
import params
import statistics


class ParameterMetrics():

	def __init__(self, environment, pos):

		self.environment = environment
		self.pos = pos


	def interactions(self):
		unique_values = list(set(self.pos))
		counts = []
		for i in unique_values:
			x = self.pos.count(i)
			counts.append(x)

		#return sum(np.array(counts) - 1)
		return sum(np.array(counts) > 1)
	
	def connectivity(self):
		if len(self.pos):
			pos = list(set(self.pos)) # eliminate duplicates
			k_length = []
			branch = []
			while len(pos) > 0:
				current_path = [pos.pop(0)]
				while len(current_path):
					# get neighbors
					neighbors = np.array(self.environment.grid.get_neighbors(current_path[-1]))
					# get only the neighbors that are available in path
					idx = np.where([(n[0], n[1]) in pos for n in neighbors])
					branch.extend(list(map(tuple, neighbors[idx])))
					
					while len(branch):
						current_path.append(branch.pop(0))
						pos.remove(current_path[-1])
						neighbors = np.array(self.environment.grid.get_neighbors(current_path[-1]))
						idx = np.where([(n[0], n[1]) in pos for n in neighbors])
						branch.extend(list(map(tuple, neighbors[idx])))
					
					k_length.append(len(current_path))
					current_path = []

			return statistics.mean(k_length)
			#return k_length
		else:
			return 0

	
	def efficiency(self, tfood):
		if not len(self.environment.food_cluster):
			self.environment.cluster_food()

		food_found = np.array(list(map(lambda x: x == True, tfood['Flag'])))
		
		for p in list(self.environment.food_cluster.keys()):
			patch = []

			for i in self.environment.food_cluster[p]:
				food_visited = np.array(list(map(lambda x: x == i, tfood['Pos'])))
				
				idx = int(np.where(np.logical_and(food_found == True, food_visited == True))[0])
				patch.append(tfood['Time'][idx])

			self.environment.food_cluster[p] = {'x': [x[0] for x in self.environment.food_cluster[p]],
			'y': [y[1] for y in self.environment.food_cluster[p]],
			 't': patch}

		return self.environment.food_cluster


class GillespieAlgorithm():

	def __init__(self, agents, environment):

		# Variables to measure
		self.T = [0]
		self.K = [0]
		self.N = [0]
		self.I = [0]
		self.F = [0]

		self.agents = agents
		self.environment = environment
		#self.environment.cluster_food()
		self.metrics = ParameterMetrics(self.environment, None)

		# debugging
		self.sample = []

		# states of the population
		self.population = {State.WAITING: [len(self.agents)], 
		State.EXPLORING: [0],
		State.EXPLORING_FOOD: [0],
		State.RECRUITING: [0],
		State.RECRUITING_FOOD: [0],
		State.DEAD: [0], # track number of recovered individuals
		Tag.INFORMED: [0] # track number of informed individuals
		}
		
		self.r = [params.alpha] * len(self.agents)
		self.r_norm = np.array(self.r) / sum(self.r)
		self.R_t = sum(self.r)
		

		self.rng_t = random.random() # random number to sample the time
		self.rng_action = random.random() # random number to determine if action occurs

		self.time = abs(np.log(self.rng_t)/self.R_t)

		# time flags on food pick up
		self.tfood = {'Pos': [self.environment.initial_node],
		'Flag': [False], 'Time': [self.time]}


	def actualize_population(self):

		list_of_states = dir(State())[:6]
		states = [self.agents[i].state for i in list(self.environment.out_nest.keys())]
		indices = np.where([(i not in states) for i in list_of_states])[0]
		non_used_states = list(map(list_of_states.__getitem__, indices))

		for i in list(set(states)):
			self.population[i].append(states.count(i))

		for i in non_used_states:
			self.population[getattr(State(), i)].append(0)

	
	def step(self):
		
		#sample = rv_discrete(values=(list(self.agents.keys()), self.r_norm)).rvs(size=1)
		sample = rv_discrete(values=(list(range(len(self.agents))), self.r_norm)).rvs(size=1)


		if self.rng_action < float(self.r_norm[sample]):
			self.sample.append(sample) # for debugging and tracking agents' actions

			# get the index of the ant performing an action
			idx = int(sample)

			# do action & report if food is found (@bool flag)
			flag = self.agents[idx].action(self.environment, self.agents)
			self.agents[idx].actualize_path()

			# actualize population states
			#self.actualize_population()
			#self.population[State.WAITING] = len(self.environment.waiting_ants)

			tmp_pos = []
			tmp_info = []
			for a in list(self.environment.out_nest.keys()):
				tmp_pos.append(self.agents[a].pos)
				tmp_info.append(self.agents[a].tag) 

			self.metrics.pos = tmp_pos
			# actualize number of informed ants
			self.population[Tag.INFORMED].append(tmp_info.count(Tag.INFORMED))

			# actualize food collection times
			self.tfood['Pos'].append(self.agents[idx].pos)
			self.tfood['Flag'].append(flag)
			self.tfood['Time'].append(self.time)

			# get dynamics of the population
			self.retrieve_data(self.metrics.interactions(), self.metrics.connectivity())

			# actualize rates
			self.r[idx] = self.agents[idx].r_i
			self.r_norm = np.array(self.r) / sum(self.r)
			self.R_t = sum(self.r)

		# get rng for next iteration
		self.rng_t = random.random()
		self.rng_action = random.random()

		# get time for next iteration
		self.time += abs(np.log(self.rng_t)/self.R_t)

	def retrieve_data(self, ints, connectivity):
		self.N.append(len(self.environment.out_nest))
		self.T.append(self.time)
		self.I.append(ints)
		self.K.append(connectivity)
		self.F.append(self.environment.food_in_nest)
		# food nest / patches evolution !
		# exploring / recruiting ants !	