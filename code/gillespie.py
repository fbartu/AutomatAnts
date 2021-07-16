import numpy as np
import random
from scipy.stats import rv_discrete
from agent import *
import params
import statistics


class GillespieAlgorithm():

	def __init__(self, agents, environment):

		# Variables to measure
		self.T = [0]
		self.K = [0]
		self.N = [0]
		self.I = [0]

		self.agents = agents
		self.environment = environment

		# debugging
		self.sample = []

		# states of the population
		self.population = {State.WAITING: len(self.agents), 
		State.EXPLORING: 0,
		State.EXPLORING_FOOD: 0,
		State.RECRUITING: 0,
		State.RECRUITING_FOOD: 0,
		State.DEAD: 0, # track number of recovered individuals
		Tag.INFORMED: 0 # track number of informed individuals
		}

		'''
		# tasks performed by the population
		self.tasks = {State.WAITING: self.population[State.WAITING], 
		State.EXPLORING: self.population[State.EXPLORING] + self.population[State.EXPLORING_FOOD],
		State.RECRUITING: self.population[State.RECRUITING] + self.population[State.RECRUITING_FOOD]
		}

		'''
		
		self.r = [params.alpha] * len(self.agents)
		self.r_norm = np.array(self.r) / sum(self.r)
		self.R_t = sum(self.r)
		

		self.rng_t = random.random() # random number to sample the time
		self.rng_action = random.random() # random number to determine if action occurs

		self.time = abs(np.log(self.rng_t)/self.R_t)


	def actualize_population(self):

		states = [self.agents[i].state for i in list(self.environment.out_nest.keys())]
		for i in list(set(states)):
			self.population[i] = states.count(i)

	
	def step(self):
		
		#sample = rv_discrete(values=(list(self.agents.keys()), self.r_norm)).rvs(size=1)
		sample = rv_discrete(values=(list(range(len(self.agents))), self.r_norm)).rvs(size=1)


		if self.rng_action < float(self.r_norm[sample]):
			self.sample.append(sample)

			# get the index of the ant performing an action
			idx = int(sample)

			# do action
			informed = self.agents[idx].action(self.environment, self.agents)

			# actualize number of informed ants
			#self.population[Tag.INFORMED] += informed

			# actualize population tasks
			self.actualize_population()
			self.population[State.WAITING] = len(self.environment.waiting_ants)
			#self.actualize_tasks()

			# actualize rates
			self.r[idx] = self.agents[idx].r_i
			self.r_norm = np.array(self.r) / sum(self.r)
			self.R_t = sum(self.r)

			#pos_list = [self.agents[a].pos for a in list(self.environment.out_nest.keys())]

			self.retrieve_data(0, 0)
			#self.retrieve_data(self.get_interactions(pos_list), self.get_connectivity(pos_list))
		# get rng for next iteration
		self.rng_t = random.random()
		self.rng_action = random.random()

		# get time for next iteration
		self.time += abs(np.log(self.rng_t)/self.R_t)

	def get_interactions(self, pos):
		unique_values = list(set(pos))
		counts = [unique_values.count(x) for x in unique_values]
		return sum(np.array(counts) - 1)
		# return sum(np.array(counts) > 1)
	
	def get_connectivity(self, pos):
		if len(pos):
			pos = list(set(pos)) # eliminate duplicates
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
						'''
						sometimes pos.remove() raises an error !!
						'''
						pos.remove(current_path[-1])
						neighbors = np.array(self.environment.grid.get_neighbors(current_path[-1]))
						idx = np.where([(n[0], n[1]) in pos for n in neighbors])
						branch.extend(list(map(tuple, neighbors[idx])))
					
					k_length.append(len(current_path))
					current_path = []

			#return statistics.mean(k_length)
			return k_length
		else:
			return [0]


	def retrieve_data(self, ints, connectivity):
		self.N.append(len(self.environment.out_nest))
		self.T.append(self.time)
		self.I.append(ints)
		self.K.append(connectivity)
		# informed ants !
		# times of food collection !
		# food nest / patches evolution !
		# exploring / recruiting ants !	