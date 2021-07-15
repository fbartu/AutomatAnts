from mesa.space import NetworkGrid
import networkx as nx
import numpy as np
import random
from scipy.stats import rv_discrete
from scipy.stats.morestats import _anderson_ksamp_midrank
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

		# states of the population
		self.population = {State.WAITING: len(self.agents), 
		State.EXPLORING: 0,
		State.EXPLORING_FOOD: 0,
		State.RECRUITING: 0,
		State.RECRUITING_FOOD: 0,
		Tag.INFORMED: 0
		}

		# tasks performed by the population
		self.tasks = {State.WAITING: self.population[State.WAITING], 
		State.EXPLORING: self.population[State.EXPLORING] + self.population[State.EXPLORING_FOOD],
		State.RECRUITING: self.population[State.RECRUITING] + self.population[State.RECRUITING_FOOD]
		}

		self.r = [params.alpha] * len(State.WAITING)
		self.r_norm = np.array(self.r) / sum(self.r)
		self.R_t = sum(self.r)
		

		self.rng_t = random.random() # random number to sample the time
		self.rng_action = random.random() # random number to determine if action occurs

		self.time = abs(np.log(self.rng_t)/self.R_t)
	
	def actualize_tasks(self):
		self.tasks = {State.WAITING: self.population[State.WAITING], 
		State.EXPLORING: self.population[State.EXPLORING] + self.population[State.EXPLORING_FOOD],
		State.RECRUITING: self.population[State.RECRUITING] + self.population[State.RECRUITING_FOOD]
		}

	def step(self, agents):
		
		sample = rv_discrete(values=(list(agents.keys()), self.r_norm)).rvs(size=1)

		if self.rng_action < float(self.r_norm[sample]):

			# get the index of the ant performing an action
			idx = int(sample)

			# do action
			informed = agents[idx].action(self.environment, self.agents)

			# actualize number of informed ants
			self.population[Tag.INFORMED] += informed

			# actualize population tasks
			if agents[idx].state != State.WAITING:
				self.population[agents[idx].state] =+ 1
				self.actualize_tasks()

			# actualize rates
			self.r[idx] = agents[idx].r_i
			self.population[agents[idx].state] =+ 1
			self.r_norm = np.array(self.r) / sum(self.r)
			self.R_t = sum(self.r)

			pos_list = [self.agents[a].pos for a in list(self.environment.out_nest.keys())]

			self.retrieve_data(self.get_interactions(pos_list))
		
		# get rng for next iteration
		self.rng_t = random.random()
		self.rng_action = random.random()

		# get time for next iteration
		self.time = abs(np.log(self.rng_t)/self.R_t)

	def get_interactions(self, pos):
		unique_values = list(set(pos))
		counts = [unique_values.count(x) for x in unique_values]
		return sum(np.array(counts) - 1)
		# return sum(np.array(counts) > 1)
	
	def get_connectivity(self, pos):
		pos = list(set(pos)) # eliminate duplicates
		k_length = []
		branch = []
		while len(pos) > 0:
			current_path = [pos.pop(0)]
			while len(current_path):
				neighbors = self.environment.grid.get_neighbours(current_path[-1])
				available_neighbors = list(p in pos for p in neighbors)
				neighbors = list(i for True in available_neighbors)
				branch.append(dict(zip(neighbors, available_neighbors)))
				idx = np.where(available_neighbors)

				if len(neighbors):
					branch.append(neighbors)
				else:
					k_length.append(len(current_path))

		for p in list(range(len(pos))):
			self.environment


	def retrieve_data(self, ints, connectivity):
		self.N.append(self.population[len(self.agents) - State.WAITING])
		self.T.append(self.time)
		self.I.append(ints)
		self.K.append(connectivity)
		

		### ++++++++++++++++++++++++++++++++++++++++++++ ###

		positions.append(agent.pos)
		#### Connectivity ####
		self.k1 = 0
		self.k2 = 0
		for i in range(len(ShortPaths.path[0])):
			if ((ShortPaths.path[0][i]) in positions):
				self.k1 += 1

		for i in range(len(ShortPaths.path[1])):
			if ((ShortPaths.path[1][i]) in positions):
				self.k2 += 1

		if self.k1 > self.k2:
			self.k = self.k1
		else:
			self.k = self.k2

		#Now we normalize the different rates for each ant
		#and save the value and the id in two diferent vectors
		r_norm = []
		id_list = []

		#Save agents to sample
		object_list = []

		for agent in self.agent_buffer(shuffled=False):
			r_norm.append(agent.r_i / R_t)
			id_list.append(agent.unique_id)
			object_list.append(agent)

		#Sample from a discrete distribution,which ant will try to move
		sample = rv_discrete(values=(id_list,r_norm)).rvs(size=1)
		int_sample = sample.astype(int)
		agent_sampled = object_list[int_sample[0]]
	
		#Gillespie step
		rand1 = random.random()
		rand2 = random.random()

		#Gillespie process
		#Differnet gillespie time definitions
		#t_gillespie = (1.0/R_t)*np.log(1.0/rand1)
		t_gillespie = abs(np.log(rand1)/R_t)
		

		