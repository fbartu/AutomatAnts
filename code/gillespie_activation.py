
import numpy as np
import random

from mesa.time import RandomActivation
from collections import OrderedDict
from scipy.stats import rv_discrete

# mypy
from typing import Dict, Iterator, List, Optional, Union

from model import *
from agent import *


class ActivationStep(RandomActivation):

	"""
	A scheduler which activates 1 agent per step sampling from a discrete
	distribution, and a gillespie step.
	Assumes that all agents have a step() method.
	"""

	def __init__(self, model):
		super().__init__(model)

		self.model = model
		self.steps = 0
		self.time = 0.0

		self._agents: Dict[int, Agent] = OrderedDict()

	def add(self, agent):
		"""
		Add an Agent object to the schedule
		Args:
		agent: An Agent to be added to the schedule.
		"""
		self._agents[agent.unique_id] = agent

	def remove(self, agent):
		"""
		Remove all instances of a given agent from the schedule.
		"""
		del self._agents[agent.unique_id]
		
	def step(self):
		"""
		Gillespie Step
		Executes the step of one ant agent, one for each time/step.
		and sample from a discrete distribution.
		"""
		"""
		We add a counter to get the number of ants at each step
		and the positions for the connectivity
		"""

		### States Counter ###
		self.W_count  = 0
		self.E_count  = 0
		self.R_count  = 0
		self.EM_count = 0
		self.RM_count = 0

		### Positions ###
		positions = []

		### Tag counter ###
		self.tag_null     = 0
		self.tag_naif     = 0
		self.tag_informed = 0

		# First, we need to compute R_t = Sum(r_i)
		#R_t Varies on each step (or time step)

		R_t = 0.0

		for agent in self.agent_buffer(shuffled=False):

			R_t += agent.r_i

			#Counter for each step
			if (agent.state == State.WAITING):
				self.W_count += 1
			if (agent.state == State.EXPLORING):
				self.E_count += 1
			if (agent.state == State.RECRUITING):
				self.R_count += 1
			if (agent.state == State.EXPLORING_M):
				self.EM_count += 1
			if (agent.state == State.RECRUITING_M):
				self.RM_count += 1

			#Tag counter
			if (agent.tag == Tag.NULL):
				self.tag_null += 1
			if (agent.tag == Tag.NAIF):
				self.tag_naif += 1
			if (agent.tag == Tag.INFORMED):
				self.tag_informed += 1
				
			#Connectivity
			positions.append(agent.pos)

		### Interactions ###
		duplicates = []
		my_list = positions
		for i in my_list:
			if my_list.count(i) > 1:
				if i not in duplicates:
					duplicates.append(i)
		self.interactions = len(duplicates)

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
		
		if (rand1 < r_norm[agent_sampled.unique_id]):

			#"Decision making" process
			if (agent_sampled.state == State.EXPLORING):
				if (rand2 < agent.model.omega/(agent.model.omega+agent.model.eta)):
					agent_sampled.step()
				else:
					agent_sampled.dead()

			#Step
			else:
				agent_sampled.step()

		else:
			pass
		
		self.steps += t_gillespie
		self.time  += t_gillespie
		