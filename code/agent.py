import random
import numpy as np
import networkx as nx

from enum import IntEnum
from copy import deepcopy

# from lattice import Lattice
import gillespie_activation
from model import *
import params

#-----------------------------------------------------------
#
# Different States
#
#-----------------------------------------------------------

class State(IntEnum):
	"""Possible states of the ant
	N = W + E + R + EM+ RM 
	"""

	WAITING = 1
	EXPLORING = 2
	RECRUITING = 3
	EXPLORING_M = 4
	RECRUITING_M = 5
	DEAD = 6


#-----------------------------------------------------------
#
# Different Tags
#
#-----------------------------------------------------------

class Tag(IntEnum):

	"""Study the explotation of the food / information flux
	N = Naif + informed """

	NAIF = 1
	INFORMED = 2

				
#-----------------------------------------------------------
#
# Ant Agent Definition
#
#-----------------------------------------------------------

# lattice = Lattice(params.n_agents, params.width, params.height, params.nest_node, params.food)

class Ant():
	""" An ant agent."""
	
	def __init__(self, unique_id, recruitment_strategy):

		self.id = unique_id
		
		# Variables to save and initialization
		self.pos = params.nest_node
		self.state = State.WAITING
		self.r_i = params.alpha
		self.path2nest = []
		self.path2food = []

		# Recruitment related
		self.recruitment_strategy = recruitment_strategy
		self.recruit = self.choose_recruitment()

		# No information about food
		self.tag = Tag.NAIF

	def choose_recruitment(cls):

		if cls.recruitment_strategy == 'IR':
			return cls.IR

		elif cls.recruitment_strategy == 'HR':
			return cls.HR

		elif cls.recruitment_strategy == 'GR':
			return cls.GR

		else:
			return cls.NR
	
	def IR(self, environment, ant_pool):
		s = random.choice(environment.waiting_ants)
		environment.waiting_ants.remove(s)
		agent = ant_pool[s]
		agent.r_i = params.omega
		agent.tag = Tag.INFORMED
		agent.state = State.RECRUITING
	
	def HR(self, environment, ant_pool):
		r = random.randrange(0, 6)
		if len(r) > 0:
			for s in len(range(r)):
				environment.waiting_ants.remove(s)
				agent = ant_pool[s]
				agent.r_i = params.omega
				agent.tag = Tag.INFORMED
				agent.state = State.RECRUITING
		
		# One possibility is: if you don't recruit, stay in nest
		'''
		else:
			self.state = State.WAITING
			self.r_i = params.alpha
		'''

	def GR(self, environment, ant_pool):
		r = random.randrange(3, 6)
		for s in len(range(r)):
			environment.waiting_ants.remove(s)
			agent = ant_pool[s]
			agent.r_i = params.omega
			agent.tag = Tag.INFORMED
			agent.state = State.RECRUITING

	def NR(self, environment):
		pass

	def move(self, environment, type = 'random'):
		if type == 'random':
			possible_steps = environment.grid.get_neighbors(
				self.pos,
				include_center = False)
			self.pos = random.choice(possible_steps)
		elif type == '2nest':
			self.pos = self.path2nest.pop(0)
		elif type == '2food':
			self.pos = self.path2food.pop(0)
		'''
		Future possible movement types could include levy walks or directional persistance
		'''


	def Action(self, environment):

		# If ant is waiting on the nest, explore.
		if (self.state == State.WAITING):

			self.state = State.EXPLORING
			self.tag = Tag.NAIF
			self.r_i = params.omega
		
		elif self.state == State.DEAD:
			
			if self.pos == environment.initial_node:
				self.r_i = params.alpha
				self.state = State.WAITING
				environment.waiting_ants.append(self.id)
			else:
				self.pos = self.move(self.pos, type = '2nest')

		# If not waiting...
		else:
			if (self.state == State.EXPLORING):
				if (self.pos in params.food_location and environment.food[self.pos] > 0):
					self.state = State.EXPLORING_M
					self.r_i = params.beta_1
					environment.food[self.pos] -= 1
				else:
					'''
					ETA DECISION: GO TO NEST ??
					rng = random.random()
					if rng < params.eta / (params.omega + params.eta)
						self.path2nest = nx.shortest_path(environment.G, self.pos, environment.initial_node)
						self.pos = self.move(self.pos, type = '2nest')
						self.state = State.DEAD
					'''
					self.pos = self.move(self.pos, type = 'random')

			elif (self.state == State.EXPLORING_M or self.state == State.RECRUITING_M):
				# if in nest node -> recruitment happens
				if(self.pos == params.nest_node):
					self.state = State.RECRUITING
					environment.food_in_nest =+ 1
					if(self.state == State.EXPLORING_M):
						self.r_i = params.gamma_1
					else:
						self.r_i = params.gamma_2

				# else keep moving to nest
				else:
					self.pos = self.move(self.pos, type = '2nest')
			elif (self.state == State.RECRUITING):
				if (self.pos in params.food_location):
					if environment.food[self.pos] > 0:
						self.state = State.RECRUITING_M
						self.r_i = params.beta_2
						environment.food[self.pos] -= 1
					else:
						self.state = State.EXPLORING
						self.r_i = params.omega
				
				elif self.pos == params.nest_node:
					self.recruit()


	### Recruitment Funtion ###

	def recruitment(self):

		"""Recruitment at the nest"""

		if(((self.state == State.EXPLORING_M) or (self.state == State.RECRUITING_M)) 
			and (self.pos == self.model.initial_node)):

			cellmates = self.model.grid.get_cell_list_contents([self.pos])

			if (len(cellmates)>1):

				###Type of recruitment###

				if params.individual_recruitment:
					n_recruits = 1                       #Reclute random  n = 1

				if params.group_recruitment:
					n_recruits = random.randrange(3,6,1) #Reclute random  n = [3,5]

				if params.hybrid_recruitment:
					n_recruits = random.randrange(0,6,1) #Reclute random  n = [0,5]

				other = random.sample(cellmates,n_recruits)

				for agent in range(len(other)):

					if other[agent].state == State.WAITING:
						
						other[agent].state = State.RECRUITING
						other[agent].r_i = self.model.omega

						### Tag ###
						other[agent].tag = Tag.INFORMED

						if ((self.remember_food_position in self.model.food_node[0])):
							other[agent].remember_food_position = self.remember_food_position
							other[agent].save_path_food = deepcopy(self.reversed_paths[0])
							other[agent].save_path_nest = deepcopy(self.paths[0])
						
						if ((self.remember_food_position in self.model.food_node[1])):
							other[agent].remember_food_position = self.remember_food_position
							other[agent].save_path_food = deepcopy(self.reversed_paths[1])
							other[agent].save_path_nest = deepcopy(self.paths[1])

						
	#-------------------------
	#
	#    MOVEMENT FUNCTIONS 
	#
	#--------------------------


		

	def return_to_nest(self):
	
		"""Return to nest once food is found"""

		if (len(self.save_path_nest) > 0):
			new_position = (self.save_path_nest[0])
			self.model.grid.move_agent(self, new_position)
			self.save_path_nest.pop(0)
	
	def return_to_food(self):

		"""Return to the node where food is found"""
		
		if (len(self.save_path_food) > 0):
			new_position = (self.save_path_food[0])
			self.model.grid.move_agent(self, new_position)
			self.save_path_food.pop(0)

	def move(self):

		if(self.state == State.EXPLORING):
			self.r_i = self.model.omega + self.model.eta
			self.random_move()

			#Save Position where food is found and path to return (1)
			if ((self.pos in self.model.food_node[0])):

				self.remember_food_position = self.pos
				self.save_path_food = deepcopy(self.reversed_paths[0])
				self.save_path_nest = deepcopy(self.paths[0])

			#Save Position where food is found and path to return (2)
			if ((self.pos in self.model.food_node[1])):

				self.remember_food_position = self.pos
				self.save_path_food = deepcopy(self.reversed_paths[1])
				self.save_path_nest = deepcopy(self.paths[1])


		if((self.state == State.EXPLORING_M) or (self.state == State.RECRUITING_M)):
			self.r_i = self.model.omega
			self.return_to_nest()


		if(self.state == State.RECRUITING):
			self.r_i = self.model.omega
			self.return_to_food()

			#Save position to return to the food node
			if ((self.pos in self.model.food_node[0])):
				self.save_path_food = deepcopy(self.reversed_paths[0])
				self.save_path_nest = deepcopy(self.paths[0])

			if ((self.pos in self.model.food_node[1])):
				self.save_path_food = deepcopy(self.reversed_paths[1])
				self.save_path_nest = deepcopy(self.paths[1])
	
	#-------------------------------
	#
	#        STEP FUNCTION 
	#
	#--------------------------------

	def step(self):

		"""Step of each ant"""
		self.move()

		if params.recruitment:
			self.recruitment()

		self.nest()

		self.food()

	#-------------------------------
	#
	#        ADITIONAL FUNCTIONS 
	#
	#--------------------------------

	def dead(self):
		"""Dead agent that returns instantaniously to the nest
		And with state waiting; Parameter = eta """

		self.model.grid.move_agent(self, self.model.initial_node)
		self.state = State.WAITING
		self.r_i = self.model.alpha

		### Tag ###
		self.tag = Tag.NULL
		