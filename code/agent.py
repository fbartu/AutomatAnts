
import random
import numpy as np
import networkx as nx

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from enum import IntEnum
from copy import deepcopy

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


#-----------------------------------------------------------
#
# Different Tags
#
#-----------------------------------------------------------

class Tag(IntEnum):

	"""Study the explotation of the food / information flux
	N = Null + Naif + Informed"""

	NULL = 1
	NAIF = 2
	INFORMED = 3

#-----------------------------------------------------------
#
#  Food Counter Class
#
#-----------------------------------------------------------

class FoodCounter:

	def __init__(self):
		self.f_nest   = 0
		self.f_site_1 = deepcopy(params.food_site_1_list)
		self.f_site_2 = deepcopy(params.food_site_2_list)

	def add_food(self):
		self.f_nest += 1

	def substract_site_1(self,item):
		self.f_site_1[item] -= 1 
	
	def substract_site_2(self,item):
		self.f_site_2[item] -= 1


#-----------------------------------------------------------
#
#  Different shortest path of the grid
#  
#-----------------------------------------------------------

class ShortPaths:

	"""Generate a data set with all the possible guided paths
	Initializate at the start of the program, 
	Save all the short paths in a vector"""

	path = []

	def __init__(self, model):

		for i in range(len(model.food_node)):
			ShortPaths.source = model.food_node[i][0]
			ShortPaths.target = model.initial_node
			self.Graph = model.G
			ShortPaths.path.append(nx.shortest_path(self.Graph, ShortPaths.source, ShortPaths.target))
				
#-----------------------------------------------------------
#
# Ant Agent Definition
#
#-----------------------------------------------------------

def move(pos, path = ShortPaths().path, type = 'random'):
	if type == 'random':
		possible_steps = AntAgent.model.grid.get_neighbors(
			pos,
			include_center = False)
		pos = random.choice(possible_steps)
	elif type == 'informed':
		'a'

	return pos
		# self.model.grid.move_agent(self, pos)

def Action(pos, state, tag, is_recruited = False):
	# Possible actions if ant is on the nest
	if (state == State.WAITING):
		
		if (is_recruited):
			state = State.RECRUITING
			tag = Tag.INFORMED
		else:
			pos = move(pos, type = 'random')
			state = State.EXPLORING
			tag = Tag.NAIF

		r_i = params.omega
	else:
		if (state == State.EXPLORING):
			if (pos in params.food_location):


	return pos, state, tag, r_i

class AntAgent(Agent):
	""" An ant agent."""
	
	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)
		
		#Variables to save and initialization
		self.pos = params.nest_node
		self.state = State.WAITING
		self.r_i = self.model.alpha

		### Tag ###
		self.tag = Tag.NULL

		#Different possible paths
		test =  model.short_paths.path
		self.paths = []
		self.reversed_paths = []
		for i in range(len(test)):
			self.paths.append(model.short_paths.path[i])
			self.reversed_paths.append(list(reversed(model.short_paths.path[i])))

	def nest(self):

		"""Possibles estats al niu"""

		if (self.pos == self.model.initial_node):

			"""The ant Leave the colony to explore
			Parameter : Alpha"""	

			if (self.state == State.WAITING):
				self.state = State.EXPLORING
				self.r_i = self.model.alpha

			"""The ants is recuited by an explorer
			Parameter : Gamma_1"""

			if (self.state == State.EXPLORING_M):
				self.state = State.RECRUITING
				self.r_i = self.model.gamma_1
				self.model.food_counter.add_food()
				
			"""La formiga es reclutada per una recluta
			Parameter : Gamma_2"""

			if (self.state == State.RECRUITING_M):
				self.state = State.RECRUITING
				self.r_i = self.model.gamma_2
				self.model.food_counter.add_food()

			if (self.state == State.EXPLORING):
				self.r_i = self.model.omega + self.model.etta
				

	def food(self):

		"""Food encounter"""
		"""Recluta - Pos = posició_menjar
			Parameter : if state = explorer Beta_1
			if state = reclutat Beta_2"""
		
		#Food nodes 1 
		if ((self.pos in self.model.food_node[0])):

			for item in range(len(self.model.food_node[0])):
				if (self.model.food_node[0][item] == self.pos):
					
					if (self.model.food_counter.f_site_1[item] != 0):
						if (self.state == State.EXPLORING):
							self.state = State.EXPLORING_M
							self.r_i = self.model.beta_1
							self.model.food_counter.substract_site_1(item) #Counter

						if (self.state == State.RECRUITING):
							self.state = State.RECRUITING_M
							self.r_i = self.model.beta_2
							self.model.food_counter.substract_site_1(item) #Counter

						### Tag ###
						if (self.tag == Tag.NULL):
							self.tag = Tag.NAIF
							
					if (self.model.food_counter.f_site_1[item] == 0):
						if (self.state == State.RECRUITING):
							self.state = State.EXPLORING
							self.r_i = self.model.omega + self.model.etta

		#Food nodes 2
		if ((self.pos in self.model.food_node[1])):

			for item in range(len(self.model.food_node[1])):
				if (self.model.food_node[1][item] == self.pos):

					if (self.model.food_counter.f_site_2[item] != 0):
						if (self.state == State.EXPLORING):
							self.state = State.EXPLORING_M
							self.r_i = self.model.beta_1
							self.model.food_counter.substract_site_2(item) #Counter
							
						if (self.state == State.RECRUITING):
							self.state = State.RECRUITING_M
							self.r_i = self.model.beta_2
							self.model.food_counter.substract_site_2(item) #Counter

						### Tag ###
						if (self.tag == Tag.NULL):
							self.tag = Tag.NAIF
							
					if (self.model.food_counter.f_site_2[item] == 0):
						if (self.state == State.RECRUITING):
							self.state = State.EXPLORING
							self.r_i = self.model.omega + self.model.etta

					
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
			self.r_i = self.model.omega + self.model.etta
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
		And with state waiting; Parameter = etta """

		self.model.grid.move_agent(self, self.model.initial_node)
		self.state = State.WAITING
		self.r_i = self.model.alpha

		### Tag ###
		self.tag = Tag.NULL
		