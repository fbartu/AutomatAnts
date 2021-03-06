
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

import networkx as nx
import gillespie_activation
from agent import *

#-----------------------------------------------------------
#
# 							Model
#
#----------------------------------------------------------

class AntModel(Model):

	"""A model with some number of ant agents."""
 
	def __init__(self, N, width, height,nest_node,food_node,
				alpha,beta_1,beta_2,gamma_1,gamma_2,
				omega,eta):
		
		self.num_agents = N

		self.alpha   = alpha 
		self.beta_1  = beta_1
		self.beta_2  = beta_2
		self.gamma_1 = gamma_1
		self.gamma_2 = gamma_2
		self.omega   = omega
		self.eta     = eta

		self.initial_node = nest_node
		self.food_node = food_node

		#Create the hexagonal lattice
		self.G = nx.hexagonal_lattice_graph(width,height,periodic=False)
		self.grid = NetworkGrid(self.G)
		
		#Compute the shortest path of the lattice
		self.short_paths = (ShortPaths(self))

		#Initialize food quantity
		self.food_counter = (FoodCounter())
		
		#Activation step initialization
		self.schedule = gillespie_activation.ActivationStep(self)
		
		self.running = True

		#Create Agents
		for i in range(self.num_agents):
			a = AntAgent(i, self)
			self.schedule.add(a)
			self.grid.place_agent(a, self.initial_node)
			
		
	def step(self):

		"""Model Step """
		self.schedule.step()
		