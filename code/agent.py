import random
import networkx as nx
from mesa.space import NetworkGrid
from copy import deepcopy

# from lattice import Lattice
import params


'''
STATES AND TAGS
'''

class State():
	
	WAITING = 'W' # waiting in nest
	EXPLORING = 'E' # exploring
	RECRUITING = 'R' # recruiting or getting recruited
	EXPLORING_FOOD = 'EF' # transporting food as explorer
	RECRUITING_FOOD = 'RF' # transporting food as recruiter
	DEAD = 'D' # back to nest

class Tag():

	NAIVE = 'N' # no information about key locations
	INFORMED = 'I' # gets information when recruited by another ant

'''
ANT AGENT 
'''

class Ant():
	
	def __init__(self, unique_id, recruitment_strategy):

		# id initialization
		self.id = unique_id
		
		# Position, state and rates initialization
		self.pos = params.nest_node
		self.state = State.WAITING
		self.r_i = params.alpha

		# Key locations initialization
		self.path2nest = []
		self.path2food = []
		self.locations = []

		# Recruitment related
		self.recruitment_strategy = recruitment_strategy
		self.recruit = self.choose_recruitment()

		# No information about food
		self.tag = Tag.NAIVE

	# Pick the suitable recruitment method
	def choose_recruitment(cls):

		if cls.recruitment_strategy == 'IR':
			return cls.IR

		elif cls.recruitment_strategy == 'HR':
			return cls.HR

		elif cls.recruitment_strategy == 'GR':
			return cls.GR

		else:
			return cls.NR
	
	'''
	RECRUITMENT METHODS ARE DESCRIBED IN params.py:
	NR = 0
	IR = 1
	HR = [0, 5]
	GR = [3, 5]
	'''

	# Individual recruitment
	def IR(self, environment, ant_pool):
		sample = random.choice(environment.waiting_ants)
		del environment.waiting_ants[sample]
		environment.out_nest[sample] = sample
		agent = ant_pool[sample]
		agent.r_i = params.omega
		agent.tag = Tag.INFORMED
		agent.state = State.RECRUITING
		agent.path2food = deepcopy(environment.paths2food[self.locations[-1]]) # path to where food was found
		return 1 # return number of informed ants
	
	# Hybrid recruitment
	def HR(self, environment, ant_pool):
		r = random.randrange(0, 6)
		if len(r) > 0:
			samples = random.sample(list(environment.waiting_ants), r)
			for sample in samples:
				del environment.waiting_ants[sample]
				environment.out_nest[sample] = sample
				agent = ant_pool[sample]
				agent.r_i = params.omega
				agent.tag = Tag.INFORMED
				agent.state = State.RECRUITING
				agent.path2food = deepcopy(environment.paths2food[self.locations[-1]]) # path to where food was found
			return r # return number of informed ants
		else:
			return len(r)

		# One possibility is: if you don't recruit, stay in nest
		'''
		else:
			self.state = State.WAITING
			self.r_i = params.alpha
		'''

	# Group recruitment
	def GR(self, environment, ant_pool):
		r = random.randrange(3, 6)
		samples = random.sample(list(environment.waiting_ants), r)
		for sample in samples:
			del environment.waiting_ants[sample]
			environment.out_nest[sample] = sample
			agent = ant_pool[sample]
			agent.r_i = params.omega
			agent.tag = Tag.INFORMED
			agent.state = State.RECRUITING
			agent.path2food = deepcopy(environment.paths2food[self.locations[-1]]) # path to where food was found
		return r # return number of informed ants

	# No recruitment
	def NR(self):
		return 0


	# Move method
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
		
		return 0
		'''
		Future possible movement types could include levy walks or directional persistance
		'''


	# Possible actions the ant may take, based on the position and the state
	# returns 0 when no new informed ants are generated
	# this value is fed into the gillespie algorithm
	def action(self, environment, ant_pool):

		# If ant is waiting on the nest, explore.
		if (self.state == State.WAITING):

			self.state = State.EXPLORING
			self.r_i = params.omega + params.eta
			environment.out_nest[self.id] = self.id
			del environment.waiting_ants[self.id]

			return 0

		# If ant's state is "Dead" go back to nest.
		elif self.state == State.DEAD:
			
			if self.pos == environment.initial_node:
				self.r_i = params.alpha
				self.state = State.WAITING
				environment.waiting_ants[self.id] = self.id
				del environment.out_nest[self.id]
				self.path2nest = []
				
			else:
				self.move(environment, type = '2nest')
			return 0

		elif self.state == State.EXPLORING:

		
			# If food is found
			if (self.pos in environment.food and environment.food[self.pos] > 0):
				self.state = State.EXPLORING_FOOD
				self.r_i = params.beta_1
				environment.food[self.pos] -= 1
				self.locations.append(self.pos) # remember where food was found
				self.path2nest = deepcopy(environment.paths2nest[self.pos])
				self.path2food = deepcopy(environment.paths2food[self.pos])
			else:
				# Go to nest with a probability eta / (eta + omega) ~ 0.05%
				rng = random.random()
				if rng < params.eta / (params.omega + params.eta):
					self.path2nest = nx.shortest_path(environment.G, self.pos, environment.initial_node)
					self.move(environment, type = '2nest')
					self.state = State.DEAD

				else:
					self.move(environment, type = 'random')
			
			return 0

		elif (self.state == State.EXPLORING_FOOD):
			# if in nest node -> recruitment happens
			if(self.pos == environment.initial_node):
				self.state = State.RECRUITING
				environment.food_in_nest =+ 1
				self.r_i = params.gamma_1
				
			# else keep moving to nest
			else:
				self.move(environment, type = '2nest')
			
			return 0

		elif (self.state == State.RECRUITING_FOOD):
			# if in nest node -> recruitment happens
			if(self.pos == environment.initial_node):
				self.state = State.RECRUITING
				environment.food_in_nest =+ 1
				self.r_i = params.gamma_2
			# else keep moving to nest
			else:
				self.move(environment, type = '2nest')
			
			return 0

		elif (self.state == State.RECRUITING):
			# pick up food
			if (self.pos in environment.food):
				if environment.food[self.pos] > 0:
					self.state = State.RECRUITING_FOOD
					self.r_i = params.beta_2
					environment.food[self.pos] -= 1
					self.locations.append(self.pos) # remember where food was found
					self.path2nest = deepcopy(environment.paths2nest[self.pos])
					self.path2food = deepcopy(environment.paths2food[self.pos])

				else:
					self.state = State.EXPLORING
					self.r_i = params.omega + params.eta
				return 0
			# recruit other ants
			elif self.pos == environment.initial_node:
				if self.r_i == params.gamma_1 or self.r_i == params.gamma_2:
					self.r_i = params.omega
					self.recruit(environment, ant_pool)

				else:
					self.move(environment, type = '2food')
					return 0
			else:
				self.move(environment, type = '2food')
				return 0