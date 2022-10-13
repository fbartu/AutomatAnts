import random
import networkx as nx
from mesa import space, Agent, datacollection, Model
import json
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from copy import deepcopy
# from scipy.stats import rv_discrete
import statistics

# from networkx.algorithms.cluster import square_clustering
# import params

""""""""""""""""""
""" PARAMETERS """
""""""""""""""""""
N = 60 # number of automata
g = 0.8 # gain (sensitivity) parameter
Theta = 0
theta = 10**-16 # threshold of activity (inactive if Activity < theta) 
Sa = 10**-6 # spontaneous activation activity
Pa = 10**-16 # probability of spontaneous activation
Jij = {'Active-Active': 1, 'Active-Inactive' : 0.2,
 'Inactive-Active': 0.5, 'Inactive-Inactive': 0.1, 
 'Active-FoodActive': 1, 'Active-FoodInactive': -1,
 'Inactive-FoodActive': 1, 'Inactive-FoodInactive': 0} # Coupling coefficients
weight = 2 # number of times it is more likely to choose the preferred direction over the other possibilities

nest = (0,22)
food_positions = [(6, 33), (6, 34), (7, 34), # patch 1
	(7, 33), (7, 32), (6, 32),
	(6, 11), (6, 12), (7, 12), # patch 2
	(7, 11), (7, 10), (6, 10)]
foodXvertex = 1

food = dict.fromkeys(food_positions, foodXvertex)

#Lattice size
width    = 22   
height   = 13 

# Si_t+1 = tanh { g [ sum(Jij * Sj_t-1) + Jii * Si_t-1 ] }
# Si_t1 = Phi * ( g * ( Jii * Si_t + np.sum( Jij * Sj_t - Theta_i ) ) ) # // STATE FUNCTION

""""""""""""""""""
""""""""""""""""""

""""""""""""""""""
""" FUNCTIONS  """
""""""""""""""""""

def dist(origin, target):
	return distance.euclidean(origin, target)


""""""""""""""#
""" CLASSES """
""""""""""""""#

''' FOOD '''

class Food:
	def __init__(self):
		self.state = 'FoodInactive'
		self.Si = 1
		self.Si_t1 = self.Si

''' ANT AGENT '''

class Ant(Agent):
	
	def __init__(self, unique_id, model):
		
		super().__init__(unique_id, model)

		self.Si = random.uniform(-1.0, 0.0) # activity
		self.Si = self.Si_t1
		self.food = []
		
		self.check_state()
		self.movement = 'random'

	def update_activity(self):
		self.Si = self.Si_t1

	def check_state(self):
		if self.Si < theta:
			self.state = 'Inactive'
			
		else:
			self.state = 'Active'

	# Si_t+1 = tanh { g [ sum(Jij * Sj_t-1) + Jii * Si_t-1 ] }
	def compute_activity(self):
		z = [Jij[self.state+"-"+n.state] * n.Si - Theta for n in self.model.grid.get_cell_list_contents([self.pos])]
		z = sum(z) + Jij[self.state + "-" + self.state]* self.Si
		self.Si_t1 = math.tanh(g * z)

	def ant2nest(self):
		self.target = self.model.coords[nest]
		self.movement = 'persistant'

	def pick_food(self):
		self.model.remove_agent(self.model.food[self.pos][0])
		self.food.append(self.model.food[self.pos].pop(0))
		self.model.food[self.pos].append(self.model.time)
		food[self.pos] -= 1

	def food2nest(self):
		self.model.place_agent(self.food.pop(), self.pos)
			
	# Move method
	def move(self):

		possible_steps = self.model.grid.get_neighbors(
		self.pos,
		include_center = False)

		if self.movement == 'random':
			pos = random.choice(possible_steps)

		else:
			d = [dist(self.target, self.model.coords[i]) for i in possible_steps]
			idx = np.argmin(d)

			if self.movement == 'persistant':
				v = 1 / (len(d) + weight)
				p = [weight / (len(d) + weight) if i == idx else v for i in range(len(d))]
				pos = np.random.choice(possible_steps, p = p)

			else:
				pos = possible_steps[idx]

		self.model.grid.move_agent(self, pos)
			

	# Possible actions the ant may take
	def action(self):

		if self.state == 'Active':

			if self.pos in food_positions:
				if not len(self.food):
					if food[self.pos] > 0:
						# get time of retrieval?
						self.pick_food()
						self.ant2nest()

					else:
						# smell food ???
						self.Si += weight

				else:
					self.ant2nest()
					self.move()

			elif self.pos == nest:
				if len(self.food):
					self.food2nest()
				if self.movement == 'random':
					self.move()

				else:
					if self.food:
						self.food2nest() ## ??

			else:
				self.move()

		self.compute_activity()

''' MODEL '''

class Model(Model):

	def __init__(self, N = 100, width = width, height = height):

		super().__init__()
		self.update = []

		# Lattice
		self.g = nx.hexagonal_lattice_graph(width, height, periodic = False)
		self.grid = space.NetworkGrid(self.g)
		self.coords = nx.get_node_attributes(self.g, 'pos')
		self.in_nest = list(range(N))
		self.out_nest = [N - len(self.in_nest)]

		# Agents
		self.agents = []
		self.food = {}
		for i in range(N):
			self.agents.append(Ant(i, self))
			self.grid.place_agent(self.agents[-1], nest)
		for i in food:
			self.food[i] = [Food()] * foodXvertex
			for f in range(foodXvertex):
				self.grid.place_agent(self.food[i][f], i)

		idx = random.randrange(N)
		self.agents[idx].Si = 30 # ON PARAMETRIZATION !
		self.agents[idx].check_state()
		
		# Rates
		self.r = np.array([a.Si for a in self.agents])
		self.r = self.r + abs(np.min(self.r)) # normalization (1: convert to positive values)
		self.r = self.r / np.sum(self.r) # normaliation (2: convert to probabilities)


		# Time & Gillespie
		self.time = 0
		self.T = [0]
		self.rnorm = []
		self.rng_t = random.random()
		self.rng_action = random.random()

	def remove_agent(self, agent: Agent) -> None:
		""" Remove the agent from the network and set its pos variable to None. """
		pos = agent.pos
		self._remove_agent(agent, pos)
		agent.pos = None

	def _remove_agent(self, agent: Agent, node_id: int) -> None:
		""" Remove an agent from a node. """

		self.g.nodes[node_id]["agent"].remove(agent)

	def step(self):
		
		idx = int(np.random.choice(list(range(len(self.agents))), 1, p = self.r))

		if self.rng_action < float(self.r_norm[idx]):
			# get sampled id
			self.sample.append(idx) 

			# do action & report 
			# step 1: scheduled action
			# step 2: get results to data collector
			flag = self.agents[idx].action(self.environment, self.agents)

			# update rates in the model
			self.r[idx] = self.agents[idx].r_i
			for i in self.agents[idx].recruited_ants:
					self.r[i] = self.agents[i].r_i
			self.r_norm = np.array(self.r) / sum(self.r)
			self.R_t = sum(self.r)

			# update time
			self.T.append(self.time)

		if self.rng_action < Sa:
			pass # activate random ant in nest

		# get rng for next iteration
		self.rng_t = random.random()
		self.rng_action = random.random()

		# get time for next iteration
		self.time += abs(np.log(self.rng_t)/self.R_t)

	
	def run(self):

		print('+++ RUNNING MODEL +++')
		if self.steps == 0:
			i = 0
			while self.T[-1] < 10800:
				i+=1
				if i % 2000 == 0:
					print('Iteration ', str(i))
					
				self.step()
		
		else:
			for i in list(range(self.steps)):
				if i % 2000 == 0:
					print('Iteration ', str(i))
				self.step()
			
		print('Model completed... Saving results !')

		self.save_data()
		print('+++ Results saved +++')

	def time2minutes(self):
		self.T = [t / 60.0 for t in self.T]

	def retrieve_positions(self):
		# result = {'agent': [],'pos': [],'t': [], 'state': [], 'tag': [], 'mia': []}
		result = {'agent': [],'pos': [],'t': [], 'state': []}
		for i in range(len(self.agents)):
			result['agent'].extend([i] * len(self.agents[i].path))
			result['pos'].extend(self.agents[i].path)
			result['t'].extend(list(map(self.T.__getitem__, np.where(self.sample == np.array([i]))[0])))
			if len(self.agents[i].state_history) > 1:
				result['state'].extend(self.agents[i].state_history)
			#result['tag'].extend(self.agents[i].tag)
			#result['mia'].append(self.agents[i].MIA)
		
		return result

	def save_data(self):
		cols = list(self.population.columns)
		try:
			cols.remove('W')
			self.population[State.WAITING] = len(self.agents) - self.population[cols]
		except:
			pass
		
		self.results = [{'Time (s)': self.T,
		# 'Connectivity': self.K,
		'N': self.N,
		'Interactions': self.I,
		'Food in nest': self.F,
		'Exploring from food': self.population[State.EXPLORING_FROM_FOOD],
		#'Informed': self.population[Tag.INFORMED],
		'Waiting': self.population[State.WAITING],
		'Carrying food': self.population[[State.EXPLORING_FOOD, State.RECRUITING_FOOD]].sum(axis = 1),
		'Exploring': self.population[State.EXPLORING],
		'Recruiting': self.population[State.RECRUITING]},
		self.metrics.efficiency(self.tfood),
		self.retrieve_positions()]
		
	def data2json(self, folder = '', filename = 'Run_1', get_pos = False):    
		
		if not hasattr(self, 'results'):
			self.save_data()

		pd.DataFrame(self.results[0]).to_csv(self.path + 'results/' + folder + filename + '_data.csv')

		# with open(self.path + 'results/' + folder + filename + '_data.json', 'w') as f:
		#     json.dump(self.results[0], f)

		with open(self.path + 'results/' + folder + filename + '_food.json', 'w') as f:
			json.dump(self.results[1], f)

		if get_pos:
			
			with open(self.path + 'results/' + folder + filename + '_pos.json', 'w') as f:
				json.dump(self.results[2], f)

''' METRICS '''

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
		return int(sum(np.array(counts) > 1))
	
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
						try:
							pos.remove(current_path[-1])
						except:
							next
						neighbors = np.array(self.environment.grid.get_neighbors(current_path[-1]))
						idx = np.where([(n[0], n[1]) in pos for n in neighbors])
						branch.extend(list(map(tuple, neighbors[idx])))
					
					k_length.append(len(current_path))
					current_path = []
					branch = []

			return statistics.mean(k_length)
			#return k_length
		else:
			return 0

	
	def efficiency(self, tfood):
		if not len(self.environment.food_cluster):
			self.environment.cluster_food(pos = 0) # pos has no default // need to workaround

		food_found = np.array(list(map(lambda x: x == True, tfood['Flag'])))
		
		for p in list(self.environment.food_cluster.keys()):
			patch = []

			for i in self.environment.food_cluster[p]:
				food_visited = np.array(list(map(lambda x: x == i, tfood['Pos'])))
				try:
					idx = int(np.where(np.logical_and(food_found == True, food_visited == True))[0])
					patch.append(tfood['Time'][idx])
				except:
					patch.append('Not Found')

			self.environment.food_cluster[p] = {'x': [x[0] for x in self.environment.food_cluster[p]],
			'y': [y[1] for y in self.environment.food_cluster[p]],
			 't': patch}

		return self.environment.food_cluster