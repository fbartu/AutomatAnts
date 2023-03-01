import networkx as nx
from mesa import space, Agent, Model
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

""""""""""""""""""
""" PARAMETERS """
""""""""""""""""""
N = 100 # number of automata
alpha = 0.5 / N # expected average of a random uniform U(0, 1)
beta = 2.5
Theta = 0 # threshold
theta = 10**-16 # threshold of activity (inactive if Activity < theta) 
Interactions = 4 # integer
weight = 3 # must be an integer equal or greater than 1

# Coupling coefficients
# 1: No info, 2: Indirect info, 3: Direct info
Jij = {'1-1': 0.5, '1-2': 1, '1-3': 1.5,
	   '2-1': 0.5, '2-2': 0.5, '2-3': 1.5,
	   '3-1': 0.5, '3-2': 0.5, '3-3': 1}

# nest coords
nest = (0, 22)

food_positions = [(6, 33), (6, 34), (7, 34), # patch 1
	(7, 33), (7, 32), (6, 32),
	(6, 11), (6, 12), (7, 12), # patch 2
	(7, 11), (7, 10), (6, 10)]
foodXvertex = 1

food = dict.fromkeys(food_positions, foodXvertex)

#Lattice size
width    = 22   
height   = 13 

'''FOOD'''

class Food:
	def __init__(self, pos):
		self.state = '3'
		self.is_active = True
		self.Si = 1 # Interactions
		self.Si_t1 = self.Si
		self.initial_pos = pos
		self.rate = 0
		self.is_collected = False
   
	def __repr__(self):
		if self.is_collected:
			t = self.collection_time / 120
			t = (int(t), round((t - int(t)) * 60))

			msg = 'Food collected at %s minutes and %s seconds' % t
   
		else:
			msg = 'Food not collected yet!!'
   
		return msg
	
	def collected(self, time):
		self.collection_time = time
		self.is_collected = True

	def compute_activity(self):
		pass

	def activity(self, n):
		pass

	def update(self):
		pass
		# self.Si -= 0.15
		# if self.Si < 0:
		# 	self.Si = 0


''' ANT AGENT '''
class Ant(Agent):
	
	def __init__(self, unique_id, model):
		
		super().__init__(unique_id, model)

		self.Si = np.random.uniform(0.0, 1.0)# np.random.normal(0.5, 0.2)
		self.rate = alpha
		# self.update_rate()
		self.g = np.random.normal(0.5, 0.125) #np.random.uniform(0.0, 1.0)# 0.8 # np.random.normal(0.5, 0.125)
		self.history = []
		# self.history = 0
  
		self.is_active = False
		self.state = '1'
		self.food = []
  
		self.pos = 'nest'
		self.movement = 'random'
  
		# self.memory = {'1': [0] * 21600, '2': [0] *21600, '3': [0]* 21600}
		

	# Move method
	def move(self):

		possible_steps = self.model.grid.get_neighbors(
		self.pos,
		include_center = False)

		if self.movement == 'random':
			idx = np.random.choice(list(range(len(possible_steps))))
			pos = possible_steps[idx]

		else:
			d = [dist(self.target, self.model.coords[i]) for i in possible_steps]
			idx = np.argmin(d)

			v = 1 / (len(d) + weight - 1)
			p = [weight / (len(d) + weight - 1) if i == idx else v for i in range(len(d))]
			pos = np.random.choice(range(len(d)), p = p)
			pos = possible_steps[pos]

		self.model.grid.move_agent(self, pos)
  
	def find_neighbors(self):
     
		if self.pos == 'nest':
			l = len(self.model.in_nest)
      
			if l > Interactions:
      
				alist = list(filter(lambda a: a.unique_id in self.model.in_nest and
					a.unique_id != self.unique_id,
					list(self.model.agents.values())))
   
				neighbors = np.random.choice(alist, size = np.random.choice(list(range(Interactions))), replace = False)

			elif l > 1:
				alist = list(filter(lambda a: a.unique_id in self.model.in_nest and
				a.unique_id != self.unique_id,
				list(self.model.agents.values())))
   
				neighbors = np.random.choice(alist, size = np.random.choice(list(range(l - 1))), replace = False)
			else:
				neighbors = []
   
		else:
			neighbors = self.model.grid.get_cell_list_contents([self.pos])
   
			if len(neighbors):
				neighbors = list(filter(lambda a: a.unique_id != self.unique_id and a.unique_id > -1, neighbors))
				if len(neighbors) > Interactions:
					neighbors = np.random.choice(neighbors, size = Interactions, replace = False)

		return neighbors

	def interaction(self):
		neighbors = self.find_neighbors()

		# g = [] # gain
		s = [] # state
		f = [] # food
		z = [] # activity
  
		for i in neighbors:
			# g.append(i.g)
			s.append(i.state)
			f.append(len(i.food))
			z.append(Jij[self.state + "-" + i.state]* i.Si - Theta)
		
		z = sum(z)

		self.Si = math.tanh(self.g * (z + self.Si) ) # update activity
		# self.g = np.mean(g + [self.g]) # update gain
		self.update_role(s, f) # update state 

	# tambe podria fer-ho tirant una moneda: 50/50 de canviar o seguir igual
	def update_role(self, states, food):
		# pass
		if self.state == '3' or sum(food) > 0:
			self.state = '3'
		elif self.state == '1' and '2' in states:
			self.state = '2'
		elif '3' in states:
			self.state = '2'

	def update_rate(self):
		self.rate = beta
		# self.rate = abs(self.Si)
		# self.rate = self.g
  
	def report(self):
		neighbors = self.find_neighbors()
		for i in neighbors:
			i.Si = math.tanh(i.g * i.Si + Jij[i.state + "-" + self.state] * self.Si)
			# i.Si = math.tanh(i.g * np.mean([i.Si, Jij[i.state + "-" + self.state] * self.Si]))
			# i.Si = math.tanh(i.g * Jij[i.state + "-" + self.state] * self.Si)
			# self.model.r[i.unique_id] = rate
			# i.rate = rate

	# def report(self):
	# 	if self.Si > theta:
	# 		neighbors = self.find_neighbors()
	# 		for i in neighbors:
	# 			rate = math.tanh(i.g * (i.Si + Jij[i.state + "-" + self.state] * self.Si))
	# 			self.model.r[i.unique_id] = rate
	# 			i.rate = rate
    
  
	def leave_nest(self):
		self.model.grid.place_agent(self, nest)
		self.is_active = True
		self.model.in_nest.remove(self.unique_id)
		self.update_rate()
		# self.rate = beta

	def enter_nest(self):
		# self.model.grid.remove_agent(self)
		self.model.remove_agent(self)
		self.is_active = False
		self.model.in_nest.append(self.unique_id)
		self.pos = 'nest'
		self.movement = 'random'
		del self.target
		# self.report()
		# self.update_rate()
		self.rate = alpha
		
	def ant2nest(self):
		self.target = self.model.coords[nest]
		self.movement = 'persistant'
	
	def ant2explore(self):
		if hasattr(self, 'target'):
			del self.target
		self.movement = 'random'

	def pick_food(self):
		self.model.remove_agent(self.model.food[self.pos][0])
		self.food.append(self.model.food[self.pos].pop(0))
		self.model.food[self.pos].extend(self.food)
		self.model.food[self.pos][-1].collected(self.model.time)
		food[self.pos] -= 1
		self.food_location = self.pos

	def drop_food(self):
		self.food.pop() # possibilitat de que el food actui dins el nest (com a estat '3')
  
	def action(self):
		# update = False

		if self.is_active: # in arena
	  
			if len(self.food) or self.Si < theta:
				self.ant2nest()
    
			# else:
			# 	if np.random.uniform() < self.Si:
			# 		self.ant2explore()
    

			# elif self.Si < theta:
			# elif self.Si < np.random.exponential(10**-16):
				# self.ant2nest()
			# else:
			# 	self.movement = np.random.choice(['random', self.movement], p = [0.2, 0.8])
				# self.movement = 'random'
	
		
			if self.pos == nest:
				if hasattr(self, 'target') and self.target == self.model.coords[nest]:
					self.enter_nest()
					# update = True
	
				else:
					self.move()
	
			elif self.pos in food_positions:
				if food[self.pos] > 0 and not len(self.food):
					self.pick_food()
					
				else:
					self.move()
	
			else:
				self.move()
     
		else:
      
			if len(self.food):
				self.drop_food()

			else:
				# if self.Si > theta:
				# 	self.leave_nest()
				# if self.rate == alpha:
				# 	self.update_rate()
				self.leave_nest()
	 
		self.interaction()
		# if update:
		# 	self.update_rate()
  
''' MODEL '''
class Model(Model):

	def __init__(self, N = N, width = width, height = height):

		super().__init__()
  
		# Lattice
		self.g = nx.hexagonal_lattice_graph(width, height, periodic = False)
		self.grid = space.NetworkGrid(self.g)
		self.coords = nx.get_node_attributes(self.g, 'pos')

		# Agents
		self.agents = {}
		self.ids = list(range(N))
		for i in range(N):
			self.agents[i] = Ant(i, self)
   
		self.Si = [self.agents[i].Si for i in self.agents]
   
		self.in_nest = list(range(N))
		# self.lefood = 0
  
  		# Food
		self.food_id = -1
		self.food_in_nest = 0
		if foodXvertex > 0:
			self.food = {}
			for i in food:
				self.food[i] = [Food(i)] * foodXvertex
				for x in range(foodXvertex):
					self.grid.place_agent(self.food[i][x], i)
					self.food[i][x].unique_id = self.food_id
					self.food_id -= 1

		else:
			self.food = dict.fromkeys(food.keys(), [np.nan])

		# Rates
		self.r = [self.agents[i].rate for i in self.agents]
		self.rate2prob()

		# Time & Gillespie
		self.time = 0
		self.sample_time()

		# Metrics
		self.T = [0] # time
		self.N = [0] # population
		self.I = [0] # interactions
		# self.A = [0] # alpha ~ nest departures and entries
		self.XY = {self.T[-1]: [a.pos for a in self.agents.values()]}
		self.iters = 0
  
		self.sampled_agent = []
  
	def rate2prob(self):
		self.R_t = np.sum(self.r)
		self.r_norm = self.r / self.R_t
  
	def sample_time(self):
		self.rng_t = np.random.exponential(1 / self.R_t)
  
	def update_alpha(self):
		global alpha
		global N
		alpha = np.mean(self.Si) / N

	def remove_agent(self, agent: Agent) -> None:
		""" Remove the agent from the network and set its pos variable to None. """
		pos = agent.pos
		self._remove_agent(agent, pos)
		agent.pos = None

	def _remove_agent(self, agent: Agent, node_id: int) -> None:
		""" Remove an agent from a node. """

		self.g.nodes[node_id]["agent"].remove(agent)
 
	def step(self, tmax):

		while self.time < tmax:
	  
			id = np.random.choice(self.ids, p = self.r_norm)
			self.sampled_agent.append(id)
	  
			agent = self.agents[id]
   
			# prev_state = agent.is_active
	
			# do action
			agent.action()
   
			# curr_state = agent.is_active
   
			self.r[agent.unique_id] = agent.rate
			self.Si[agent.unique_id] = agent.Si
			self.Si = [self.agents[i].Si for i in self.agents]

			self.rate2prob()

			# get time for next iteration
			self.time += self.rng_t

			# get rng for next iteration
			self.sample_time()

			self.iters += 1
		
			# update activity
			self.N.append(N - len(self.in_nest) + self.food_in_nest)

			# update time
			self.T.append(int(self.time))
   
			# if prev_state == curr_state:
			# 	self.A.append(0)
			# elif prev_state is True and curr_state is False:
			# 	self.A.append(-1)
			# elif prev_state is False and curr_state is True:
			# 	self.A.append(1)
   
			self.XY[self.T[-1]] = [a.pos for a in self.agents.values()]
		self.update_alpha()

	def run(self, steps = 21600):
		for i in range(steps):
			self.step(tmax = i)
   
		c = []
		for i in self.XY:
			c += self.XY[i]
   
		self.z = [0 if i == nest else c.count(i) for i in self.coords]
		q = np.quantile(self.z, 0.95)
		self.z = [i if i < q else q for i in self.z] 
  
		self.plots()

	def save_results(self, path):
		self.results = pd.DataFrame({'N': self.N, 'T': self.T})
		x = [xy[0] for xy in self.coords.values()]
		y = [xy[1] for xy in self.coords.values()]
		xy = [rotate(x[i], y[i], theta = math.pi / 2) for i in range(len(x))]
		self.xyz = pd.DataFrame({'x': [i[0] for i in xy],
						  'y': [i[1] for i in xy],
						  'z': self.z})
		self.results.to_csv(path + 'N.csv')
		self.xyz.to_csv(path + 'xyz.csv')

   
	def plot_lattice(self, z = None):
		x = [xy[0] for xy in self.coords.values()]
		y = [xy[1] for xy in self.coords.values()]
		xy = [rotate(x[i], y[i], theta = math.pi / 2) for i in range(len(x))]
		coordsfood = [self.coords[i] for i in self.food]
		xyfood = [[rotate(x[0], x[1], theta= math.pi / 2) for x in coordsfood[:6]],
			 [rotate(x[0], x[1], theta= math.pi / 2) for x in coordsfood[6:]]]
  
		plt.fill([x[0] for x in xyfood[0]], [x[1] for x in xyfood[0]], c = 'grey')
		plt.fill([x[0] for x in xyfood[1]], [x[1] for x in xyfood[1]], c = 'grey')
  
		if z is None:

			plt.scatter([x[0] for x in xy], [x[1] for x in xy])
		
		else:
			plt.scatter([x[0] for x in xy], [x[1] for x in xy], c = z, cmap = 'coolwarm')
		xynest = rotate(self.coords[nest][0], self.coords[nest][1], math.pi / 2)
		plt.scatter(xynest[0], 0, marker = '^', s = 50, c = 'black')
		plt.show()
  
	def plot_N(self):
	 
		t2min = 120
		v = discretize_time(self.N, self.T)
		t = np.array(list(range(len(v)))) / t2min
		plt.plot(t, v)
  
		if 0 in list(food.values()):

			times = list(filter(lambda i: i[0].is_collected, self.food.values()))

			minv = np.min([i[0].collection_time for i in times]) / t2min
			maxv = np.max([i[0].collection_time for i in times]) / t2min
   
		else:
			minv = np.nan
			maxv = np.nan

		plt.axvline(x = minv, ymin = 0, ymax = np.max(self.N), color = 'midnightblue', ls = '--')
		plt.axvline(x = maxv, ymin = 0, ymax = np.max(self.N), color = 'midnightblue', ls = '--')
		plt.xlabel('Time (min)')
		plt.ylabel('Number of active ants')
		plt.xticks(list(range(0, 185, 15)))
		plt.show()
  
	def plot_I(self):
		plt.plot(self.T, self.I)
		plt.show()
  
	def plots(self):
		self.plot_N()
		self.plot_lattice(self.z)
  
	# def depart_entry_correlation(self):
	 
	# 	if not hasattr(self, 'dpt_ent'):
			
	# 		# diff = np.diff(self.N)
	
	# 		dpt = [1 if i > 0 else 0 for i in self.A]
	# 		ent = [1 if i < 0 else 0 for i in self.A]
	
	# 		dpt_disc = discretize_time(dpt, self.T, solve = 0)
	# 		ent_disc = discretize_time(ent, self.T, solve = 0)

	# 		dpt_ma = moving_average(dpt_disc, t = 60, overlap = 30)
	# 		ent_ma = moving_average(ent_disc, t = 60, overlap = 30)
	
	# 		dpt_ma = [i if i < 1 else 0 for i in dpt_ma]
	# 		ent_ma = [i if i < 1 else 0 for i in ent_ma]
   
	# 		R = pearsonr(dpt_ma, ent_ma)
   
	# 		self.dpt_ent = {'dpt': dpt_ma, 'ent': ent_ma, 'R': R[0], 'pvalue': R[1]}
   
	# 	else:
	  
	# 		dpt_ma = self.dpt_ent['dpt']
	# 		ent_ma = self.dpt_ent['ent']
	# 		R = (self.dpt_ent['R'])
   
	# 	print('R = %s' % round(R[0], 3))
	# 	plt.scatter(dpt_ma, ent_ma)
	# 	plt.xlabel('Nest departures')
	# 	plt.ylabel('Nest entries')
	# 	plt.show()