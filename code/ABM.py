import random
import networkx as nx
from mesa import space, Agent, Model
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import Counter

""""""""""""""""""
""" PARAMETERS """
""""""""""""""""""
alpha = 0.00005
beta = 0.5
N = 100 # number of automata
g = 0.9 # gain (sensitivity) parameter
Theta = 0
theta = 10**-16 # threshold of activity (inactive if Activity < theta) 

Jij = {'Active-Active': 1, 'Active-Inactive': 1,
 'Inactive-Active': 1, 'Inactive-Inactive': 1, 
 'Active-FoodActive': 1, 'Active-FoodInactive': 1,
 'Inactive-FoodActive': 1, 'Inactive-FoodInactive': 1} # Coupling coefficients

weight = 3 # number of times it is more likely to choose the preferred direction over the other possibilities

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

""""""""""""""""""
""" FUNCTIONS  """
""""""""""""""""""

def dist(origin, target):
	return distance.euclidean(origin, target)

def rotate(x, y, theta = math.pi / 2):
	x1 = round(x * math.cos(theta) - y * math.sin(theta), 2)
	y1 = round(x * math.sin(theta) + y * math.cos(theta), 2)
	return x1, y1

""""""""""""""#
""" CLASSES """
""""""""""""""#

class Food:
	def __init__(self, pos):
		self.state = 'FoodInactive'
		self.is_active = True
		self.Si = 5
		self.Si_t1 = self.Si
		self.unique_id = -1
		self.initial_pos = pos

	def compute_activity(self):
		pass

	def activity(self, n):
		pass

	def update(self):
		pass

''' ANT AGENT '''
class Ant(Agent):
	
	def __init__(self, unique_id, model):
		
		super().__init__(unique_id, model)
  
		self.rate = alpha
		self.Si = np.random.uniform(-1.0, 1.0)
		self.history = []
		# self.history = 0
  
		self.is_active = False
		self.state = 'Inactive'
		self.food = []
  
		self.pos = 'nest'
		self.movement = 'random'
		
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
				v = 1 / (len(d) + weight - 1)
				p = [weight / (len(d) + weight - 1) if i == idx else v for i in range(len(d))]
				pos = np.random.choice(range(len(d)), p = p)
				pos = possible_steps[pos]

			else:
				pos = possible_steps[idx]

		self.model.grid.move_agent(self, pos)
  
  
	def leave_nest(self):
		self.model.grid.place_agent(self, nest)
		self.is_active = True
		self.model.in_nest.remove(self.unique_id)
		self.rate = beta
		self.state = 'Active'
  
	def enter_nest(self):
		self.model.grid.remove_agent(self)
		self.is_active = False
		self.model.in_nest.append(self.unique_id)
		self.rate = alpha
		self.state = 'Inactive'
		self.pos = 'nest'
		
	def ant2nest(self):
		self.target = self.model.coords[nest]
		self.movement = 'persistant'
  
	def ant2food(self):
		self.target = self.model.coords[self.food_location]
		self.movement = 'persistant'

	def pick_food(self):
		self.model.remove_agent(self.model.food[self.pos][0])
		self.food.append(self.model.food[self.pos].pop(0))
		self.model.food[self.pos].append(self.model.time)
		food[self.pos] -= 1
		self.food_location = self.pos

	def food2nest(self):
		self.model.grid.place_agent(self.food.pop(), self.pos)
		self.movement = 'random'

  
	def action(self):
     
		if self.is_active:
			
			if self.pos == nest:
       
				if self.Si < theta:
					self.enter_nest()
     
				elif len(self.food):
					self.food2nest()

				elif hasattr(self, 'food_location'):
					self.ant2food()
		
				else:
					self.move()
      
			elif self.pos in food_positions:
				if not len(self.food):
					if food[self.pos] > 0:

						self.pick_food()
						self.ant2nest()

					else:
						if hasattr(self, 'food_location') and self.pos == self.food_location:
							self.movement = 'random'
      
						self.move()
				else:
					if self.Si < theta:
						self.ant2nest()
					
					self.move()
     
			else:
				if self.Si < theta:
					self.ant2nest()
     
				self.move()
    
		else:
			if self.Si > theta:
				self.leave_nest()

		self.compute_activity()
		self.history.append(self.model.time)
	
	def activity(self, agents):
		z = [Jij[self.state+"-"+n.state] * n.Si - Theta for n in agents]
		z = sum(z) + Jij[self.state + "-" + self.state]* self.Si
		self.Si_t1 = math.tanh(g * z)

	def update(self):
		self.Si = self.Si_t1
	
	def compute_activity(self):
		if self.pos == 'nest':
			alist = list(filter(lambda a: a.unique_id in self.model.in_nest, list(self.model.agents.values())))
			neighbors = np.random.choice(alist, size = 4, replace = False)

		else:
      
			neighbors = self.model.grid.get_cell_list_contents([self.pos])
			neighbors = list(filter(lambda a: a.unique_id != self.unique_id, neighbors))
   
		self.activity(neighbors)
		idx = list(range(len(neighbors)))

		# for a in range(len(neighbors)):
		# 	idx.remove(a)
		# 	neighbors[a].activity(list(filter(lambda i: i in idx, neighbors))+ [self])
		# 	idx.append(a)
		
		self.update()
		# self.model.update_agents(neighbors)

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
   
		self.in_nest = list(range(N))
  
  		# Food
		self.food = {}
		for i in food:
			self.food[i] = [Food(i)]
			self.grid.place_agent(self.food[i][0], i)

		# Rates
		self.r = np.array([alpha] * N)
		self.rate2prob()

		# Time & Gillespie
		self.time = 0
		self.sample_time()

		# Metrics
		self.T = [0] # time
		self.N = [0]
		self.I = [0] # interactions
		self.XY = {self.T[-1]: [a.pos for a in self.agents.values()]}
		self.iters = 0
  
		self.sampled_agent = []
  
	def update_agents(self, agents):
		for a in agents:
			a.update()
  
	def rate2prob(self):
		self.R_t = np.sum(self.r)
		self.r_norm = self.r / self.R_t
  
	def sample_time(self):
		self.rng_t = np.random.exponential(1 / self.R_t)

	def remove_agent(self, agent: Agent) -> None:
		""" Remove the agent from the network and set its pos variable to None. """
		pos = agent.pos
		self._remove_agent(agent, pos)
		agent.pos = None

	def _remove_agent(self, agent: Agent, node_id: int) -> None:
		""" Remove an agent from a node. """

		self.g.nodes[node_id]["agent"].remove(agent)

 
	def step(self, tmax):
     
		samples = []
		
		while self.time < tmax:
      
			id = np.random.choice(self.ids, p = self.r_norm)
			# self.sampled_agent.append(id)
	  
			agent = self.agents[id]
			samples.append(agent.pos)
   
	
			# do action & report interactions
			agent.action()
   
			self.r[agent.unique_id] = agent.rate

			self.rate2prob()

			# get time for next iteration
			self.time += self.rng_t

			# get rng for next iteration
			self.sample_time()

			self.iters += 1
		
  		# update activity
		self.N.append(N - len(self.in_nest))
		counts = Counter(samples).values()
		self.I.append(sum([1 if i > 1 else 0 for i in counts]))
  
		# update time
		self.T.append(int(self.time))
		self.XY[self.T[-1]] = [a.pos for a in self.agents.values()]

	def run(self, steps = 21600):
		for i in range(steps):
			self.step(tmax = i)
   
		c = []
		for i in self.XY:
			c += self.XY[i]

		self.z = [0 if i == 'nest' else c.count(i) for i in self.coords]
  
   
		self.plots()

   
	def plot_lattice(self, z = None):
		x = [xy[0] for xy in self.coords.values()]
		y = [xy[1] for xy in self.coords.values()]
		xy = [rotate(x[i], y[i], theta = math.pi / 2) for i in range(len(x))]
  
		if z is None:
			plt.scatter([x[0] for x in xy], [x[1] for x in xy])
			plt.show()
		else:
			plt.scatter([x[0] for x in xy], [x[1] for x in xy], c = z)
			plt.show()
  
	def plot_N(self):
		plt.plot(self.T, self.N)
		plt.show()
  
	def plot_I(self):
		plt.plot(self.T, self.I)
		plt.show()
  
	def plots(self):
		self.plot_N()
		# self.plot_lattice(self.z)