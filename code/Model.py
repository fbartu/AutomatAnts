from mesa import space, Model, Agent
from Food import Food
import networkx as nx
from Ant import np, Ant, math, nest
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from functions import rotate, moving_average, discretize_time

N = 100 # number of automata
alpha = 4*10**-3 # rate of action in nest
beta = 2 # rate of action in arena
gamma = 10**-5 # spontaneous activation

foodXvertex = 1

# sto_1: randomly distributed food (stochastic)
# sto_2: stochastic with clusterized food (hexagon patches)
# det: deterministic (sto_2 but with a specific and fixed positioning, emulating deterministic experiments)
# nf: no food (simulations without food)
food_condition = 'sto_1' # 'det', 'sto_2', 'nf'

#Lattice size
width    = 22
height   = 13

''' MODEL '''
class Model(Model):

	def __init__(self, alpha = alpha, beta = beta, gamma = gamma, N = N, width = width, height = height,
			  food_condition = food_condition):

		super().__init__()
  
		nds = [(0, i) for i in range(1, 44, 2)]

		# Lattice
		self.g = nx.hexagonal_lattice_graph(width, height, periodic = False)
		[self.g.remove_node(i) for i in nds]
		self.coords = nx.get_node_attributes(self.g, 'pos')
		self.grid = space.NetworkGrid(self.g)
		x = [xy[0] for xy in self.coords.values()]
		y = [xy[1] for xy in self.coords.values()]
		xy = [rotate(x[i], y[i], theta = math.pi / 2) for i in range(len(x))]
		self.xy = dict(zip(self.coords.keys(), xy))
  
		# Agents
		self.agents = {}
		for i in range((N-1), -1, -1):
			self.agents[i] = Ant(i, self)
   
  		# states & rates
		self.states = {'alpha': list(self.agents.values()), 'beta': [], 'gamma': list(self.agents.values())}
		self.S = np.array([N, 0, N])
		self.rates = np.array([alpha, beta, gamma])
  
		# Init first active agent
		self.agents[0].Si = np.random.uniform(0.0, 1.0)
		self.agents[0].update_status()
		self.Si = [np.mean([i.Si for i in list(self.agents.values())])]
   

  
  		# Food
		self.food_condition = food_condition
		self.init_food()
   
		self.init_state = {'Si': [self.agents[i].Si for i in self.agents],
					 'g': [self.agents[i].g for i in self.agents],
					 'food': len(self.food), 'alpha': alpha, 'beta': beta,
					 'gamma': gamma, 'N': N}

		# Rates
		self.update_rates()
		self.rate2prob()

		# Time & Gillespie
		self.time = 0
		self.sample_time()

		# Metrics
		self.T = [0] # time
		self.N = [0] # population
		self.I = [0] # interactions
		self.XY = dict(zip(list(self.coords.keys()), [0] *len(self.coords.keys())))
		self.n = [np.mean([self.agents[i].Si for i in self.agents])]
		self.o = [0]
		self.gOut = [0]
		self.gIn = [np.mean([self.agents[i].g for i in self.agents])]
		self.iters = 0
		self.a = [self.r[0]]
		self.gamma_counter = 0

		self.sampled_agent = []
  
	def update_rates(self):
		self.S = np.array([len(i) for i in list(self.states.values())])
		self.r = self.S * self.rates # np.array([self.alpha, self.beta, self.gamma]) 

	def rate2prob(self):
		self.R_t = np.sum(self.r)
		self.r_norm = self.r / self.R_t

	def sample_time(self):
		self.rng = np.random.random()
		self.rng_t = (1 /self.R_t) * np.log(1 / self.rng)

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
	  
			process = np.random.choice(['alpha', 'beta', 'gamma'], p = self.r_norm)
   
			if process == 'alpha':
					
				agent = np.random.choice(self.states['alpha'])
	
			elif process == 'beta':

				agent = np.random.choice(self.states['beta'])
	
			else:

				agent = np.random.choice(self.states['gamma'])

				self.gamma_counter += 1

			self.sampled_agent.append(agent.unique_id)

			# do action
			agent.action(process)
			if agent.pos != 'nest':
				self.XY[agent.pos] += 1
			self.collect_data()
   
			self.update_rates()
			self.rate2prob()
			
			# get time for next iteration
			self.time += self.rng_t
			self.T.append(self.time)
			agent.activity['t'].append(self.time)

			# get rng for next iteration
			self.sample_time()
			self.iters += 1
   
	def collect_data(self):
		self.N.append(len(self.states['beta']))
		self.n.append(np.mean([i.Si for i in self.states['alpha']]))
		self.o.append(np.mean([i.Si for i in self.states['beta']]))
		self.gIn.append(np.mean([i.g for i in self.states['alpha']]))
		self.gOut.append(np.mean([i.g for i in self.states['beta']]))
		self.Si.append(np.mean([i.Si for i in list(self.agents.values())]))
		self.a.append(self.r[0])
   
	def init_food(self):
     
		self.food_in_nest = 0
		if food_condition == 'det':
			self.init_det()
		elif food_condition == 'sto_1':
			self.init_sto()
		elif food_condition == 'sto_2':
			self.init_stoC()
		elif food_condition == 'nf':
			self.init_nf()
		else:
			Warning('No valid food conditions, initing non-clustered stochastic by default')
			self.init_sto()
			
	def init_det(self):
		food_positions = [(6, 33), (6, 34), (7, 34), # patch 1
	(7, 33), (7, 32), (6, 32),
	(6, 11), (6, 12), (7, 12), # patch 2
	(7, 11), (7, 10), (6, 10)]
  
		food_id = -1
  
		self.food = {}
		for i in food_positions:
			self.food[i] = [Food(i)] * foodXvertex
			for x in range(foodXvertex):
				self.grid.place_agent(self.food[i][x], i)
				self.food[i][x].unique_id = food_id
				food_id -= 1
     
	def init_sto(self):
		food_id = -1
		self.food_in_nest = 0
  
		nodes = np.array(list(self.xy.keys()))
		food_indices = np.random.choice(len(self.xy), size = 12, replace = False)
		self.food_positions = [tuple(x) for x in nodes[food_indices]]
		self.food_dict = dict.fromkeys(self.food_positions, foodXvertex)
		self.food = {}
		for i in self.food_dict:
			self.food[i] = [Food(i)] * foodXvertex
			for x in range(foodXvertex):
				self.grid.place_agent(self.food[i][x], i)
				self.food[i][x].unique_id = food_id
				food_id -= 1
    
	def init_stoC(self):
		## WORK IN PROGRESS ##
		self.init_sto()

	def init_nf(self):
		self.food = dict.fromkeys((), [np.nan])
			
	def run(self, tmax = 10800, plots = False):

		self.step(tmax = tmax)
		if plots:
			self.plot_N()
			# self.plot_I()
  
	def run_food(self, tmax, plots = False):
		n = sum(self.model.food_dict.values())
		t = 1
		while sum(self.model.food_dict.values()) == n:
			self.step(t)
			t += 1
		self.step(tmax + t)
		if plots:
			self.plot_N()
			# self.plot_I()

	def save_results(self, path):

		self.results = pd.DataFrame({'N': self.N, 'T': self.T, 'I':self.I})
		self.results.to_csv(path + 'N.csv')

	def plot_lattice(self, z = None, labels = False):
		
		coordsfood = [self.xy[i] for i in self.food]

		if self.food_condition == 'det' or self.food_condition == 'sto_2':

			xyfood = [coordsfood[:6],coordsfood[6:]]
			plt.fill([x[0] for x in xyfood[0]], [x[1] for x in xyfood[0]], c = 'grey')
			plt.fill([x[0] for x in xyfood[1]], [x[1] for x in xyfood[1]], c = 'grey')
   
		elif self.food_condition == 'sto_1':
			plt.scatter([x[0] for x in coordsfood], [x[1] for x in coordsfood], c = 'grey', s = 200, alpha = 0.5)

		if z is None:

			plt.scatter([x[0] for x in self.xy.values()], [x[1] for x in self.xy.values()])

		else:
			plt.scatter([x[0] for x in self.xy.values()], [x[1] for x in self.xy.values()], c = z, cmap = 'coolwarm')
   
		if labels:
			v = list(self.xy.values())
			for i, txt in enumerate(self.coords.keys()):
				plt.annotate(txt, v[i])
		plt.scatter(self.xy[nest][0], self.xy[nest][1], marker = '^', s = 125, c = 'black')
		plt.show()
  
	def plot_trajectory(self, id):

		coordsfood = [self.xy[i] for i in self.food]
		xyfood = [coordsfood[:6],coordsfood[6:]]
		plt.fill([x[0] for x in xyfood[0]], [x[1] for x in xyfood[0]], c = 'grey')
		plt.fill([x[0] for x in xyfood[1]], [x[1] for x in xyfood[1]], c = 'grey')
  
		xy = [self.xy[i] for i in self.agents[id].path]
		plt.scatter([x[0] for x in xy], [x[1] for x in xy], alpha = 1, c = list(range(len(xy))),cmap = 'viridis', zorder = 2)
  
		e = list(self.g.edges)
		for i in e:
			coords = self.xy[i[0]], self.xy[i[1]]
			x = coords[0][0], coords[1][0]
			y = coords[0][1], coords[1][1]
			plt.plot(x, y, linewidth = 3, c = '#999999', zorder = 1)
   


		plt.scatter(self.xy[nest][0], self.xy[nest][1], marker = '^', s = 50, c = 'black')
		plt.show()
  
  
	def plot_N(self):

		t2min = 60
		# t2min = 120
		v = self.N
		t = np.array(self.T) / t2min
		plt.plot(t, v)

		if 0 in list(self.food_dict.values()):

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
		i = moving_average(discretize_time(self.I, self.T, solve = 0), t = 60, overlap = 30)
		plt.plot(np.array(list(range(len(i)))) / 60, i)
		plt.xlabel('Time (min)')
		plt.ylabel('Interactions')
		plt.xticks(list(range(0, 185, 15)))
		plt.show()
  
	def plot_cumI(self):
		plt.plot(np.array(self.T) / 60, np.cumsum(self.I))
		plt.xlabel('Time (min)')
		plt.ylabel('Cumulated interactions')
		plt.xticks(list(range(0, 185, 15)))

	def depart_entry_correlation(self):

		if not hasattr(self, 'dpt_ent'):

			dpt = [1 if i > 0 else 0 for i in self.C]
			ent = [1 if i < 0 else 0 for i in self.C]

			dpt_disc = discretize_time(dpt, self.T, solve = 0)
			ent_disc = discretize_time(ent, self.T, solve = 0)

			dpt_ma = moving_average(dpt_disc, t = 30, overlap = 15)
			ent_ma = moving_average(ent_disc, t = 30, overlap = 15)

			dpt_ma = [i if i < 1 else 0 for i in dpt_ma]
			ent_ma = [i if i < 1 else 0 for i in ent_ma]

			R = pearsonr(dpt_ma, ent_ma)

			self.dpt_ent = {'dpt': dpt_ma, 'ent': ent_ma, 'R': R[0], 'pvalue': R[1]}

		else:

			dpt_ma = self.dpt_ent['dpt']
			ent_ma = self.dpt_ent['ent']
			R = [self.dpt_ent['R']]

		print('R = %s' % round(R[0], 3))
		plt.scatter(dpt_ma, ent_ma)
		plt.xlabel('Nest departures')
		plt.ylabel('Nest entries')
		plt.show()