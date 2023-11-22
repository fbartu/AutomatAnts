from mesa import space, Model, Agent
from Food import Food
import networkx as nx
from Ant import np, Ant, math, nest, dist
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from functions import rotate, moving_average, discretize_time, fill_hexagon, concatenate_values#, parse_states
from parameters import N, alpha, beta, gamma, foodXvertex, food_condition, width, height, mot_matrix, Jij, Theta

''' MODEL '''
class Model(Model):

	def __init__(self, alpha = alpha, beta = beta, gamma = gamma, Theta = Theta, Jij = Jij, N = N, 
              width = width, height = height, food_condition = food_condition, mot_matrix = mot_matrix, **kwargs):

		super().__init__()

		if 'Theta' not in kwargs:
			self.Theta = Theta
		else:
			self.Theta = kwargs['Theta']
   
		if 'Jij' not in kwargs:
			self.Jij = Jij
		else:
			self.Jij = kwargs['Jij']

  
		nds = [(0, i) for i in range(1, 44, 2)]

		# Lattice
		self.g = nx.hexagonal_lattice_graph(width, height, periodic = False)
		[self.g.remove_node(i) for i in nds]
		self.coords = nx.get_node_attributes(self.g, 'pos')
		for i in self.coords:
			self.coords[i] = tuple(np.round(self.coords[i], 5))
		self.grid = space.NetworkGrid(self.g)
		x = [xy[0] for xy in self.coords.values()]
		y = [xy[1] for xy in self.coords.values()]
		xy = [rotate(x[i], y[i], theta = math.pi / 2) for i in range(len(x))]
		self.xy = dict(zip(self.coords.keys(), xy))
		if 'd' in kwargs:
			d = float(kwargs["d"])
			if d < 0:
				self.distance = 3
			elif d > 26:
				self.distance = 26
			else:
				self.distance = d
		else:
			self.distance = 13
  
		# Agents
		self.init_agents(**kwargs)
			# self.agents[i] = Ant(i, self)
		self.mot_matrix = mot_matrix
   
  		# states & rates
		self.states = {'alpha': list(self.agents.values()), 'beta': [], 'gamma': list(self.agents.values())}
		self.S = np.array([N, 0, N])
		self.rates = np.array([alpha, beta, gamma])
  
		# Init first active agent
		self.agents[0].Si = np.random.uniform(0.0, 1.0)
		self.agents[0].update_status()
		self.Si = [self.agents[0].Si / N]# [np.mean([i.Si for i in list(self.agents.values())])]
  
  		# Food
		self.food_condition = food_condition
		self.init_food()
   
		self.init_state = {'Si': [self.agents[i].Si for i in self.agents],
					 'g': [self.agents[i].g for i in self.agents],
					 'food': str(list(self.food.keys())),'alpha': [alpha], 'beta': [beta],
					 'gamma': [gamma], 'N': [N]}

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
		self.H = ['Inactive'] # ant states
		self.position_history = ['nest']
		self.ids = [-1]
		self.SI = [0]
		# self.XY = dict(zip(list(self.coords.keys()), [0] *len(self.coords.keys())))
  
		self.n = [np.mean([self.agents[i].Si for i in self.agents])]
		self.o = [0]
		self.gOut = [0]
		self.gIn = [np.mean([self.agents[i].g for i in self.agents])]
		self.iters = 0
		# self.a = [self.r[0]]
		self.gamma_counter = 0
		self.init_nodes() ## initializes some metrics by node
		self.comm_count = 0 ## addition of target movement
		self.expl_count = 0
  
		self.report = pd.DataFrame({'T': [0], 'Si': [self.agents[0].Si], 'nest_active': [1], 
                              'N': [0], 'Si_pc' : [self.agents[0].Si], 'p_active': [1/N]})

		self.sampled_agent = [np.nan]
  
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
			self.position_history.append(agent.pos)
			self.SI.append(agent.Si)
			self.ids.append(agent.unique_id)
			if agent.pos != 'nest':
				# self.XY[agent.pos] += 1
				self.nodes.loc[self.nodes['Node'] == agent.pos, 'Si'] += agent.Si
			self.H.append(agent.get_state())
			self.collect_data()
   
			self.update_rates()
			self.rate2prob()
			
			# get time for next iteration
			self.time += self.rng_t
			self.T.append(self.time)
			# agent.activity['t'].append(self.time)

			# get rng for next iteration
			self.sample_time()
			self.iters += 1
   
   
			# WORKING ON MACROSCOPIC MEASURES AND ORDER PARAMETERS 
			# Si_nest = np.sum([i.Si for i in self.states['alpha']])
			# N_nest = len(self.states['alpha'])
			# self.report.loc[len(self.report)] = [self.T[-1], Si_nest,
            #                             N_nest, self.N[-1], Si_nest / N_nest,
            #                             np.sum([1 if i.Si > 0 else 0 for i in self.states['alpha']])/ N_nest]

   
	def collect_data(self):
		self.N.append(len(self.states['beta']))
		self.n.append(np.mean([i.Si for i in self.states['alpha']]))
		self.o.append(np.mean([i.Si for i in self.states['beta']]))
		self.gIn.append(np.mean([i.g for i in self.states['alpha']]))
		self.gOut.append(np.mean([i.g for i in self.states['beta']]))
		self.Si.append(np.mean([i.Si for i in list(self.agents.values())]))
		# self.a.append(self.r[0])
   
	def init_agents(self, **kwargs):
     
		if 'default_movement' in kwargs:
			dmove = kwargs['default_movement']
		else:
			dmove = 'random'
   
		if 'g' in kwargs:
			
			# must be passed as: size, type, value 1, value 2
			# e.g. '75,N,0.5,0.2' -> 75 % of gaussian with mu = 0.5 and sigma = 0.2
			# e.g. '25,B,0.5,0.5' -> 25 % of beta with shapes a = b = 0.5
			code = kwargs['g'].split(':')
			g = []
			try:
				for i in range(len(code)):
						s, t, v1, v2 = code[i].split(',')
						if t == 'N':
							g += list(np.random.normal(loc = float(v1), scale = float(v2), size = int(s)))
						elif t == 'B':
							g += list(np.random.beta(a = float(v1), b = float(v2), size = int(s)))
						else:
							g += list(np.random.uniform(low = float(v1), high = float(v2), size = int(s)))
				if len(g) < N:
					print('Warning: Less gains than population size passed to parametrization')
					g += np.random.uniform(low = 0.0, high = 1.0, size = N - len(g))
				elif len(g) > N:
					print('Warning: More gains than population size passed to parametrization')
					g = g[:N]

			except:
				print('Warning: Values must be passed in a 4 sized list separated by commas.',
           '\n Switching to default behaviour in gain initialization')
				g = np.random.uniform(low = 0.0, high = 1.0, size = N)
     
		else:
			g = np.random.uniform(low = 0.0, high = 1.0, size = N)
   
		if 'recruitment' in kwargs:
			r = kwargs['recruitment']
		else:
			r = False

		self.agents = {}
		for i in range((N-1), -1, -1):
			self.agents[i] = Ant(i, self, default_movement=dmove, g=g[i], recruitment=r)
   
	def init_food(self):
     
		self.food_in_nest = 0
		if self.food_condition == 'det':
			self.init_det()
		elif self.food_condition == 'dist':
			self.init_dist()
		elif self.food_condition == 'sto_1':
			self.init_sto()
		elif self.food_condition == 'sto_2':
			self.init_stoC()
		elif self.food_condition == 'nf':
			self.init_nf()
		else:
			print('No valid food conditions, initing non-clustered stochastic by default')
			self.init_sto()
			
	def init_det(self):
		self.food_positions = [(6, 33), (6, 34), (7, 34), # patch 1
	(7, 33), (7, 32), (6, 32),
	(6, 11), (6, 12), (7, 12), # patch 2
	(7, 11), (7, 10), (6, 10)]
		self.food_dict = dict.fromkeys(self.food_positions, foodXvertex)  
		food_id = -1
  
		self.food = {}
		for i in self.food_positions:
			self.food[i] = [Food(i)] * foodXvertex
			for x in range(foodXvertex):
				self.grid.place_agent(self.food[i][x], i)
				self.food[i][x].unique_id = food_id
				food_id -= 1
    
	def init_dist(self):
		tolerance = 1.5
		food_id = -1
		darray = np.array([dist(self.xy[i], self.xy[nest]) for i in self.xy])
		idx = np.where((darray > (self.distance - tolerance)) & (darray < (self.distance + tolerance)))[0]

		nodes = np.array(list(self.xy.keys()))
		food_indices = np.random.choice(idx, size = 12, replace = False)
		self.food_positions = [tuple(x) for x in nodes[food_indices]]
		self.food_dict = dict.fromkeys(self.food_positions, foodXvertex)
		self.food = {}
		for i in self.food_dict:
			self.food[i] = [Food(i)] * foodXvertex
			for x in range(foodXvertex):
				self.grid.place_agent(self.food[i][x], i)
				self.food[i][x].unique_id = food_id
				food_id -= 1
    
	def init_sto(self):
		food_id = -1
  
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
		food_id = -1
  
		nodes = np.array(list(filter(lambda a: self.xy[a][1] > 0.5, self.xy)))
		food_indices = np.random.choice(len(nodes), size = 2, replace = False)
		self.food_positions = [tuple(x) for x in nodes[food_indices]]
		bl = [(i, j) for j in range(45, 2, -2) for i in range(1, 12, 2)] + [(i, j) for j in range(44, 1, -2) for i in range(2, 13, 2)]
		clusters = []
		for i in self.food_positions:
			if i in bl:
				clusters.extend(fill_hexagon(i))
			else:
				l = [dist(self.xy[i], self.xy[target]) for target in bl]
				idx = np.argmin(l)
				clusters.extend(fill_hexagon(bl[idx]))
    
		self.food_positions = clusters
		self.food_dict = dict.fromkeys(self.food_positions, foodXvertex)
		self.food = {}
		for i in self.food_dict:
			self.food[i] = [Food(i)] * foodXvertex
			for x in range(foodXvertex):
				self.grid.place_agent(self.food[i][x], i)
				self.food[i][x].unique_id = food_id
				food_id -= 1
    
	def init_nf(self):
		self.food = dict.fromkeys((), [np.nan])
		self.food_positions = []
		self.food_dict = {}
			
	def run(self, tmax = 10800, plots = False):

		self.step(tmax = tmax)
		if plots:
			self.plot_N()
   
		self.z = [self.nodes.loc[self.nodes['Node'] == i, 'N'] for i in self.xy]
		self.zq = np.unique(self.z, return_inverse = True)[1]
		self.collect_results()
  
	def collect_results(self, fps = 2):
		result = pd.DataFrame({'T': self.T, 'N': self.N, 'I': self.I, 'SiOut': self.o, 
                         'pos': list(zip(self.sampled_agent, self.position_history)), 'S': self.H})
		result['Frame'] = result['T'] // (1 / fps)
		df = result.groupby('Frame').agg({'N': 'mean', 'I': 'sum', 'SiOut': 'mean', 'pos': concatenate_values, 'S': concatenate_values}).reset_index()

		# df['pos'] = result.groupby('Frame').agg({'pos': concatenate_values}).reset_index()['pos']
		food = pd.DataFrame({'node': list(self.food.keys()),
				't': [round(food.collection_time,3) if food.is_collected else np.nan for foodlist in self.food.values() for food in foodlist ]})
  
		self.df = df
		self.food_df = food
		self.data = pd.DataFrame({'Frame': [round(i*2) for i in self.T], 'id': self.ids,
                            'gNest': self.gIn, 'gArena': self.gOut,
                            'states': self.H, 'si_out': self.Si, 'si_nest': self.n})
		# e = parse_states(self)
		# self.entropy = -np.sum([i * np.log(i) for i in e])
  
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

	def save_results(self, path, filename):

		self.df.to_csv(path + filename + '.csv', index=False)
		self.data.to_csv(path + filename + '_data.csv', index=False)
		self.food_df.to_csv(path + filename + '_food.csv', index=False)
		self.nodes.to_csv(path + filename + '_positions.csv', index=False)

	def plot_lattice(self, z = None, labels = False):
		
		coordsfood = [self.xy[i] for i in self.food]

		if self.food_condition == 'det' or self.food_condition == 'sto_2':

			xyfood = [coordsfood[:6],coordsfood[6:]]
			plt.fill([x[0] for x in xyfood[0]], [x[1] for x in xyfood[0]], c = 'grey')
			plt.fill([x[0] for x in xyfood[1]], [x[1] for x in xyfood[1]], c = 'grey')
   
		elif self.food_condition == 'sto_1' or self.food_condition == 'dist':
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
  
	def init_nodes(self, chunks_x = 3 ,chunks_y = 2):
		if not hasattr(self, 'nodes'):
			xykeys = list(self.xy.keys())
			keyarray = np.array(xykeys)
			xyvals = list(self.xy.values())
			xyarray = np.array(xyvals)
			maxx = np.max([x[0] for x in xyarray])
			minx = np.min([x[0] for x in xyarray])
			maxy = np.max([x[1] for x in xyarray]) +0.1 # otherwise it does not catch the top vertices
			miny = np.min([x[1] for x in xyarray])
   
			delta_x = (maxx - minx) / chunks_x
			delta_y = (maxy - miny) / chunks_y
			xcoords = [minx + delta_x * i for i in range(chunks_x)]
			ycoords = [miny + delta_y * i for i in range(chunks_y)]
			lims = [((j, i), (j+ delta_x, i + delta_y) ) for i in ycoords for j in xcoords]
   
			nodelist = [(xyarray[::, 0] >= i[0][0]) &
                  (xyarray[::, 0] < i[1][0]) &
                  (xyarray[::, 1] >= i[0][1] ) &
                  (xyarray[::, 1] < i[1][1]) for i in lims]
   
			self.nodes = pd.DataFrame({'Node': [], 'Coords': [], 'Sector': []})
			for i in range(len(nodelist)):
				nds = [tuple(x) for x in xyarray[nodelist[i]]]
				tags = [tuple(x) for x in keyarray[nodelist[i]]]
				self.nodes = pd.concat([self.nodes, 
                             pd.DataFrame({'Node': tags, 'Coords': nds, 'Sector': [i+1] * len(nds)})])
    
			self.nodes['N'] = 0
			self.nodes['Si'] = 0


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
