from mesa import Agent
import numpy as np
from functions import dist
import math

''' LATTICE PARAMETERS '''
nest = (0, 22)
nest_influence = [nest, (1, 21), (1, 22), (1, 23)] 
weight = 3 # integer >= 1, direction bias

''' THRESHOLDS ''' 
theta = 0
Theta = 10**-15


''' Coupling coefficients matrix '''
# 0 - No info; 1 - Info
Jij = {'0-0': 0.35, '0-1': 1,
	   '1-0': 0.35, '1-1': 1}

''' ANT AGENT '''
class Ant(Agent):

	def __init__(self, unique_id, model, **kwargs):

		super().__init__(unique_id, model)

		self.Si = 0
		self.g = np.random.uniform(0.0, 1.0)

		self.is_active = False
		self.state = '0'
		self.status = 'gamma'

		self.activity = {'t': [0], 'Si': [self.Si]}
  
		self.food = []

		self.pos = 'nest'
		if not 'default_movement' in kwargs:
			default_movement = 'random'
		self.movement = 'default'
		self.move_default = self.check_movement(default_movement)
  
		# self.last_move = None
		self.path = []
  
	def check_movement(self, type):
		if type == 'random':
			return self.move_random
		else:
			print('Invalid default movement, defaulting to random')
			return self.move_random

	def move_random(self, pos):
		l = list(range(len(pos)))
		idx = np.random.choice(l)
		return pos[idx]

	def move_persistance(self, pos):
		l = list(range(len(pos)))
		d = [dist(self.target, self.model.coords[i]) for i in pos]
		idx = np.argmin(d)
		v = 1 / (len(d) + weight - 1)
		p = [weight / (len(d) + weight - 1) if i == idx else v for i in l]
		idx = np.random.choice(l, p = p)
		return pos[idx]


	# Move method
	def move(self):
     
		possible_steps = self.model.grid.get_neighbors(
		self.pos,
		include_center = False)

		if self.movement == 'default':
			pos = self.move_default(possible_steps)
	
		else:
			pos = self.move_persistance(possible_steps)

		self.model.grid.move_agent(self, pos)
 
	def reset_movement(self):
		self.movement = 'default'

	def find_neighbors(self):

		if self.pos == 'nest':
   
			alist = self.model.states['alpha']

		else:
			alist = self.model.grid.get_cell_list_contents([self.pos])
   
		flist = list(filter(lambda a: a.unique_id != self.unique_id, alist))
  
		if len(flist):
			neighbors = np.random.choice(flist, size = 1, replace = False)
		else:
			neighbors = []

		return neighbors

	def interaction(self):
		neighbors = self.find_neighbors()

		s = [] # state
		z = [] # activity
  
		l = len(neighbors)
		if l:
			for i in neighbors:
				s.append(i.state)
				z.append(Jij[self.state + "-" + i.state]* i.Si - Theta)

			z = sum(z)
   
			if self.pos in ['nest'] + nest_influence:
				self.model.I.append(0)
			else:
				self.model.I.append(+1)
	
		else:
			z = -Theta
			self.model.I.append(0)
		self.Si = math.tanh(self.g * (z + self.Si) ) # update activity
	
	def update_status(self):
		self.check_status()
		for i in self.model.states:
			try:
				self.model.states[i].remove(self)
			except:
				continue
	
		if self.status == 'gamma':
			self.model.states['alpha'].append(self)
			self.model.states['gamma'].append(self)
   
		else:
			self.model.states[self.status].append(self)
	
	def check_status(self):
		if self.is_active:
			self.status = 'beta'
		else:
			if self.Si > theta:
				self.status = 'alpha'
			else:
				self.status = 'gamma'
 
	def leave_nest(self):
		self.model.grid.place_agent(self, nest)
		self.is_active = True

	def enter_nest(self):
		self.model.remove_agent(self)
		self.is_active = False
		self.pos = 'nest'
		self.ant2explore()

	def ant2nest(self):
		self.target = self.model.coords[nest]
		self.movement = 'homing'

	def ant2explore(self):
		if hasattr(self, 'target'):
			del self.target
		self.reset_movement()

	def pick_food(self):
		self.model.remove_agent(self.model.food[self.pos][0])
		self.food.append(self.model.food[self.pos].pop(0))
		self.model.food[self.pos].extend(self.food)
		self.model.food[self.pos][-1].collected(self.model.time)
		self.model.food_dict[self.pos] -= 1
		self.food_location = self.pos
		self.state = '1'

	def drop_food(self):
		self.food.pop()
  
	def action(self, rate):
		
		if rate == 'alpha':
			if len(self.food):
				self.drop_food()
			else:
				if self.Si > theta:
					self.leave_nest()

		elif rate == 'beta':
	  
			if len(self.food) or self.Si < theta:
				self.ant2nest()

			if self.pos == nest:
				if hasattr(self, 'target') and self.target == self.model.coords[nest]:
					self.enter_nest()

				else:
					self.move()

			elif self.pos in self.model.food_positions:
	   
				if self.model.food_dict[self.pos] > 0 and not len(self.food):
					self.neighbors = self.find_neighbors()
					self.pick_food()

				else:
					self.move()

			else:
				self.move()
   
		else:
			self.Si = np.random.uniform(0.0, 1.0) ## spontaneous activation

		self.interaction()
		self.update_status()
		self.activity['Si'].append(self.Si)