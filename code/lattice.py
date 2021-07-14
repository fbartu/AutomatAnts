#from mesa.space import NetworkGrid
import networkx as nx

class Lattice():

	"""A model with some number of ant agents."""

	def __init__(self, n_agents, width, height, nest_node, food):

		# Initialize key components in the lattice
		self.initial_node = nest_node
		self.food = food
		self.food_in_nest = 0
		self.food_cluster = {}
		self.waiting_ants = n_agents

		#Create the hexagonal lattice
		self.G = nx.hexagonal_lattice_graph(width,height,periodic=False)
		#self.grid = NetworkGrid(self.G)

		# Compute the shortest paths of the lattice (food to nest, and backwards)
		# The key of the dictionary holds the food position, so it is easy to keep track of the path
		self.paths2food = [nx.shortest_path(self.G, pos, self.initial_node) for pos in self.food.keys()]
		self.paths2food = dict(zip(list(self.food.keys(), self.paths2food)))

		self.paths2nest = [list(reversed(path)) for path in self.paths2food]
		self.paths2nest = dict(zip(list(self.food.keys(), self.paths2nest)))


	# splits food list into packs of 6 positions (that usually conform a patch of food)
	# assumes ordered positions 
	def split_food_patches(self, each = 6):
		lst = list(self.food.keys())
		self.patch_positions = [lst[i:i + each] for i in lst]

	# method for grouping the food (in a list of positions) in clusters (patches)
	def cluster_food(self, pos, id = 0, iterative = True):
		if iterative:
			self.split_food_patches()
			for i in range(len(self.patch_positions)):
				self.food_cluster{'Patch ' + str(i): self.patch_positions[i]}
		else:
			food_cluster = {'Patch ' + str(id): pos}
			self.food_cluster['Patch ' + str(id)] = food_cluster
