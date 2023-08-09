""""""""""""""""""""""""""
""" DEFAULT PARAMETERS """
""""""""""""""""""""""""""

''' MODEL '''
N = 100 # number of automata
alpha = 4*10**-3 # rate of action in nest
beta = 2 # rate of action in arena
gamma = 10**-5 # spontaneous activation
foodXvertex = 1
pheromone_quantity = (0.1, 0.5) # in any node, in food node

# sto_1: randomly distributed food (stochastic)
# sto_2: stochastic with clusterized food (hexagon patches)
# det: deterministic (sto_2 but with a specific and fixed positioning, emulating deterministic experiments)
# nf: no food (simulations without food)
food_condition = 'sto_1' # 'det', 'sto_2', 'nf'

''' LATTICE PARAMETERS '''
#Lattice size
width    = 22
height   = 13

nest = (0, 22)
nest_influence = [nest, (1, 21), (1, 22), (1, 23)] 
direction_bias = 3 # integer >= 1

''' THRESHOLDS ''' 
theta = 0
Theta = 10**-15

''' Coupling coefficients matrix '''
# 0 - No info; 1 - Info
Jij = {'0-0': 0.35, '0-1': 1,
	   '1-0': 0.35, '1-1': 1}