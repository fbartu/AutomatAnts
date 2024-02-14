""""""""""""""""""""""""""
""" DEFAULT PARAMETERS """
""""""""""""""""""""""""""

''' MODEL '''
N = 100 # number of automata
alpha = 2*10**-3 # 4.25*10**-3 # rate of action in nest
beta = 0.6 # 1.5 # rate of action in arena
gamma = 10**-5 # spontaneous activation
foodXvertex = 1

# sto_1: randomly distributed food (stochastic)
# sto_2: stochastic with clusterized food (hexagon patches)
# det: deterministic (sto_2 but with a specific and fixed positioning, emulating deterministic experiments)
# nf: no food (simulations without food)
food_condition = 'det'# 'sto_1', 'sto_2', 'nf'

''' LATTICE PARAMETERS '''
#Lattice size
width    = 22
height   = 13

nest = (0, 22)
nest_influence = [nest, (1, 21), (1, 22), (1, 23)] 
# direction_bias = 2 

''' THRESHOLDS ''' 
theta = 0
Theta = 10**-10 # 10**-15

''' Coupling coefficients matrix '''
# 0 - No info; 1 - Info
Jij = {'0-0': 0.01, '0-1': 1,
	   '1-0': 0.01, '1-1': 1}
# Jij = {'0-0': 0.4, '0-1': 1,
# 	   '1-0': 0.4, '1-1': 1}
    
''' Rounded matrix '''
mot_matrix = {1: [0.35, 0.15, 0.50], # <-- OG!
              0: [0.40, 0.20, 0.40],
              -1:[0.50, 0.15, 0.35]}
              
### NF
              
# mot_matrix_LR = {1: [0.40, 0.15, 0.45], # <-- LR-scouts!
#               0: [0.45, 0.10, 0.45],
#               -1:[0.45, 0.15, 0.40]}      
              
# mot_matrix_SR = {1: [0.375, 0.250, 0.375], # <-- SR-scouts!
#               0: [0.450, 0.100, 0.450],
#               -1:[0.375, 0.250, 0.375]}
              
### DET
mot_matrix_LR = {1: [0.40, 0.20, 0.40], # <-- LR-scouts!
              0: [0.40, 0.20, 0.40],
              -1:[0.40, 0.20, 0.40]}      
              
mot_matrix_SR = {1: [0.36, 0.28, 0.36], # <-- SR-scouts!
              0: [0.40, 0.20, 0.40],
              -1:[0.36, 0.28, 0.36]}
        
## DET LR
#                    1         0        -1
# 1  0.3493101 0.3801874 0.4365482
# 0  0.2018882 0.2248996 0.2131980
# -1 0.4488017 0.3949130 0.3502538


## DET SR
#            1         0        -1
# 1  0.3355350 0.3719165 0.3693333
# 0  0.2787318 0.2277040 0.2946667
# -1 0.3857332 0.4003795 0.3360000

## NF LR
#            1          0        -1
# 1  0.3865699 0.46774194 0.4609665
# 0  0.1669691 0.09139785 0.1431227
# -1 0.4464610 0.44086022 0.3959108
              
## NF SR
#            1          0        -1
# 1  0.3876698 0.52916667 0.4160105
# 0  0.2685475 0.08333333 0.2493438
# -1 0.3437827 0.38750000 0.3346457
              
# mot_matrix = {1: [0.40, 0.20, 0.40],
# 0: [0.40, 0.20, 0.40],
# -1:[0.40, 0.20, 0.40]}
              
# mot_matrix = {1: [0.20, 0.10, 0.70],
#               0: [0.45, 0.10, 0.45],
#               -1:[0.70, 0.10, 0.20]}
              
# mot_matrix = {1: [0.30, 0.10, 0.60],
#               0: [0.40, 0.20, 0.40],
#               -1:[0.60, 0.10, 0.30]}