""" RELATED PAPERS """
"""
R.V. Solé, O.Miramontes and B.C.Goodwin. J. Theor. Biol.
161 (1993) 343.

R.V. So1é, O. Miramontes and B.C. Goodwin, Emergent
Behaviour in Insect Societies: Global Oscillations, Chaos and
Computation, in: Interdisciplinary Approaches to Nonlinear
Complex Systems, eds. H. Haken and A. Mikhailov, Springer
Series in Synergetics, Vol. 62 (Springer, 1993) p. 77.

O. Miramontes, R.V. Solé and B.C. Goodwin, Physica D 63
(1993) 145. 
LINK: https://www.sciencedirect.com/science/article/pii/016727899390152Q?via%3Dihub


O. Miramontes, R.V. Solé and B.C. Goodwin, Antichaos in
Ants: the Excitability Metaphor at Two Hierarchical Levels,
in: Selforganization and Life (MIT Press, 1993) p. 790.

D.J. Amit, Modelling Brain Function (Cambridge University
Press, 1989).

"""

''' INFORMATION AT THE EDGE OF CHAOS IN FLUID NEURAL NETWORKS (Solé & Miramontes, 1993)'''
import math
import numpy as np
import random

""" PARAMETERS """
L = 10 # lattice grid
N = 10 # number of automata
rho = N / L**2 # agent density
g = 0.05 # 0.035 # gain (sensitivity) parameter; from 0.005 to 0.5
theta = 10**-16 # threshold of spontaneous activation (if Si_t > theta) 
# Sa = 10**-6 # spontaneous activation activity
Sa = 0.01 # spontaneous activation activity
Pa = 0.01 # probability of spontaneous activation
# Coupling coefficients (Jij, intensity of interaction?) are 1


# Si_t = [] # state of the ith automata at time t
# Phi = [] # sigmoid function describing the state of the neuron or automata (D.J. Amit, Modelling Brain Function; Cambridge University Press, 1989)
# Theta = 0 # threshold of activation, set to Theta = 0 for simplicity (says the paper). Virtually, Theta = theta (10**-16)
# this spontaneous activation brings S_i to Sa, with some probability Pa
# B = [] # boundary. The state of the ith automaton is the sum of the states of the eight neighbours plus its state
# states available
# Active = [] # moves to one of the eight nearest cells, if able to
# Inactive = []


# Si_t1 = Phi * ( g * ( Jii * Sj_t + np.sum( Jij * Sj_t - Theta_i ) ) ) # // STATE FUNCTION

# Initial positions and activity (-1, 1) chosen randomly

class Agent:
    def __init__(self, pos):
        self.pos = pos
        self.new_pos = pos

        self.Si = random.uniform(-1.0, 1.0) # activity
        self.old_activity = self.Si
        self.new_activity = 0
        
        
        self.check_state()
        
    def check_state(self):
        if self.Si <= theta:
            self.Si = 0
            self.old_activity = 0
            self.state = 'inactive'
            
        else:
            self.state = 'active'
            
    def action(self, grid):
        
        self.check_state()
        
        if self.state == 'inactive':
            if random.random() < Pa:
                self.Si = Sa
                
        else:
            self.move(grid)
    
    def assign_activity(self):
        self.old_activity = self.Si
        self.Si = self.new_activity
        self.pos = self.new_pos
        
    def move(self, grid):
        m = grid.available_positions(self.pos)
        if len(m):
            pos = tuple(random.choice(m))
        else:
            pos = self.pos
            
        self.new_pos = pos
        grid[self.new_pos] = 1
        
    def compute_activity(self, neighbors):
        if len(neighbors):
            s = []
            for i in neighbors:
                s.append(i.old_activity)
                
            s.append(self.old_activity)
                
        else:
            s = [self.old_activity]
            
        self.new_activity = math.tanh(g * (sum(s)))
        

class Lattice:
    
    def __init__(self, L):
        
        self.L = L
        self.grid = np.zeros((L, L), dtype = int)
        self.neighborhood = [(-1, 1), (0, 1), (1, 1), (-1, 0),
             (1, 0), (-1, -1), (0,-1), (1,-1)]
        # i = np.indices((L, L))
        # x = np.concatenate(i[0]).ravel().tolist()
        # y = np.concatenate(i[1]).ravel().tolist()
        # self.coords = list(zip(x, y))
        # self.x, self.y = np.meshgrid(np.linspace(0, L, L), np.linspace(0, L, L))
        
    def available_positions(self, pos):
        new_positions = np.array(pos) + self.neighborhood
        
        # filter 1, eliminate non-existing coordinates
        idx = np.sum((new_positions > -1) & (new_positions < (L+1)), axis = 1)
        new_positions = new_positions[np.where(idx == 2)]
        
        # filter 2, eliminate busy coordinates
        idx = np.where(self.grid[tuple(new_positions.T.tolist())] == 0)[0]
        if len(idx):
            return list(new_positions[idx])
        
        else:
            return []
        
        
class Model:
    def __init__(self):
        self.grid = Lattice(L = L)
        self.agents = []
        for i in range(N):
            pos = random.choice(list(zip(*np.where(self.grid == 0))))
            self.agents.append(Agent(pos = pos))
        



# sigmoid function
# def Phi(g, z):
#     mu = g * z
#     return math.tanh(mu)



# Step 1: Check state (active, inactive)
# Step 2: Compute state ()
# Step 3: Assign states