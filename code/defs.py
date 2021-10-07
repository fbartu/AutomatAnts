import sys
sys.path.append('G:/research/AutomatAnts/code/')

from model import Model
from lattice import Lattice
import params
import random
import time
from copy import deepcopy

def create_instance(filename):
    environment = Lattice(params.n_agents, params.width, params.height, params.nest_node, params.food)
    return Model(params.n_agents, params.recruitment, environment, params.n_steps, path, filename)

def run_parallel(seed):
    model = create_instance()
    random.seed(random.randrange(99999) * random.random() + time.time())
    model.run()
    return deepcopy(model)