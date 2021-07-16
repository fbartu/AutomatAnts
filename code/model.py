from gillespie import GillespieAlgorithm
from agent import *
import json

class Model(GillespieAlgorithm):

    def __init__(self, n_agents, recruitment_strategy, environment, steps, path, filename):

        ants = list(Ant(a, recruitment_strategy) for a in list(range(n_agents)))

        super().__init__(ants, environment)

        self.steps = steps
        self.path = path
        self.filename = filename
    
    def run(self):

        for i in list(range(self.steps)):
            if float(i / 2000) == int(i /2000):
                print(i)
            self.step()

        # self.save_data()

    def save_data(self):

        data = {'Time (s)': self.T,
        'Connectivity': self.K,
        'N': self.N,
        'Interactions': self.I}
        
        with open(self.path + 'results/' + self.filename + '.json', 'w') as f:
            json.dump(data, f)

