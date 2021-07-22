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

        self.time2seconds()
        # self.save_data()

    def time2seconds(self):
        self.T = [t / 2.0 for t in self.T] # convert from frames to seconds (FPS = 2)

    def time2minutes(self):
        self.T = [t / (2.0 * 60) for t in self.T]

    def save_data(self):

        data = {'Time (s)': self.T,
        'Connectivity': self.K,
        'N': self.N,
        'Interactions': self.I,
        'Food in nest': self.F,
        'Informed': self.population[Tag.INFORMED],
        'Patches of food': self.metrics.efficiency(self.tfood),
        'Number of explorers': self.population[State.EXPLORING],
        'Number of recruiters': self.population[State.RECRUITING]}

        self.results = data
        
        with open(self.path + 'results/' + self.filename + '.json', 'w') as f:
            json.dump(data, f)

