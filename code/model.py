from gillespie import GillespieAlgorithm
from agent import *
import json
import numpy as np

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
           
        # self.time2seconds()
        self.save_data()


    def time2seconds(self):
        self.T = [t / 2.0 for t in self.T] # convert from frames to seconds (FPS = 2)

    def time2minutes(self):
        self.T = [t / (2.0 * 60) for t in self.T]

    def retrieve_positions(self):
        result = {'agent': [],'pos': [],'t': [], 'tag': [], 'mia': []}
        for i in range(len(self.agents)):
            result['agent'].extend([i] * len(self.agents[i].path))
            result['pos'].extend(self.agents[i].path)
            result['t'].extend(list(map(self.T.__getitem__, np.where(self.sample == np.array([i]))[0])))
            result['tag'].extend(self.agents[i].tag)
            result['mia'].append(self.agents[i].MIA)
        
        return result

    def save_data(self):

        # data = {'Time (s)': self.T,
        # 'Connectivity': self.K,
        # 'N': self.N,
        # 'Interactions': self.I,
        # 'Food in nest': self.F,
        # 'Informed': self.population[Tag.INFORMED],
        # 'Patches of food': self.metrics.efficiency(self.tfood),
        # 'Number of explorers': self.population[State.EXPLORING],
        # 'Number of recruiters': self.population[State.RECRUITING],
        # 'Positions': self.retrieve_positions()}

        self.results = [{'Time (s)': self.T,
        'Connectivity': self.K,
        'N': self.N,
        'Interactions': self.I,
        'Food in nest': self.F,
        'Informed': self.population[Tag.INFORMED]},
        self.metrics.efficiency(self.tfood),
        self.retrieve_positions()]

        # self.results = {'data': [data['Time (s)'], data['Connectivity'], data['N'],
        # data['Interactions'], data['Food in nest'], data['Informed'],
        # data['Number of explorers'], data['Number of recruiters']],
        # 'food': data['Patches of food'], 'pos': data['Positions']}
        
    def data2json(self, folder = '', filename = 'Run_1'):    
        
        if not hasattr(self, 'results'):
            self.save_data()

        with open(self.path + 'results/' + folder + filename + '_data.json', 'w') as f:
            json.dump(self.results[0], f)

        with open(self.path + 'results/' + folder + filename + '_food.json', 'w') as f:
            json.dump(self.results[1], f)

        with open(self.path + 'results/' + folder + filename + '_pos.json', 'w') as f:
            json.dump(self.results[2], f)


