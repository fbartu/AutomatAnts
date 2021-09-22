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

        print('+++ RUNNING MODEL +++')
        for i in list(range(self.steps)):
            self.step()
           
        print('Model completed... Saving results !')

        self.save_data()
        print('+++ Results saved +++')

    def time2minutes(self):
        self.T = [t / 60.0 for t in self.T]

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

        self.results = [{'Time (s)': self.T,
        'Connectivity': self.K,
        'N': self.N,
        'Interactions': self.I,
        'Food in nest': self.F,
        'Informed': self.population[Tag.INFORMED],
        'Waiting': self.population[State.WAITING],
        'Carrying food': list(map(lambda x, y: x+y,
         self.population[State.EXPLORING_FOOD],
          self.population[State.RECRUITING_FOOD])),
        'Exploring': self.population[State.EXPLORING],
        'Recruiting': self.population[State.RECRUITING]},
        self.metrics.efficiency(self.tfood),
        self.retrieve_positions()]
        
    def data2json(self, folder = '', filename = 'Run_1', get_pos = False):    
        
        if not hasattr(self, 'results'):
            self.save_data()

        with open(self.path + 'results/' + folder + filename + '_data.json', 'w') as f:
            json.dump(self.results[0], f)

        with open(self.path + 'results/' + folder + filename + '_food.json', 'w') as f:
            json.dump(self.results[1], f)

        if get_pos:
            
            with open(self.path + 'results/' + folder + filename + '_pos.json', 'w') as f:
                json.dump(self.results[2], f)


