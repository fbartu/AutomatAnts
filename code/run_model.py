import sys
sys.path.append('~/research/AutomatAnts/code/')

#import numpy as np
#import time

from model import Model
from lattice import Lattice

import params

path = params.path
filename = params.file_name

def create_instance():
	environment = Lattice(params.n_agents, params.width, params.height, params.nest_node, params.food)
	return Model(params.n_agents, params.recruitment, environment, params.n_steps, path, filename)
	
if params.n_runs > 1:
	if params.run_parallel:
		def run_parallel(n_runs):
			results = {}			
			for i in range(n_runs):
				model = create_instance()	
				model.run_model()
				results[i] = model.results

			return results

		import multiprocessing as mp
		pool = mp.Pool(mp.cpu_count())
		results = pool.apply_async(run_parallel, [params.n_runs])
		results = results.get()
		pool.close()
		
	else:
		for run in range(params.n_runs):
			model = create_instance()
			model.run_model()

else:
	model = create_instance()
	model.run_model()



'''

#-----------------------------------------------------------
#
# Save Data
#
#-----------------------------------------------------------

### Save food data ###
time1     = []
f_nest    = []
f_site_1  = []
f_site_2  = []

### Save states/evolution data ###
W_count  = []
E_count  = []
R_count  = []
EM_count = []
RM_count = []

### Save Connectivity and interactions ###
time2   = []
k_count = []
i_count = []

### Save tag ###
tag_null     = []
tag_naif     = []
tag_informed = []


#-----------------------------------------------------------
#
# Save the data in files
#
#-----------------------------------------------------------

#Save States data
DataOut_states = np.column_stack((time1,W_count,E_count,R_count,EM_count,RM_count))
data_states = open(path + "results/"+params.file_name+"_state.dat", 'x')
data_states.close()
np.savetxt(path + "results/"+params.file_name+"_state.dat",DataOut_states)

#Save food data
DataOut_food = np.column_stack((time1,f_nest,f_site_1,f_site_2))
data_food = open(path + "results/"+params.file_name+"_food.dat", 'x')
data_food.close()
np.savetxt(path + "results/"+params.file_name+"_food.dat",DataOut_food)

#Save Connectivity
DataOut_k = np.column_stack((time2,k_count,i_count))
data_k = open(path + "results/"+params.file_name+"_k.dat", 'x')
data_k.close()
np.savetxt(path + "results/"+params.file_name+"_k.dat",DataOut_k)

#Save Tag
DataOut_tag = np.column_stack((time1,tag_null,tag_naif,tag_informed))
data_tag = open(path + "results/"+params.file_name+"_tag.dat", 'x')
data_tag.close()
np.savetxt(path + "results/"+params.file_name+"_tag.dat",DataOut_tag)

#Save run parameters to use on the plots
file = open(path + "results/"+params.file_name+"_info.dat",'a+')
file.write("Number of nodes = ")
file.write(str(model.G.number_of_nodes()))
file.write("\n")
file.write("Distance to food 1 = ")
file.write(str(len(model.short_paths.path[0])))
file.write("\n")
file.write("Distance to food 2 = ")
file.write(str(len(model.short_paths.path[1])))
file.write("\n")
file.close()


"""
G = model.G
print("Number of nodes =", G.number_of_nodes())
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos=pos, with_labels=True, 
		font_weight='bold', node_color= "Green")
plt.axis('off')
plt.show()
"""

'''
