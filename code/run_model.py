
import sys
sys.path.append('~/research/AutomatAnts/code/')

import numpy as np
import time

from model import Model
from lattice import Lattice

import params

path = params.path
filename = params.file_name

environment = Lattice(params.n_agents, params.width, params.height, params.nest_node, params.food)
model = Model(params.n_agents, params.recruitment, environment, params.n_steps, path, filename)
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
# Run the model
#
#-----------------------------------------------------------

#Initialize the model
model = AntModel(n_agents,width,height,nest_node,food_node,
				alpha,beta_1,beta_2,gamma_1,gamma_2,
				omega,eta)

print("Runing Program:")
print("Agents: ", n_agents)
print("Steps: " , n_steps)
print("Number of nodes: ", model.G.number_of_nodes())
print("Distance to food: ", len(model.short_paths.path[0]),",",len(model.short_paths.path[1]))
print("-----------")

start_time = time.time()
flag1 = True
flag2 = True

for i in range(n_steps):

	model.step()

	#Save food quantity
	time1.append(model.schedule.time)
	f_nest.append(model.food_counter.f_nest)
	f_site_1.append(sum(model.food_counter.f_site_1))
	f_site_2.append(sum(model.food_counter.f_site_2))

	#Save diferent states
	W_count.append(model.schedule.W_count)
	E_count.append(model.schedule.E_count)
	R_count.append(model.schedule.R_count)
	EM_count.append(model.schedule.EM_count)
	RM_count.append(model.schedule.RM_count)

	#Save tags evolution
	tag_null.append(model.schedule.tag_null)
	tag_naif.append(model.schedule.tag_naif)
	tag_informed.append(model.schedule.tag_informed)

	#Snapshot of the connectivity and interactions
	if (i % 1000 == 0):
		time2.append(model.schedule.time)
		k_count.append(model.schedule.k)	
		i_count.append(model.schedule.interactions)

	#Prints / runing program
	if (sum(model.food_counter.f_site_1) == (params.food_site_1 - 1) and flag1):
		flag1 = False
		print("(Site 1 Encounter)")
	if (sum(model.food_counter.f_site_2) == (params.food_site_2 - 1) and flag2):
		flag2 = False
		print("(Site 2 Encounter)")
		
	if (i % 10000 == 0):
		print("Step:" , int(i/10000))
	if (i == n_steps-1):
		print("Successful run - End")
		print("Execution time = -- %s seconds --" % (time.time() - start_time))

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
