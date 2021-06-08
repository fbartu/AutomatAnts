#-----------------------------------------------------------
#
# Script to explore different parameter values
#
#-----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

from model import *
import params

#-----------------------------------------------------------
#
# Model Parameters
#
#-----------------------------------------------------------

n_agents = params.n_agents
n_steps  = params.n_steps

#Lattice size
width    = params.width 
height   = params.height 

#Parameters
alpha   = params.alpha
beta_1  = params.beta_1
beta_2  = params.beta_2
gamma_1 = params.gamma_1
gamma_2 = params.gamma_2
omega   = params.omega
eta     = params.eta

#Key Position
nest_node = params.nest_node
food_cluster_1 = params.food_cluster_1 
food_cluster_2 = params.food_cluster_2
food_node = [food_cluster_1,food_cluster_2]

#-----------------------------------------------------------
#
# Run the model
#
#-----------------------------------------------------------

print("Runing Program:")
start_time = time.time()

eta_list    = []
n_mean_list = []
n_std_list  = []

run_steps = 50
eta = 0.0

for j in range(run_steps):

	eta_list.append(eta)
	
	#Initialize the model
	model = AntModel(n_agents,width,height,nest_node,food_node,
					alpha,beta_1,beta_2,gamma_1,gamma_2,
					omega,eta)


	#Save states/evolution data 
	E_count = []

	for i in range(n_steps):
		model.step()
		if (model.schedule.time > 2500):
			E_count.append(model.schedule.E_count)

	E_count_array = np.array(E_count)
	n_mean_list.append(np.mean(E_count_array))
	n_std_list.append(np.std(E_count_array))

	#Prints while runing the program
	if (j % 1 == 0):
		print("Step: ", int(j/1), "eta: ", eta, "mean: ",np.mean(E_count_array), np.std((E_count_array)))
	if (j == run_steps-1):
		print("Successful run - End")
		print("Execution time = -- %s seconds --" % (time.time() - start_time))

	eta += 0.01

#-----------------------------------------------------------
#
# Save the data in files
#
#-----------------------------------------------------------

#Save States data
DataOut_eta = np.column_stack((eta_list,n_mean_list,n_std_list))
np.savetxt("Results/"+params.file_name+"_eta.dat",DataOut_eta)

#Save run parameters to use on the plots
file = open(params.file_name+"_info.dat",'a+')
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