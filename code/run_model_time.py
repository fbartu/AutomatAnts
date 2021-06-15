import sys
sys.path.append('~/research/AutomatAnts/code/')

import numpy as np
import matplotlib.pyplot as plt
import time

from model import *
import params

path = params.path

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
# Save Data
#
#-----------------------------------------------------------

t_0 = 0.0
t_1 = [] #1rst encounter times (s)
t_2 = [] #Last encounter times (s)
t_3 = [] #Tot recollit al niu (temps) (s)

time_all    = [] #Vetor of vectors with the time of all the data
density_all = [] #Vector of vectors with all the density data

connectivity_all = [] #Vector of vectors with the connectivy of all the data
interactions_all = [] #Vector of vectors with the interacions of all the data

tag_naif_all     = [] #Vector of vectors with the tags naig of all the data
tag_informed_all = [] #Vector of vectors with the tags informed of all the data

food_nest_all   = [] #Vector of vectors with all the food data
food_site_1_all = []
food_site_2_all = []
#-----------------------------------------------------------
#
# Main loop
#
#-----------------------------------------------------------

start_time = time.time()

#Run info
run_steps = params.run_steps
seed = 1

for j in range(run_steps):
	
	print("Step ", j+1)

	#Random number generator initialization
	random.seed(seed)
	
	#vectors to save data
	time_run = []
	W_run    = []
	k_run    = []
	i_run    = []

	tag_naif_run     = []
	tag_informed_run = []

	food_nest_run   = []
	food_site_1_run = []
	food_site_2_run = []

	#Flags to exit loops and save data
	flag1 = True
	flag2 = True
	flag3 = True

	#Initialize the model
	model = AntModel(n_agents,width,height,nest_node,food_node,
					alpha,beta_1,beta_2,gamma_1,gamma_2,
					omega,eta)


	for i in range(n_steps):
		model.step()
		
		time_run.append(model.schedule.time)
		W_run.append(model.schedule.W_count)
		k_run.append(model.schedule.k)
		i_run.append(model.schedule.interactions)

		food_nest_run.append(model.food_counter.f_nest)
		food_site_1_run.append(sum(model.food_counter.f_site_1))
		food_site_2_run.append(sum(model.food_counter.f_site_2))

		tag_naif_run.append(model.schedule.tag_naif)
		tag_informed_run.append(model.schedule.tag_informed)

		#First Encounter
		if ((sum(model.food_counter.f_site_1) == (params.food_site_1 - 1)) and flag1):
			flag1 = False
			t_1.append(model.schedule.time)
		if ((sum(model.food_counter.f_site_2) == (params.food_site_2 - 1)) and flag1):
			flag1 = False
			t_1.append(model.schedule.time)
		#Last encounter
		if ((sum(model.food_counter.f_site_1) == 1) and flag2):
			flag2 = False
			t_2.append(model.schedule.time)
		if ((sum(model.food_counter.f_site_2) == 1) and flag2):
			flag2 = False
			t_2.append(model.schedule.time)
		#All food recollected
		if ((model.food_counter.f_nest) == 
			(params.food_site_1 + params.food_site_2)  and flag3):
			flag3 = False
			t_3.append(model.schedule.time)
	
		
	#Save data
	time_all.append(time_run)
	density_all.append(W_run)
	connectivity_all.append(k_run)
	interactions_all.append(i_run)

	#Food
	food_nest_all.append(food_nest_run)
	food_site_1_all.append(food_site_1_run)
	food_site_2_all.append(food_site_2_run)

	#Tags
	tag_naif_all.append(tag_naif_run)
	tag_informed_all.append(tag_informed_run)

	seed += 1

print("Execution time = -- %s seconds --" % (time.time() - start_time))


#-----------------------------------------------------------
#
# Data treatment + Saving in files
#
#-----------------------------------------------------------

#----------
#Time mean + standard deviation
#----------

t_1_mean = np.mean(np.array(t_1))
t_2_mean = np.mean(np.array(t_2))
t_3_mean = np.mean(np.array(t_3))

t_1_std = np.std(np.array(t_1))
t_2_std = np.std(np.array(t_2))
t_3_std = np.std(np.array(t_3))

print(t_1_mean,t_1_std)
print(t_2_mean,t_2_std)
print(t_3_mean,t_3_std)

#---------
#Density
#---------

#Mean + std of each time step
#Tranposed list
time_list = list(map(list,zip(*time_all)))
density_list = list(map(list,zip(*density_all)))
connectivity_list = list(map(list,zip(*connectivity_all)))
interactions_list = list(map(list,zip(*interactions_all)))

#Food
food_nest_list   = list(map(list,zip(*food_nest_all)))
food_site_1_list = list(map(list,zip(*food_site_1_all)))
food_site_2_list = list(map(list,zip(*food_site_2_all)))

#Tags
tag_naif_list     = list(map(list,zip(*tag_naif_all)))
tag_informed_list = list(map(list,zip(*tag_informed_all)))

mean_t = []
#Density
mean_d = []
std_d  = []
#Connectivity
mean_k = []
std_k  = []
#Interactions
mean_i = []
std_i  = []
#tag
tag_naif         = []
tag_naif_std     = []
tag_informed     = []
tag_informed_std = []
#Food
food_nest        = []
food_nest_std    = []
food_site_1      = []
food_site_1_std  = []
food_site_2      = []
food_site_2_std  = []

for i in range(len(density_list)):
	mean_t.append(np.mean(np.array(time_list[i])))

	mean_d.append(np.mean(np.array(density_list[i])))
	std_d.append(np.std(np.array(density_list[i])))

	mean_k.append(np.mean(np.array(connectivity_list[i])))
	std_k.append(np.std(np.array(connectivity_list[i])))

	mean_i.append(np.mean(np.array(interactions_list[i])))
	std_i.append(np.std(np.array(interactions_list[i])))

	food_nest.append(np.mean(np.array(food_nest_list[i])))
	food_nest_std.append(np.std(np.array(food_nest_list[i])))
	food_site_1.append(np.mean(np.array(food_site_1_list[i])))      
	food_site_1_std.append(np.std(np.array(food_site_1_list[i])))
	food_site_2.append(np.mean(np.array(food_site_2_list[i])))
	food_site_2_std.append(np.std(np.array(food_site_2_list[i])))

	tag_naif.append(np.mean(np.array(tag_naif_list[i])))
	tag_naif_std.append(np.std(np.array(tag_naif_list[i])))
	tag_informed.append(np.mean(np.array(tag_informed_list[i])))
	tag_informed_std.append(np.std(np.array(tag_informed_list[i])))
#-----------------------------------------------------------
#
# Save Data
#
#-----------------------------------------------------------


DataOut_runs_states = np.column_stack((mean_t,mean_d,std_d,mean_k,std_k,mean_i,std_i))
data_runs_states = open(path + "results/"+params.file_name+"_evolution_runs.dat",'x')
data_runs_states.close()
np.savetxt(path + "results/"+params.file_name+"_evolution_runs.dat",DataOut_runs_states)

DataOut_runs_time = np.column_stack((t_1_mean,t_1_std,t_2_mean,t_2_std,t_3_mean,t_3_std))
data_runs_time = open(path + "results/"+params.file_name+"_time_runs.dat",'x')
data_runs_time.close()
np.savetxt(path + "results/"+params.file_name+"_time_runs.dat",DataOut_runs_time)

DataOut_runs_food = np.column_stack((mean_t,food_nest,food_nest_std,food_site_1,food_site_1_std,food_site_2,food_site_2_std))
data_runs_food = open(path + "results/"+params.file_name+"_time_food_runs.dat",'x')
data_runs_food.close()
np.savetxt(path + "results/"+params.file_name+"_time_food_runs.dat",DataOut_runs_food)

DataOut_runs_tag = np.column_stack((mean_t,tag_naif,tag_naif_std,tag_informed,tag_informed_std))
data_runs_tag = open(path + "results/"+params.file_name+"_time_tag_runs.dat",'x')
data_runs_tag.close()
np.savetxt(path + "results/"+params.file_name+"_time_tag_runs.dat",DataOut_runs_tag)


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
file.write("Number of different runs = ")
file.write(str(run_steps))
file.write("\n")
file.close()
