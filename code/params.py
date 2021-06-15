
#-----------------------------------------------------------
#
# Model Parameters 
#
#-----------------------------------------------------------

import os

path = os.getcwd()+'/research/AutomatAnts/'
pathL = os.listdir(path + '/results/')

# File name
if len(pathL) == 0:
	file_name = 'Run_1'
else:
	file_name = []
	for i in pathL:
		file_name.append(int(i.split('_')[1]))
	file_name = 'Run_' + str(max(file_name)+1)

#File_name
# file_name = "Run_33"

#Model
n_agents = 100 #250 #500 
n_steps  = 10000 #800000 #200000 #800000

#Lattice size
width    = 22   
height   = 13  

#Parameters
alpha   = 0.05
beta_1  = 1.0
beta_2  = 1.0
gamma_1 = 1.0
gamma_2 = 1.0
omega   = 2.5  
eta     = 0.035

#Number of different tuns to average results
run_steps = 20

#Recruitment (True/False) #No recruitment: 0
recruitment = True

#Type of recruitment (Only one can be True)
individual_recruitment = False  #Individual recruitment: 1
group_recruitment      = True  #Group recruiment: 3-5
hybrid_recruitment     = False   #Hybrid recruiment: 0-5

#Key Position
nest_node = (7,0)

#Food quantity of each node on the food cluster
food_site_1_list = [0,0,0,0,0,0]
food_site_2_list = [0,0,0,0,0,0]

#[1,1,1,1,1,1]
#[0,0,0,0,0,0]

##Food clusters experiments  (Dist 16,17) (16,5)
#food_cluster_1 = [(5,14),(5,15),(5,16),(4,14),(4,15),(4,16)]    
#food_cluster_2 = [(9,13),(9,14),(9,15),(10,13),(10,14),(10,15)] 

##Food cluter distancia 5 (7 contant niu i menjar) (4,5)
#food_cluster_1 = [(5,4),(5,5),(5,6),(4,6),(4,4),(4,5)]
#food_cluster_2 = [(9,3),(9,4),(9,5),(10,5),(10,4),(10,3)]

##Food cluters distancia 11 (13 contant niu i menjar) (10,5)
#food_cluster_1 = [(5,10),(5,11),(5,12),(4,12),(4,1),(4,10)]
#food_cluster_2 = [(9,9),(9,10),(9,11),(10,11),(10,10),(10,9)]

##Food clusters distancia 25 (27 contant niu i menjar) (24,5)
#food_cluster_1 = [(5,24),(5,25),(5,26),(4,26),(4,25),(4,24)]
#food_cluster_2 = [(9,23),(9,24),(9,25),(10,25),(10,24),(10,23)]

##Food cluster distancia 19 (21 contant niu i menjar)
#food_cluster_1 = [(5,18),(5,19),(5,20),(4,20),(4,19),(4,20)]
#food_cluster_2 = [(9,17),(9,18),(9,19),(10,19),(10,18),(10,17)]

##Food cluster distancia 31 (33 contant niu i menjar)
food_cluster_1 = [(5,30),(5,31),(5,32),(4,32),(4,31),(4,30)]
food_cluster_2 = [(9,29),(9,30),(9,31),(10,31),(10,30),(10,29)]

#--------------------------------
food_site_1 = sum(food_site_1_list)
food_site_2 = sum(food_site_2_list)

#-----------------------------------------------------------
#
# Save data in a file
#
#-----------------------------------------------------------

file = open(path + "results/"+file_name+"_info.dat","x")

file.write("n_agents = ")
file.write(str(n_agents))
file.write("\n")
file.write("n_steps = ")
file.write(str(n_steps))
file.write("\n")
file.write("\n")

file.write("Lattice size")
file.write("\n")
file.write("width = ")
file.write(str(width))
file.write("\n")
file.write("height = ")
file.write(str(height))
file.write("\n")
file.write("\n")

file.write("Model Parameters")
file.write("\n")
file.write("alpha  = ")
file.write(str(alpha))
file.write("\n")
file.write("beta_1 = ")
file.write(str(beta_1))
file.write("\n")
file.write("beta_2 = ")
file.write(str(beta_2))
file.write("\n")
file.write("gamma_1 = ")
file.write(str(gamma_1))
file.write("\n")
file.write("gamma_2 = ")
file.write(str(gamma_2))
file.write("\n")
file.write("omega = ")
file.write(str(omega))
file.write("\n")
file.write("eta = ")
file.write(str(eta))
file.write("\n")
file.write("\n")

file.write("Key Positions")
file.write("\n")
file.write("nest_node = ")
file.write(str(nest_node))
file.write("\n")
file.write("food_cluster_1 = ")
file.write(str(food_cluster_1))
file.write("\n")
file.write("food_cluster_2 = ")
file.write(str(food_cluster_2))
file.write("\n")
file.write("food_site_1 = ")
file.write(str((food_site_1_list)))
file.write("\n")
file.write("food_site_2 = ")
file.write(str((food_site_2_list)))
file.write("\n")
file.write("\n")
file.write("Recruitment:")
file.write(str((recruitment)))
file.write("\n")
file.write("Individual Recruitment:")
file.write(str((individual_recruitment)))
file.write("\n")
file.write("Group Recruitment:")
file.write(str((group_recruitment)))
file.write("\n")
file.write("Hybrid Recruitment:")
file.write(str((hybrid_recruitment)))
file.write("\n")

file.close()


