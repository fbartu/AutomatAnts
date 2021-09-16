import os
import sys
path = os.getcwd().split('/')
if '/research/AutomatAnts/' in os.getcwd():
	while path[-2:] != ['research', 'AutomatAnts']:
		try:
			path.pop(-1)
		except:
			print('Path is not a directory !! Exiting program...')
			sys.exit(2)

	path = '/'.join(path)
	if path[-1] != '/':
		path = path + '/'
else:
	path = os.getcwd()+'/research/AutomatAnts/'
	if not os.path.isdir(path):
		print('Path is not a directory !! Exiting program...')
		sys.exit(2)
		
#path = 'G:/research/AutomatAnts/' # for debugging
#path = '/home/polfer/research/AutomatAnts/' # for debugging
folder = None
pathL = os.listdir(path + 'results/')
#pathL.remove('run_info')

# File name
""" if len(pathL) == 0:
	file_name = 'Run_1'
else:
	file_name = []
	for i in pathL:
		file_name.append(int(i.split('_')[1]))
	file_name = 'Run_' + str(max(file_name)+1) """

file_name = 'Test'

#Model
n_agents = 250
n_steps  = 500000 # 800000

nest_node = (0,22)
foodXvertex = 1
food = dict.fromkeys(
	[(6, 33), (6, 34), (7, 34), # patch 1
	(7, 33), (7, 32), (6, 32),
	(6, 11), (6, 12), (7, 12), # patch 2
	(7, 11), (7, 10), (6, 10)],
	foodXvertex)

# ADD distance in pairs, to change location of foodpatches...
# Two options would be necessary: mirrored and non-mirrored
'''
if 'd' in locals():
	if type(d) == tuple and sum(d) % 2 == 0:
		food = [(i[0] + d[0], i[1] + d[1]) for i in list(food.keys())]
'''

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
mu 		= 0.0 # noise in recruitment

#Number of different runs to average results
n_runs = 100
run_parallel = True

'''
Possible recruitment strategies:
No recruitment ('NR)          = 0
Individual recruitment ('IR') = 1
Hybrid recruitment ('HR')     = [0, 5]
Group recruitment ('GR')      = [3, 5]
'''
recruitment = 'HR'


'''
PARAMETERS INFO FILE


file = open(path + "results/run_info/"+file_name+"_info.dat","x")

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
file.write("mu = ")
file.write(str(mu))
file.write("\n")
file.write("\n")

file.write("Food pieces = ")
file.write(str(foodXvertex*len(food)))
file.write("\n")

file.write("Recruitment type = ")
file.write(str(recruitment))

file.close()


'''