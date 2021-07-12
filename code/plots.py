
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------
#
# Plots - Read Data
#
#-----------------------------------------------------------

file_name = "Run_33"

t, fn, f1, f2 = np.loadtxt("Results/"+file_name+'_food.dat',unpack=True)
t, w, e, r, em, rm = np.loadtxt("Results/"+file_name+'_state.dat',unpack=True)
t2, k, interactions = np.loadtxt("Results/"+file_name+'_k.dat',unpack=True)

###Tag###
t, tag1, tag2, tag3 = np.loadtxt("Results/"+file_name+'_tag.dat',unpack=True)

#Read Parameters from input info file
with open("Results/"+file_name+'_info.dat') as f:
    lines = f.readlines()

num_agents = int(lines[0].split("=")[1].strip('\n'))
num_nodes  = int(lines[27].split("=")[1].strip('\n'))
dist_1     = int(lines[28].split("=")[1].strip('\n'))
dist_2     = int(lines[29].split("=")[1].strip('\n'))

#-----------------------------------------------------------
#
# States - System Evolution
# 
#-----------------------------------------------------------

fig,ax=plt.subplots()
plt.title("System Evolution")
plt.xlabel("Gillespie Time (Minutes)")
plt.ylabel("Population")
#ax.plot(t/60.0,w,label="Waiting")
ax.plot(t/60.0,e,label="Exploring")
ax.plot(t/60.0,r,label="Recruting")
ax.plot(t/60.0,em,label="Exploring_m")
ax.plot(t/60.0,rm,label="Recruting_m")
plt.grid()
leg = ax.legend()
plt.show()

#-----------------------------------------------------------
#
# Tags - System Evolution
# 
#-----------------------------------------------------------

fig,ax=plt.subplots()
plt.title("System Evolution - Tag")
plt.xlabel("Gillespie Time (Minutes)")
plt.ylabel("Population")
#ax.plot(t/60.0,w,label="Waiting")
ax.plot(t/60.0,tag2,label="Naif")
ax.plot(t/60.0,tag3,label="Informed")
plt.grid()
leg = ax.legend()
plt.show()

#-----------------------------------------------------------
#
# Food Evolution
#
#-----------------------------------------------------------

fig,ax = plt.subplots()
plt.title("Food Evolution")
plt.xlabel("Gillespie Time (Minutes)")
plt.ylabel("Food Quantity")

#Polynomial fit
"""
poly_deg_1 = 8
poly_deg_2 = 8
x = np.linspace(t.min(), t.max(), 500) 
coefs_fn = np.polyfit(t, fn, poly_deg_1)
coefs_f1 = np.polyfit(t, f1, poly_deg_2)
coefs_f2 = np.polyfit(t, f2, poly_deg_2)
fn_poly = np.polyval(coefs_fn, x)
f1_poly = np.polyval(coefs_f1, x)
f2_poly = np.polyval(coefs_f2, x)

ax.plot(x, fn_poly, label="f_nest - polynomial fit")
ax.plot(x, f1_poly, label="f_nest - polynomial fit")
ax.plot(x, f2_poly, label="f_nest - polynomial fit")
"""

ax.plot(t/60.0,fn,label="f_nest")
ax.plot(t/60.0,f1,label="f_site_1")
ax.plot(t/60.0,f2,label="f_site_2")
plt.grid()
leg = ax.legend()
plt.show()

#-----------------------------------------------------------
#
# Density (rho)
#
#-----------------------------------------------------------

dens = (num_agents - w)               #Density
#dens = (num_agents - w)/(num_agents) #Normalized density
#dens = (num_agents - w)/(num_nodes)  #Spatial density
#plt.ylim(0,1)
plt.title("Density Evolution")
plt.xlabel("Gillespie Time")
plt.ylabel("Density")
plt.grid()
plt.plot((t/60.0),dens)
plt.show()

#-----------------------------------------------------------
#
# Connectivity (k)
#
#-----------------------------------------------------------

from scipy.signal import savgol_filter

connectivity = k            #Connectivity
#connectivity = k/dist_1    #Spatial connectivity

fig,ax=plt.subplots()
plt.title("k Evolution")
plt.xlabel("Gillespie Time")
plt.ylabel("Connectivity")

#Filtered signal using Savitzky–Golay filter
#y_filt = savgol_filter(connectivity, 51, 2) # window size 51, polynomial order 2
#ax.plot(t2,y_filt, label="Filtered signal")
ax.plot(t2/60.0, connectivity, label="True signal") 
leg = ax.legend()
plt.grid()
plt.show()

#-----------------------------------------------------------
#
# Interactions (i)
#
#-----------------------------------------------------------

fig,ax=plt.subplots()
plt.title("Interactions evolution")
plt.xlabel("Gillespie Time")
plt.ylabel("Interactions")
ax.plot(t2/60.0,interactions-1, label=" ") #We substract 1 unit to not consider the nest
plt.grid()
plt.show()

#-----------------------------------------------------------
#
# etta vs n_mean (At the stationary state)
#
#-----------------------------------------------------------

"""
etta,n_mean = np.loadtxt("Results/"+file_name+'_eta.dat',unpack=True)

fig,ax = plt.subplots()
plt.title("Parameter etta")
plt.xlabel("etta")
plt.ylabel("n_explorers_mean")
plt.grid()
ax.scatter(etta,n_mean, s=20, marker='.', label="n_agents")
leg = ax.legend()
plt.show()
"""

#-----------------------------------------------------------
#
# No food Case (or all the food found) / Stationary State
# 
#-----------------------------------------------------------


#-------------------
# Autocorrelation function (ACF)
#-------------------


stationary = (num_agents - w)
time_new = t.tolist()
stationary_new = stationary.tolist()

#Delete first elements till stationary state
n = 1000
new_time = time_new[n:]
new_stationary = stationary_new[n:]

array_time = np.array(new_time)
array_stationary = np.array(new_stationary)

mean = np.mean(array_stationary)
std = np.std(array_stationary)
print(mean,std)

def acf(x, length=30000):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])

llista_acf = acf(array_stationary)
lag_2 = np.arange(1,30000+1)
plt.bar(lag_2,llista_acf)
plt.title("ACF")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.axhline(y=0.05, color='black', linestyle='dashed')
plt.grid()
plt.show()


#----------------------
# Fourier Spectrum, caracteristic frequency (w)
#----------------------

#WIP


#-----------------------------------------------------------
#
# Multiple runs
#
#-----------------------------------------------------------

"""
mean_t,mean_d,std_d,mean_k,std_k,mean_i,std_i = np.loadtxt("Results/"+file_name+"_evolution_runs.dat",unpack=True)

#Density
plt.errorbar(np.array(mean_t)/60.0,num_agents-np.array(mean_d), xerr = 0.0, yerr = std_d, errorevery = 5000)
plt.grid()
plt.axhline(color='black', lw=0.5)
plt.axvline(color='black', lw=0.5)
plt.xlabel("Time (Minutes)")
plt.ylabel("Num. of Agents")
plt.show()

#Connectivity
plt.errorbar(np.array(mean_t)/60.0,np.array(mean_k)-1.0, xerr = 0.0, yerr = std_k, errorevery = 10000)
plt.grid()
plt.axhline(color='black', lw=0.5)
plt.axvline(color='black', lw=0.5)
plt.xlabel("Time (Minutes)")
plt.ylabel("Connectivity")
plt.show()

#Interactions
plt.errorbar(np.array(mean_t)/60.0,np.array(mean_i)-1.0, xerr = 0.0, yerr = std_i, errorevery = 10000)
plt.grid()
plt.axhline(color='black', lw=0.5)
plt.axvline(color='black', lw=0.5)
plt.xlabel("Time (Minutes)")
plt.ylabel("Interactions")
plt.show()

#Phase space 
plt.scatter(num_agents-np.array(mean_d),np.array(mean_k)-1.0, s=10, marker='.')
plt.grid()
plt.axhline(color='black', lw=0.5)
plt.axvline(color='black', lw=0.5)
plt.xlabel("Density")
plt.ylabel("Connectivity")
plt.show()

"""
#-----------------------------------------------------------
#
# Plot grid
# Extract the positions and Pass the positions while drawing
# Plot in run_model.py, paste the below code in the 
# run_model.py script
#
#-----------------------------------------------------------
"""
G = model.G
print("Number of nodes =", G.number_of_nodes())
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos=pos, with_labels=True, 
		font_weight='bold', node_color= "Green")
plt.axis('off')
plt.show()
"""
