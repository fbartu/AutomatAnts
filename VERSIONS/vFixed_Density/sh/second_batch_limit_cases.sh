#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin:$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

for rho in 0 0.5 1
do
    for N in 20 40 60
    do
        python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/second_batch/last/ --filename "homing_rho_${rho}_N_${N}" -n 100 -p "rho=${rho};N=${N};init_position=nest;homing_behavior=True"
        python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/second_batch/last/ --filename "agg_rho_${rho}_N_${N}" -n 100 -p "rho=${rho};N=${N};init_position=targeted;homing_behavior=False"
        python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/second_batch/last/ --filename "random_rho_${rho}_N_${N}" -n 100 -p "rho=${rho};N=${N};init_position=random;homing_behavior=False"
    done
done


