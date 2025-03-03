#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin:$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

# for rho in $(seq 0.001 0.1 1.001)
for rho in 0 0.5 1
do
    float_rho=$(echo $rho | tr ',' '.')
    for N in 5 10 20 30 40 50 75 100 150 200 400 620
    do
        # for r in 0.05 0.1 0.2 0.5 1 2 3
        for r in 0.002 0.004 0.008 0.012 0.016 0.020 0.030 0.040 0.060 0.081 0.161 0.250
        do
            # float_r=$(echo $r | tr ',' '.')
            python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/with_memory/ --filename "rho_${rho}_N_${N}_memory_${r}_random" -n 100 -p "rho=${rho/,/.};N=${N};init_position=random;memory_rate=${r/,/.};homing_behavior=False"
            python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/with_memory/ --filename "rho_${rho}_N_${N}_memory_${r}_homing" -n 100 -p "rho=${rho/,/.};N=${N};init_position=nest;memory_rate=${r/,/.};homing_behavior=True"
        done
    done
done
