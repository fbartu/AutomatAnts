#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin:$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

for rho in $(seq 0.001 0.1 1.001)
do
    float_rho=$(echo $rho | tr ',' '.')
    for N in 5 10 20 30 40 50 75 100 150 200 400 620 1240 1860 2480 3100
    do
        python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/with_homing/ --filename "rho_${rho}_N_${N}" -n 100 -p "rho=${rho/,/.};N=${N};init_position=nest"

    done
done
