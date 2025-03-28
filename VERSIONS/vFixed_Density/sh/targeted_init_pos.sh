#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin:$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

for rho in 0 0.2 0.5 0.8 1 # $(seq 0.001 0.1 1.001)
do
    # for N in 5 10 20 30 40 50 75 100 150 200 400 620
    for N in 10 20 30 40 50 75 100
    do
        for R in 3 4 5 6 7 8 9 10
        do
            # --- ITERATIVE MAX DISTANCE --- # 
            python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/R/ --filename "rho_${rho}_N_${N}_R${R}_scouts" -n 100 -p "rho=${rho/,/.};N=${N};init_position=targeted;homing_behavior=False;R=${R};agg_scouts=True"
            python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/R/ --filename "rho_${rho}_N_${N}_R${R}_recruits" -n 100 -p "rho=${rho/,/.};N=${N};init_position=targeted;homing_behavior=False;R=${R};agg_scouts=False"

        # --- MAX DISTANCE = 4 --- #
        # with clustered recruits
        # python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/asymmetric_positioning/ --filename "rho_${rho}_N_${N}" -n 100 -p "rho=${rho/,/.};N=${N};init_position=targeted;homing_behavior=False"
        # with clustered scouts --> change in code
        # python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/asymmetric_positioning/ --filename "rho_${rho}_N_${N}_scouts" -n 100 -p "rho=${rho/,/.};N=${N};init_position=targeted;homing_behavior=False"


        # --- MAX DISTANCE = 2 --- # 
        # with clustered recruits
        # python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/asymmetric_positioning/ --filename "rho_${rho}_N_${N}_R2_recruits" -n 100 -p "rho=${rho/,/.};N=${N};init_position=targeted;homing_behavior=False"
        # with clustered scouts --> change in code
        # python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/asymmetric_positioning/ --filename "rho_${rho}_N_${N}_R2_scouts" -n 100 -p "rho=${rho/,/.};N=${N};init_position=targeted;homing_behavior=False"
        done
    done
done
