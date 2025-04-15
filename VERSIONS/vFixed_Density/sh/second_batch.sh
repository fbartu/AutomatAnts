#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin:$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/second_batch/ --filename "homing_rho_0.5_N_40" -n 100 -p "rho=0.5;N=40;init_position=nest;homing_behavior=True" &
python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/second_batch/ --filename "agg_rho_0.5_N_40" -n 100 -p "rho=0.5;N=40;init_position=targeted;homing_behavior=False" &
python3 ~/research/AutomatAnts/VERSIONS/vFixed_Density/run_until.py --directory ~/research/AutomatAnts/results/2025/agent_density/second_batch/ --filename "random_rho_0.5_N_40" -n 100 -p "rho=0.5;N=40;init_position=random;homing_behavior=False"

