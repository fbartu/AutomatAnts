#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin/$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/nest_avoid_explor/ --filename ballistic -n 100 -m bal
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/nest_avoid/ --filename ballistic -n 100 -m bal
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/nest_avoid_bias_1.5/ --filename ballistic -n 100 -m bal
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/random/ --filename random -n 100 -m random
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/modified_mot/ --filename modified_mot -n 100 -m exp

# deactivate