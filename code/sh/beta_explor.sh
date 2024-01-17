#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin/$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

beta=0.5
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/beta_exploration/beta_0.5/ --filename beta_0.5 -n 100 -p beta=${beta/,/.}
beta=0.6
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/beta_exploration/beta_0.6/ --filename beta_0.5 -n 100 -p beta=${beta/,/.}
beta=0.7
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/2024/updated_params/beta_exploration/beta_0.7/ --filename beta_0.5 -n 100 -p beta=${beta/,/.}

# deactivate