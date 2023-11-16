#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_cluster.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin/$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_16/ --food det --filename theta16 -n 100 -p Theta="10**-16"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_17/ --food det --filename theta17 -n 100 -p Theta="10**-17"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_18/ --food det --filename theta18 -n 100 -p Theta="10**-18"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_19/ --food det --filename theta19 -n 100 -p Theta="10**-19"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_20/ --food det --filename theta20 -n 100 -p Theta="10**-20"

deactivate