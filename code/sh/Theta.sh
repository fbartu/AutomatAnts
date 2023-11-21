#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_cluster.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin/$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate

# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_05/ --food det --filename theta05 -n 100 -p Theta="10**-5"
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_08/ --food det --filename theta08 -n 100 -p Theta="10**-8"
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_10/ --food det --filename theta10 -n 100 -p Theta="10**-10"
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_13/ --food det --filename theta13 -n 100 -p Theta="10**-13"

# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_16/ --food det --filename theta16 -n 100 -p Theta="10**-16"
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_17/ --food det --filename theta17 -n 100 -p Theta="10**-17"
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_18/ --food det --filename theta18 -n 100 -p Theta="10**-18"
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_19/ --food det --filename theta19 -n 100 -p Theta="10**-19"
# python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_20/ --food det --filename theta20 -n 100 -p Theta="10**-20"

python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_25/ --food det --filename theta25 -n 100 -p Theta="10**-25"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_30/ --food det --filename theta30 -n 100 -p Theta="10**-30"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_50/ --food det --filename theta50 -n 100 -p Theta="10**-50"

python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_35/ --food det --filename theta35 -n 100 -p Theta="10**-35"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_40/ --food det --filename theta40 -n 100 -p Theta="10**-40"
python3 ~/research/AutomatAnts/code/run_cluster.py --directory ~/research/AutomatAnts/results/with_recruitment/parameters/theta/theta_45/ --food det --filename theta45 -n 100 -p Theta="10**-45"



deactivate