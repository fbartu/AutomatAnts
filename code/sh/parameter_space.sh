#!/bin/bash

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 

cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin/$LD_LIBRARY_PATH

# . ~/research/automatenv/bin/activate
A=0
for alpha in $(seq 0.00004 0.00004 0.0012)
do
    for beta in $(seq 0.05 0.1 3)
    do
    let "A=A+1"
    # python3 ~/research/AutomatAnts/code/Run.py --directory ~/research/AutomatAnts/results/parameter_space/ --filename "alpha=${alpha/,/.}_beta=${beta/,/.}" -n 100 -p "alpha=${alpha/,/.},beta=${beta/,/.}"
    # python3 ~/research/2022/ANTS/AutomatAnts/code/Run.py --directory ~/research/2022/ANTS/AutomatAnts/results/parameter_space/ --filename "alpha=${alpha/,/.}_beta=${beta/,/.}" -n 1 -p "alpha=${alpha/,/.},beta=${beta/,/.}"
    done
done
echo "$A"
# deactivate