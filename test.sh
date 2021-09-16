#!/bin/bash

# root directory
cd

. ~/research/automatenv/bin/activate
python3 ~/research/AutomatAnts/code/run_model.py -x test_average --nruns 8,True
# python3 ~/research/AutomatAnts/code/run_model.py -x test_average --nruns 1,False
deactivate

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 
#qsub -pe make 20 -l h_vmem=8G ~/research/AutomatAnts/code/run_model.sh