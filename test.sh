#!/bin/bash

# root directory
cd

# . ~/research/automatenv/bin/activate
G:
G:\research\automatenv\Scripts\activate
cd G:\research\AutomatAnts\code\
python G:\research\AutomatAnts\code\run_model.py -x phi0.9_ --nruns 20,True -p phi=0.9
# python3 G:\research\AutomatAnts\code\run_model.py -x polotest1 --nruns 1
# python3 ~/research/AutomatAnts/code/run_model.py -x test_average --nruns 1,False
deactivate

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 
#qsub -pe make 20 -l h_vmem=8G ~/research/AutomatAnts/code/run_model.sh