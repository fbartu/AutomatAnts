#!/bin/bash

# root directory
cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin/$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate
# normal model
python3 ~/research/AutomatAnts/code/run_model.py
# scanning parameter mu (noise in recruitment)
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.01
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.05
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.1
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.15
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.20
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.25
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.30
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.35
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.40
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.50
python3 ~/research/AutomatAnts/code/run_model.py -p mu=0.75
python3 ~/research/AutomatAnts/code/run_model.py -p mu=1
python3 ~/research/AutomatAnts/code/run_model.py -p mu=2
python3 ~/research/AutomatAnts/code/run_model.py -p mu=3
# serial recruitment
python3 ~/research/AutomatAnts/code/run_model.py -r HR,s 
deactivate

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 
#qsub -pe make 20 -l h_vmem=8G ~/research/AutomatAnts/code/run_model.sh