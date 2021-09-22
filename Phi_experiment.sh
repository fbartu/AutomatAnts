#!/bin/bash

# root directory
cd

export PATH=/home/soft/python-3.9.5/bin:$PATH
export LD_LIBRARY_PATH=/home/soft/python-3.9.5/bin/$LD_LIBRARY_PATH

. ~/research/automatenv/bin/activate
# normal model
python3 ~/research/AutomatAnts/code/run_model.py -x control_phi
# scanning parameter phi (noise in recruitment)
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.01 -x phi0.01
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.05 -x phi0.05
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.10 -x phi0.10
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.15 -x phi0.15
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.20 -x phi0.20
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.25 -x phi0.25
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.30 -x phi0.30
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.35 -x phi0.35
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.40 -x phi0.40
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.50 -x phi0.50
python3 ~/research/AutomatAnts/code/run_model.py -p phi=0.75 -x phi0.75
python3 ~/research/AutomatAnts/code/run_model.py -p phi=1.00 -x phi1.00
python3 ~/research/AutomatAnts/code/run_model.py -p phi=2.00 -x phi2.00
python3 ~/research/AutomatAnts/code/run_model.py -p phi=3.00 -x phi3.00
# serial recruitment
python3 ~/research/AutomatAnts/code/run_model.py -r HR,s -x serial_recruitment
deactivate

#how to run
#qsub ~/research/AutomatAnts/code/run_model.sh 
#qsub -pe make 20 -l h_vmem=8G ~/research/AutomatAnts/code/run_model.sh