# FlexibleWorkingMemory
Simplified code for the paper "A Flexible Model of Working Memory" Bouchacourt and Buschman, 2019.
This code is written in Python and is using the Brian2 spiking network simulator (https://brian2.readthedocs.io/en/stable/) so you will need brian2 and brian2tools to use it.

run_a_trial.py is calling FlexibleWM.py. As an input to FlexibleWM we give a dictionnary describing the parameters taking different value from default (e.g. if you want to change the simulation or stimulation time). 

To run multiple simulations over multiple networks on a cluster, you can use the python multiprocessing package. If you plan to use it we can send you the script. In general, it is better to use multiple arrays with slurm instead.



