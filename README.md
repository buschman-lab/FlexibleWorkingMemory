# FlexibleWorkingMemory
Simplified code for the paper "A Flexible Model of Working Memory" Bouchacourt and Buschman, 2019.
This code is written in Python and is using the Brian2 spiking network simulator (https://brian2.readthedocs.io/en/stable/) so you will need brian2 and brian2tools to use it.

run_a_trial.py is calling FlexibleWM.py.

In run_a_trial.py, you can add in the dictionnary the parameters you want to change from default. 
To see these parameters name and their default value, go to FlexibleWM.py, at the beginning of the script. 

Example : In the run_a_trial.py, you change dictionnary={} to : dictionnary={'value_of_specific_load':6, 'specific_load':True} to reproduce Figure 1 because otherwise a random initial load is chosen at each simulation, as described.

It should create a folder with 2 compressed files (.npz) describing the simulation results (simulation_results.npz, description line 471 of FlexibleWM.py) and the tuning curve of neurons (Matrix_tuning.npz). It should also create a .png image called rasterplot_trial0.png where you can see a similar raster plot as the one of Figure 1 of the paper. Obviously, the number of forgotten memories varies from trial to trial. 

To run multiple simulations over multiple networks on a cluster, you can use the python multiprocessing package. If you plan to use it we can push the script. In general, it is better to use multiple arrays with slurm instead.
If you are interested in seeing the script of specific paper analyses, or of the 2D surface or Hopfield-like sensory networks, please email us and we will push it.



