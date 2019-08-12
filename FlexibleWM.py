# Bouchacourt and Buschman 2019
# A Flexible Model of Working Memory
# This is a simplified version of the code. Specific analyses of the paper can be pushed on demand. 


import pickle, numpy, scipy, pylab, os
import pdb
import scipy.stats
import math 
import sys
import os.path
import logging 
from brian2 import *
BrianLogger.log_level_error()
from brian2tools import *
import cython
prefs.codegen.target = 'cython' 
import gc as gcPython

class FlexibleWM:
  """Class running one trial of the network. We give it a dictionary including parameters not taking default value, as well as the name of the folder for saving."""
  def __init__(self,spec):
    self.specF={} # empty dictionnary
    self.specF['name_simu'] = spec.get('name_simu','FlexibleWM') # Name of the folder to save results and eventual raster plots
    self.specF['Number_of_trials'] = spec.get('Number_of_trials',1) # increase in order to run multiple trials with the same network

    # Timing
    self.specF['clock'] = spec.get('clock',0.1) # Integration with time-step in msec
    self.specF['simtime'] = spec.get('simtime',1.1) # Simulation time in sec
    self.specF['RecSynapseTau'] = spec.get('RecSynapseTau',0.010) # Synaptic time constant
    self.specF['window_save_data'] = spec.get('window_save_data',0.1)   # the time step for saving data like rates, it will also be the by default timestep for saving spikes
    if self.specF['window_save_data']!=0.1 or self.specF['simtime']!=1.1 :
      print(" ----------- WARNING : Make sure simtime divided by window_save_data is an integer ----------- ")

    # Network architecture
    self.specF['N_sensory'] = spec.get('N_sensory',512) # Number of recurrent neurons per SN
    self.specF['N_sensory_pools'] = spec.get('N_sensory_pools',8)  # Number of ring-like SN
    self.specF['N_random'] = spec.get('N_random',1024) # Number of neuron in the random network
    self.specF['self_excitation'] = spec.get('self_excitation',False) # if you want to run the SN by itself, the self excitation has to be True
    self.specF['with_random_network'] = spec.get('with_random_network',True) # with the RN, self excitation is False, so that a SN by itself cannot maintain a memory
    self.specF['fI_slope'] = spec.get('fI_slope',0.4)  # slope of the non-linear f-I function

    # Parameters describing recurrence within the SN
    self.specF['RecWNegativeWidth']= spec.get('RecWNegativeWidth',0.25);   # width of negative surround suppression
    self.specF['RecWPositiveWidth'] = spec.get('RecWPositiveWidth',1);    # width of positive amplification
    self.specF['RecWAmp_exc'] = spec.get('RecWAmp_exc',2) # amplitude of weight function
    self.specF['RecWAmp_inh'] = spec.get('RecWAmp_inh',2) # amplitude of weight function
    self.specF['RecWBaseline'] = spec.get('RecWBaseline',0.28)    # baseline of weight matrix for recurrent network

    # Parameters describing the weights between SN and RN
    self.specF['eibalance_ff'] = spec.get('eibalance_ff',-1.) # if -1, perfect feed-forward balance from SN to RN
    self.specF['eibalance_fb'] = spec.get('eibalance_fb',-1.) # if -1, perfect feedback balance from RN to SN
    self.specF['RecToRndW_TargetFR'] = spec.get('RecToRndW_TargetFR',2.1) # parameter (alpha in the paper) used to compute the feedforward weight, before balancing
    self.specF['RndToRecW_TargetFR'] = spec.get('RndToRecW_TargetFR',0.2) # parameter (beta in the paper) used to compute the feedback weight, before balancing
    self.specF['RndRec_f'] = spec.get('RndRec_f',0.35) # connectivity (gamma in the paper)
    self.specF['factor'] = spec.get('factor',1000) # factor for computing weights values (see Methods of the paper)

    # Saving/Using or not a pre-saved network
    self.specF['same_network_to_use'] = spec.get('same_network_to_use',False) # if we want to initialise the network with weights previously saved
    self.specF['create_a_specific_network'] = spec.get('create_a_specific_network',False) # if we want save weights in order to run later the model with the same network
    self.specF['path_for_same_network'] = spec.get('path_for_same_network',self.specF['name_simu']+'/network') # path for the weights

    # Stimulation
    self.specF['specific_load'] = spec.get('specific_load',False) # whether to use a specific load for all trials, or having it random
    self.specF['value_of_specific_load'] = spec.get('value_of_specific_load',1) # value of the load if specific_load is True
    self.specF['start_stimulation'] = spec.get('start_stimulation',0.1)
    self.specF['end_stimulation'] = spec.get('end_stimulation',0.2)
    self.specF['input_strength'] = spec.get('input_strength',10) # strength of the stimulation
    self.specF['N_sensory_inputwidth'] = spec.get('N_sensory_inputwidth',32)
    self.specF['InputWidthFactor'] = spec.get('InputWidthFactor',3)
    self.specF['InputWidth'] = round(self.specF['N_sensory']/float(self.specF['N_sensory_inputwidth']))  # the width for input stimulation of the gaussian distribution

    # ML decoding
    self.specF['decode_spikes_timestep'] = spec.get('decode_spikes_timestep',0.1) # decoding window
    self.specF['path_load_matrix_tuning_sensory'] = spec.get('path_load_matrix_tuning_sensory',self.specF['name_simu']+'/Matrix_tuning.npz') # path for tuning curve matrix
    self.specF['compute_tuning_curve'] = spec.get('compute_tuning_curve',True)

    # Define path_to_save and eventual raster plot
    self.specF['plot_raster'] = spec.get('plot_raster',True)   # plot a raster
    self.specF['path_to_save'] = self.define_path_to_save_results_from_the_trial()
    self.specF['path_sim'] = self.specF['path_to_save']+'simulation_results.npz'



  def define_path_to_save_results_from_the_trial(self) :
    path_to_save = self.specF['name_simu']+'/'
    if not os.path.exists(path_to_save):
      try : 
        os.makedirs(path_to_save)
      except OSError:
        pass
    return path_to_save

  def apply_matFuncMask(self, m, target, mask, axis_ziou): 
    for i in range(m.shape[0]) :
      for j in range(m.shape[1]) :
        if mask[i,j]:
          if axis_ziou :
            m[i,j]= float(self.specF['factor'])*target/numpy.sum(mask[:,j])
          else :
            m[i,j]= float(self.specF['factor'])*target/numpy.sum(mask[i,:])
    return m


  def compute_activity_vector(self,R_rn_timed) :
    neuron_angs = numpy.arange(1,self.specF['N_sensory']+1,1)/float(self.specF['N_sensory'])*2*math.pi
    exp_neuron_angs = numpy.exp(1j*neuron_angs,dtype=complex)
    Matrix_abs_timed = numpy.zeros(self.specF['N_sensory_pools'])
    Matrix_angle_timed = numpy.zeros(self.specF['N_sensory_pools'])
    R_rn_2 = numpy.ones(R_rn_timed.shape,dtype=complex)
    for index_pool in range(self.specF['N_sensory_pools']) :
      R_rn_2[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']] = numpy.multiply(R_rn_timed[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']], exp_neuron_angs, dtype=complex)
      R_rn_3 = numpy.mean(R_rn_2[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']])
      Matrix_angle_timed[index_pool] = numpy.angle(R_rn_3)*self.specF['N_sensory']/(2*math.pi)
      Matrix_abs_timed[index_pool] = numpy.absolute(R_rn_3)
    Matrix_angle_timed[Matrix_angle_timed<0]+=self.specF['N_sensory']
    return Matrix_abs_timed, Matrix_angle_timed


  def compute_drift(self,Angle_output,Angle_input):
      Angle_output_rad = Angle_output*2*math.pi/float(self.specF['N_sensory'])
      Angle_input_rad = Angle_input*2*math.pi/float(self.specF['N_sensory'])
      difference_angle = ((Angle_output_rad-Angle_input_rad+math.pi)%(2*math.pi))-math.pi
      difference_complex = numpy.exp(1j*difference_angle,dtype=complex)
      difference_angle2 = numpy.angle(difference_complex)
      return difference_angle2

  def ml_decode(self,number_of_spikes_rn,Matrix_tc) :
    Matrix_likelihood_per_stim = numpy.zeros(self.specF['N_sensory'])
    for index_stim in range(self.specF['N_sensory']) :
      Matrix_likelihood_per_stim[index_stim] = numpy.dot(number_of_spikes_rn,numpy.log(Matrix_tc[:,index_stim]))
    S_ml = numpy.argmax(Matrix_likelihood_per_stim)
    if isinstance(S_ml, numpy.ndarray) :
      pdb.set_trace()
    return S_ml 


  def give_input_stimulus(self,current_InputCenter) :
    inp_vect = numpy.zeros(self.specF['N_sensory'])
    inp_ind = numpy.arange(int(round(current_InputCenter-self.specF['InputWidthFactor']*self.specF['InputWidth'])),int(round(current_InputCenter+self.specF['InputWidthFactor']*self.specF['InputWidth']+1)),1)
    inp_scale = scipy.stats.norm.pdf(inp_ind-current_InputCenter,0,self.specF['InputWidth'])
    inp_scale/=float(numpy.amax(inp_scale))
    inp_ind = numpy.remainder(inp_ind-1, self.specF['N_sensory']) 
    inp_vect[inp_ind] = self.specF['input_strength']*inp_scale
    inp_vect[:] = inp_vect[:]-numpy.sum(inp_vect[:])/float(inp_vect[:].shape[0])
    return inp_vect

  def compute_matrix_tuning_sensory(self) :
    Matrix_tuning = numpy.zeros((self.specF['N_sensory'], self.specF['N_sensory'])) # number of neurons in each pool, number of possible stimuli
    for index_stimulus in range(self.specF['N_sensory']) :
      Vector_stim = self.give_input_stimulus(index_stimulus)
      for index_sensoryneuron in range(self.specF['N_sensory']) :
        S_ext = Vector_stim[index_sensoryneuron]
        Matrix_tuning[index_sensoryneuron,index_stimulus] = 0.4*(1+math.tanh(self.specF['fI_slope']*S_ext-3))/self.specF['RecSynapseTau']
    numpy.savez_compressed(self.specF['path_load_matrix_tuning_sensory'],Matrix_tuning=Matrix_tuning)
    return Matrix_tuning

  def compute_psth_for_mldecoding(self,time_matrix,spike_matrix,End_of_delay,timestep) :
    time_length = int(round(End_of_delay/timestep))
    Matrix = numpy.zeros((self.specF['N_sensory_pools']*self.specF['N_sensory'],time_length))
    for index_tab in range(time_matrix.shape[0]) :
      if time_matrix[index_tab]<End_of_delay :
        Time_integer = int(floor(time_matrix[index_tab]*1/timestep))
        Matrix[spike_matrix[index_tab],Time_integer]+=1
    return Matrix

  def initialize_weights(self) :
    print("Initializing the weights of the network that will be used for all trials \n ")
    numpy.random.seed()
    if self.specF['same_network_to_use'] :
      weight_data = numpy.load(self.specF['path_for_same_network'])
      Intermed_matrix_rn_rn2 = weight_data['Intermed_matrix_rn_rn2']
      Intermed_matrix_rn_to_rcn2 = weight_data['Intermed_matrix_rn_to_rcn2']
      Intermed_matrix_rcn_to_rn2 = weight_data['Intermed_matrix_rcn_to_rn2']
      weight_data.close()
      return Intermed_matrix_rn_rn2, Intermed_matrix_rn_to_rcn2, Intermed_matrix_rcn_to_rn2
    else : 
      RecToRndW_EIBalance = self.specF['eibalance_ff'];
      RecToRndW_Baseline = 0; #baseline weight from recurrent network to random network, regardless of connection existing
      RndToRecW_EIBalance = self.specF['eibalance_fb'];
      RndToRecW_Baseline = 0;  # baseline weight from random network to recurrent network, regardless of connection existing
      RndWBaseline = 0;  # Baseline inhibition between neurons in random network
      RndWSelf = 0; # Self-excitation in random network
      PoolWBaseline = 0; # baseline weight between recurrent pools (scaled for # of neurons)
      PoolWRandom = 0; # +/- range of random weights between recurrent pools (scaled for # of neurons)
      # Connection matrix for RN to RN
      Angle = 2.*math.pi*numpy.arange(1,self.specF['N_sensory']+1)/float(self.specF['N_sensory'])
      def weight_intrapool(i) :
        return self.specF['RecWBaseline'] + self.specF['RecWAmp_exc']*exp(self.specF['RecWPositiveWidth']*(cos(i)-1)) - self.specF['RecWAmp_inh']*exp(self.specF['RecWNegativeWidth']*(cos(i)-1)) 

      Matrix_weight_intrapool = numpy.zeros((self.specF['N_sensory'],self.specF['N_sensory']))
      for index1 in range(self.specF['N_sensory']) :
        for index2 in range(self.specF['N_sensory']) :
          if index1 == index2 and self.specF['self_excitation']==False :
            Matrix_weight_intrapool[index1,index2] = 0
          else :
            Matrix_weight_intrapool[index1,index2] = weight_intrapool(Angle[index1]-Angle[index2])

      Intermed_matrix_rn_rn = (PoolWBaseline + PoolWRandom*2*(numpy.random.rand(self.specF['N_sensory_pools']*self.specF['N_sensory'], self.specF['N_sensory_pools']*self.specF['N_sensory']) - 0.5))/(self.specF['N_sensory']*(self.specF['N_sensory_pools']-1))
      for index_pool in range(self.specF['N_sensory_pools']) :
        Intermed_matrix_rn_rn[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory'],index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']]=Matrix_weight_intrapool[:,:]

      Intermed_matrix_rn_rn2 = Intermed_matrix_rn_rn.flatten()  #http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html#weight-matrices

      if self.specF['with_random_network'] :
        Matrix_Sym = numpy.random.rand(self.specF['N_sensory']*self.specF['N_sensory_pools'],self.specF['N_random'])
        Matrix_Sym = Matrix_Sym < self.specF['RndRec_f']

        Matrix_co_rn_to_rcn = RecToRndW_Baseline*numpy.ones((self.specF['N_sensory_pools']*self.specF['N_sensory'],self.specF['N_random']))
        Intermed_matrix_rn_to_rcn = self.apply_matFuncMask(Matrix_co_rn_to_rcn, self.specF['RecToRndW_TargetFR'] , Matrix_Sym,True)
        

        Matrix_co_rcn_to_rn = RndToRecW_Baseline*numpy.ones((self.specF['N_random'],self.specF['N_sensory_pools']*self.specF['N_sensory']))
        Intermed_matrix_rcn_to_rn = self.apply_matFuncMask(Matrix_co_rcn_to_rn.T, self.specF['RndToRecW_TargetFR'] , Matrix_Sym, False).T

        # Balance and then flatten
        for index_control_neuron in range(self.specF['N_random']) :
          Sum_of_excitation = numpy.sum(Intermed_matrix_rn_to_rcn[:,index_control_neuron])
          Intermed_matrix_rn_to_rcn[:,index_control_neuron]+=self.specF['eibalance_ff']*Sum_of_excitation/float(self.specF['N_sensory']*self.specF['N_sensory_pools'])
        Intermed_matrix_rn_to_rcn2 = Intermed_matrix_rn_to_rcn.flatten()   

        for index_neuron in range(self.specF['N_sensory']*self.specF['N_sensory_pools']) :
          Sum_of_excitation = numpy.sum(Intermed_matrix_rcn_to_rn[:,index_neuron])   # somme sur les 1024 control neurons
          Intermed_matrix_rcn_to_rn[:,index_neuron]+=self.specF['eibalance_fb']*Sum_of_excitation/float(self.specF['N_random'])
        Intermed_matrix_rcn_to_rn2 = Intermed_matrix_rcn_to_rn.flatten()

      else :
        Intermed_matrix_rn_to_rcn = numpy.zeros((self.specF['N_sensory']*self.specF['N_sensory_pools'],self.specF['N_random']))
        Intermed_matrix_rn_to_rcn2 = Intermed_matrix_rn_to_rcn.flatten()  
        Intermed_matrix_rcn_to_rn = numpy.zeros((self.specF['N_random'],self.specF['N_sensory_pools']*self.specF['N_sensory'])) 
        Intermed_matrix_rcn_to_rn2 = Intermed_matrix_rcn_to_rn.flatten()

      # creating a specific network, saving, and plotting the weights
      if self.specF['create_a_specific_network']  :
        numpy.savez_compressed(self.specF['path_for_same_network'], Intermed_matrix_rn_rn2=Intermed_matrix_rn_rn2, Intermed_matrix_rn_to_rcn2=Intermed_matrix_rn_to_rcn2, Intermed_matrix_rcn_to_rn2=Intermed_matrix_rcn_to_rn2)
        print("Network is saved in the folder, next time you can include 'same_network_to_use':True to reuse it")
      else :
        return Intermed_matrix_rn_rn2, Intermed_matrix_rn_to_rcn2, Intermed_matrix_rcn_to_rn2


  def run_a_trial(self) : 
    numpy.random.seed()
    gcPython.enable()
    # --------------------------------- Network parameters ---------------------------------------------------------
    # Setting the simulation timestep
    defaultclock.dt = self.specF['clock']*ms     # this will be used by all objects that do not explicitly specify a clock or dt value during construction
    # Setting the simulation time
    Simtime = self.specF['simtime']*second
    activity_threshold = 3
    fI_slope = self.specF['fI_slope']

    # Parameters of the neurons
    bias = 0  # bias in the firing response (cf page 1 right column of Burak, Fiete 2012)

    # Parameters of the synapses
    RecSynapseTau = self.specF['RecSynapseTau']*second
    RndSynapseTau = self.specF['RecSynapseTau']*second
    InitSynapseRange = 0.01    # Range to randomly initialize synaptic variables

    # --------------------------------- Network setting and equations ---------------------------------------------------------
    Intermed_matrix_rn_rn2, Intermed_matrix_rn_to_rcn2, Intermed_matrix_rcn_to_rn2 = self.initialize_weights()

    # Equations
    eqs_rec = '''
    dS_rec/dt = -S_rec/RecSynapseTau      :1
    S_ext : 1
    G_rec = S_rec + bias + S_ext  :1
    rate_rec = 0.4*(1+tanh(fI_slope*G_rec-3))/RecSynapseTau     :Hz
    '''

    eqs_rcn = '''
    dS_rnd/dt = -S_rnd/RndSynapseTau      :1
    S_ext_rnd : 1
    G_rnd = S_rnd + bias + S_ext_rnd  :1 
    rate_rnd = 0.4*(1+tanh(fI_slope*G_rnd-3))/RndSynapseTau  :Hz
    '''


    # Creation of the network
    Recurrent_Pools = NeuronGroup(self.specF['N_sensory']*self.specF['N_sensory_pools'], eqs_rec, threshold='rand()<rate_rec*dt')
    Recurrent_Pools.S_rec = 'InitSynapseRange*rand()'   # What this does is initialise each neuron with a different uniform random value between 0 and InitSynapseRange

    RCN_pool = NeuronGroup(self.specF['N_random'], eqs_rcn, threshold='rand()<rate_rnd*dt')
    RCN_pool.S_rnd = 'InitSynapseRange*rand()'

    # Building the recurrent connections within pools
    Rec_RN_RN = Synapses(Recurrent_Pools, Recurrent_Pools, model='w : 1', on_pre='S_rec+=w')  # Defining the synaptic model, w is the synapse-specific weight
    Rec_RN_RN.connect()  # connect all to all

    Rec_RN_RN.w = Intermed_matrix_rn_rn2

    if self.specF['with_random_network'] :
      # Building the symmetric recurrent connections from RN to RCN and from RCN to RN 
      Rec_RCN_RN = Synapses(RCN_pool, Recurrent_Pools, model='w : 1', on_pre='S_rec+=w')
      Rec_RCN_RN.connect()  # connect all to all

      Rec_RCN_RN.w = Intermed_matrix_rcn_to_rn2

      Rec_RN_RCN = Synapses(Recurrent_Pools, RCN_pool, model='w : 1', on_pre='S_rnd+=w')
      Rec_RN_RCN.connect()   # connect all to all

      Rec_RN_RCN.w = Intermed_matrix_rn_to_rcn2

    R_rn = StateMonitor(Recurrent_Pools, 'rate_rec', record=True, dt=self.specF['window_save_data']*second)

    if self.specF['plot_raster'] :
      S_rn = SpikeMonitor(Recurrent_Pools)
      if self.specF['with_random_network'] :
        S_rcn = SpikeMonitor(RCN_pool)

    
    # STORE THE NETWORK, in case we run a large number of simulations with the same network
    store('initialized')

    # We build the baseline, random input (set to 0 for now)
    InputBaseline = 0; # strength of random inputs
    inp_baseline = numpy.zeros((self.specF['N_sensory_pools'],self.specF['N_sensory']))
    for index_pool in range(self.specF['N_sensory_pools']) :
      inp_baseline[index_pool,:] = InputBaseline*numpy.random.rand(self.specF['N_sensory'])

    inp_baseline_rnd = numpy.zeros(self.specF['N_random'])

    Matrix_all_results = numpy.ones((self.specF['Number_of_trials'],2))*numpy.nan
    Matrix_abs_all = numpy.ones((self.specF['Number_of_trials'],self.specF['N_sensory_pools']))*numpy.nan
    Matrix_angle_all = numpy.ones((self.specF['Number_of_trials'],self.specF['N_sensory_pools']))*numpy.nan
    Results_ml_spikes = numpy.ones((self.specF['Number_of_trials'],self.specF['N_sensory_pools']))*numpy.nan
    Drift_from_ml_spikes = numpy.ones((self.specF['Number_of_trials'],self.specF['N_sensory_pools']))*numpy.nan
    Matrix_initial_input = numpy.ones((self.specF['Number_of_trials'],self.specF['N_sensory_pools']))*numpy.nan

    for index_simulation in range(self.specF['Number_of_trials']) :
      print("---------- Initialisation of trial "+str(index_simulation)+' ----------')
      restore('initialized') # restore the initial network, in case trial number is > 1
    
      numpy.random.seed()

      # --------------------------------- Inputs ---------------------------------------------------------
      #Input vector into sensory network
      InputCenter = numpy.floor(numpy.random.rand(self.specF['N_sensory_pools'])*self.specF['N_sensory'])   

      # For now we implement the input at the same time for all pools
      IT1 = self.specF['start_stimulation']*second
      IT2 = self.specF['end_stimulation']*second
      
      Matrix_pools_receiving_inputs = []
      if self.specF['specific_load'] :
        load = self.specF['value_of_specific_load']
      else :
        load = numpy.random.randint(low=1,high=self.specF['N_sensory_pools']+1)
      Matrix_pools = numpy.arange(self.specF['N_sensory_pools'])
      numpy.random.shuffle(Matrix_pools)
      Matrix_pools_receiving_inputs = Matrix_pools[:load]
      print("Load for this trial is "+str(load))
    
      # We build inp_vect, a matrix which gives the stimulus input to each SN
      inp_vect = numpy.zeros((self.specF['N_sensory_pools'],self.specF['N_sensory']))
      for index_pool in Matrix_pools_receiving_inputs :
        inp_vect[index_pool,:] = self.give_input_stimulus(InputCenter[index_pool])  
        Matrix_initial_input[index_simulation,index_pool] = InputCenter[index_pool] 


      # --------------------------------- Running and recording ---------------------------------------------------------


      # Running
      print("Running...")
      for index_pool in range(self.specF['N_sensory_pools']) :
        Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool]
      if self.specF['with_random_network'] :
        RCN_pool.S_ext_rnd = inp_baseline_rnd
      run(IT1) 
      
      for index_pool in range(self.specF['N_sensory_pools']) :
        if index_pool in Matrix_pools_receiving_inputs :
          Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool] + inp_vect[index_pool]
        else :
          Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool]
      if self.specF['with_random_network'] :
        RCN_pool.S_ext_rnd = inp_baseline_rnd
      run(IT2-IT1)

      for index_pool in range(self.specF['N_sensory_pools']) :
        Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool]
      if self.specF['with_random_network'] :
        RCN_pool.S_ext_rnd = inp_baseline_rnd
      run(Simtime-IT2)


      if self.specF['plot_raster'] :
        print("Plot of a raster into the folder ..")
        if self.specF['with_random_network'] :
          figure()
          subplot(211)
          plot_raster(S_rcn.i, S_rcn.t, time_unit=second, marker=',', color='k')
          ylabel('Random neurons')
          subplot(212)
          plot_raster(S_rn.i, S_rn.t, time_unit=second, marker=',', color='k')
          for index_sn in range(self.specF['N_sensory_pools']+1) :
            plot(numpy.arange(0,self.specF['simtime']+self.specF['window_save_data']/2.,self.specF['window_save_data']),index_sn*self.specF['N_sensory']*numpy.ones(int(round(self.specF['simtime']/self.specF['window_save_data']))+1),color='b', linewidth=0.5)
          ylabel('Sensory neurons')
          savefig(self.specF['path_to_save']+'rasterplot_trial'+str(index_simulation)+'.png')
          close()
        else :
          figure()
          plot_raster(S_rn.i, S_rn.t, time_unit=second, marker=',', color='k')
          for index_sn in range(self.specF['N_sensory_pools']+1) :
            plot(numpy.arange(0,self.specF['simtime']+self.specF['window_save_data']/2.,self.specF['window_save_data']),index_sn*self.specF['N_sensory']*numpy.ones(int(round(self.specF['simtime']/self.specF['window_save_data']))+1),color='b', linewidth=0.5)
          ylabel('Sensory neurons')
          savefig(self.specF['path_to_save']+'rasterplot_trial'+str(index_simulation)+'.png')
          close()

      
      # RESULTS
      End_of_delay = self.specF['simtime']-self.specF['window_save_data']
      Time_chosen = int(round(End_of_delay/self.specF['window_save_data']))       
      R_rn_enddelay = numpy.transpose(R_rn.rate_rec[:,Time_chosen].copy())  # beware numpy.transpose is reference
      Matrix_abs, Matrix_angle = self.compute_activity_vector(R_rn_enddelay)

      Matrix_abs_all[index_simulation,:] = Matrix_abs
      Matrix_angle_all[index_simulation,:] = Matrix_angle
      if self.specF['compute_tuning_curve'] :
        Matrix_tuning = self.compute_matrix_tuning_sensory()
        print("Tuning curve is saved in the folder, next time you can include 'compute_tuning_curve':False to reuse it")


      print("\n---------- ML decoding at the end of trial "+str(index_simulation)+' ----------')
      tuning_data = numpy.load(self.specF['path_load_matrix_tuning_sensory']) 
      Matrix_tuning = tuning_data['Matrix_tuning']
      tuning_data.close()

      # S_rn.t is in second, we need to get rid of the unit time in order to compare it to timestep in the function compute_psth
      time_matrix = numpy.zeros(S_rn.t.shape[0])
      for index in range(S_rn.t.shape[0]) :
        time_matrix[index] = S_rn.t[index]

      psth_rn = self.compute_psth_for_mldecoding(time_matrix,S_rn.i,End_of_delay,self.specF['decode_spikes_timestep'])[:,int(round(End_of_delay/self.specF['decode_spikes_timestep']))-1]
      for index_pool in range(self.specF['N_sensory_pools']) :
        Results_ml_spikes[index_simulation,index_pool] = self.ml_decode(psth_rn[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']],Matrix_tuning)
        if numpy.isnan(InputCenter[index_pool])==False : # or index_pool in Matrix_pools_receiving_inputs
          Drift_from_ml_spikes[index_simulation,index_pool] = self.compute_drift(Results_ml_spikes[index_simulation,index_pool],InputCenter[index_pool])

      print('Initial inputs were (nan means no initial input into this SN)')
      print(Matrix_initial_input[index_simulation,:])
      print("Memory is found ")
      print(Matrix_abs>activity_threshold)
      print("ML decoding gives")
      print(Results_ml_spikes[index_simulation,:])
      print("Decoding from the activity vector gives")
      print(Matrix_angle)
      print('\n')
            
      print("---------- Computing capacity, maintained and spurious memories ----------")

      proba_maintained = 0
      proba_spurious = 0
      for index_pool_capacity in range(self.specF['N_sensory_pools']) :
        if Matrix_abs[index_pool_capacity]>activity_threshold :
          if index_pool_capacity in Matrix_pools_receiving_inputs :
            proba_maintained+=1
          else :
            proba_spurious+=1
      print(str(proba_maintained)+' maintained memories')
      print(str(Matrix_pools_receiving_inputs.shape[0]-proba_maintained)+' forgotten memories')
      print(str(proba_spurious)+' spurious memories\n')
      proba_maintained/=float(load)
      if load!=self.specF['N_sensory_pools'] :
        proba_spurious/=float(self.specF['N_sensory_pools']-load)

      Matrix_all_results[index_simulation,:] = numpy.asarray([proba_maintained,proba_spurious])

    # saving  
    numpy.savez_compressed(self.specF['path_sim'], Matrix_all_results=Matrix_all_results,Matrix_abs_all=Matrix_abs_all, Matrix_angle_all=Matrix_angle_all, Results_ml_spikes=Results_ml_spikes, Drift_from_ml_spikes=Drift_from_ml_spikes,Matrix_initial_input = Matrix_initial_input)
    print("All results saved in the folder")
    gcPython.collect()
    
    return 





