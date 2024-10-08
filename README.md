# Deep Reinforcement Learning (DDPG) for Home Energy Management

## Abstract
Efficient energy management on the level of prosumer households is key to alleviating grid stress in an energy transition marked by electrification, a switch to renewable generation, and decentralization. With millions of new private EV charging stations estimated to be installed by 2030, smart EV charging is one strategy to balance production and consumption locally, as well as regionally. Model-free deep reinforcement learning has emerged as a promising method to deal with the dynamic and uncertain environment of home energy management optimization problems. In this thesis, a deep deterministic policy gradient algorithm is applied to a new real-life dataset of prosumer households with PV, battery, and EV chargers and benchmarked against a rule-based power charging mode commonly used in the markets today, as well as the optimal solution under perfect information with model predictive control.

## Model Set-Up
The code for the Deep Reinforcement Learning model (RL-SHEMS), also including a rule-based power-charging (SHEMS python) as lower benchmark, was adapted from Langer (2022) in the programming language Julia. The original code is available under https://github.com/lilanger/RL-SHEMS.git.
Python was used for the descriptive analysis and most of the pre-processing of the data.
The code for the Model Predictive Control Benchmark is written in python and the approach is based on https://github.com/lilanger/SHEMS.git. 

## Data Availability
Please note that the company data used in this thesis is not publically available.

## Requirements
- Julia is used in version 1.6.1.
- The repository contains a Manifest and Project files, so that the same Julia package versions can be installed.
- Gurobi solver (package version 0.7.6) for the MPC benchmark.
- Nvidida GPU (for CUDA usage)

## Repository Structure
Having been based on the repository by Langer (2022), this repositroy contains remnants of the original model that was built for the HVAC context that do not work in their current from. These remained, mostly in achrive folders, or quoted-out directly in the code, for the purpose of recycling in future applciations and adaptations. Among these are also different alogrithms, HVAC-datasets, the function to run the model in distinct seasons, the option to run the code on a SGE cluster and different RL environments.

SHEMS-python includes the python implementation of the MPC benchmark.

RL-SHEMS includes the DRL model.

The folder input_templates includes the input files used for eval and test runs with paramters settings for rule-based (RB), default DRL (DF), the parameter-search (PS), and the grid search (GS) as described in the thesis. Here the last two digids of the JOB_ID are used to determined parameter settings (as binary number).

The folder algorithms contains the code for the DDPG implementation.

RL Environment:
The folder Reinforce.jl... contains the RL environments and the file to embed them in the Reinforce package.
The environment used in the thesis was shems_LU1.jl.
Changes to the discomfort score, penalty weight, as well as EV capacities have to be entered here.

Visualization contains figures of the descriptive analysis and of the results, produced with jupiter notebooks.

The out folder contains the results of the model runs.
out/trackercharger.csv records each runs settings and results.
out/tracker contains the resul files with detailed time series.
out/fig contains plots of the results over the episodes.
out/input contains copies of the input files used in each run.
out/bson contains the .bson (binary json) files with the trained agent's network weights.

The bash_schedulers folder contains a number of bash schedulers that were used for all the thesis runs, where the JOB_IDs 1070xxxx/1071xxxx are for the rule-based benchmark, 72/73 for default DRL, 74/75 for parameter search, 76/77 grid search, 78/79 the tuned model and 80 for rule-based on training data.
The moste recent bash scheduler scripts (starting on 11...) are contained in the main folder.

The jupiter notebooks Data_preparation, Data_pre-processing and Data_descriptive_analysis were used to pre-process, filter and analysie the data (where the julia code in Data_preparation is adapted from the original code to bring the data into the right format for the DRL model).

## Workflow

Work flow RL-SHEMS:

inputs:
In general, all input parameters and settings are fed from the input.jl file (see input_parameter_tuning_template for scheduling and tuning). This is also where the JOB_ID is set manually or inferred from the environement in case of scheduler or clust jobs.

The third and fourth last digids in the JOB_ID determine the data set (11110811 will use ChargerID08).

track = 1 is the DRL run. track = -0.X sets the model to rule-based with X being the percentage of starting SOC.
train = 0 or = 1 determines wether training should be conducted on this run.
run = "eval" or = "test" determines which dataset to evaluate/test on.

When running training for multiple seeds, a WAIT_SEC time of at least 5min (depending on parameters) to ensure training is finished on all seeds before evaluation.
The input files of previous runs are saved in out/input.

The general workflow is defined in DDPG_reinforce_charger_v1.jl.

## Final Results
JOBIDs 11790108, 11790208... to 11799808 contained the training and evaluation of the final results of the tuned DRL agent reportet in the thesis, for datasets ChargerID 01 to 09 and 98 (synthetic).
The parameter settings were:

DISCOMFORT_WEIGHT_EV = 2, 
TRAIN_EP_LENGTH = 72, 
NUM_EP = 1001, 
BATCH_SIZE = 120, 
MEM_SIZE = 24_000, 
noise_type = "gn", 
L1 = 250, 
L2 = 500, 
penalty = 0.1, 
γ = 0.99, 
τ = 1f-3, 	
η_act = 1f-4,    	
η_crit = 1f-3,   	
σ = 0.1, 
θ = 0.15, 

The .bson actor and critic weights of the best runs have the file names:
"DDPG_Shems_Charger_v1_72_1001_300_600_ChargerXX_dw0.01_p0.1_BT120_MEM24000_gn_nsact0.1_nstrg0.2_om0.1_th0.15_Y0.99_tau0.001_nact0.0001_ncrit0.001_123YY_actor_1001", 
where XX is to be replaced with the ChargerID, critic replaced with actor and YY with the respective best eval run seed.
Best eval runs were (ChargerID: seed):
01: 1236, 
02: 1231, 
03: 1235, 
04: 12328, 
05: 1233, 
06: 12339, 
07: 12310, 
08: 1233, 
09: 1238, 
98: 12320, 


## References
Lissy Langer and Thomas Volling. A reinforcement learning approach to home energy management for modulating
heat pumps and photovoltaic systems. Applied Energy, 327:120020, 2022
