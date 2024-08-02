algo="DDPG"
# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl
using Pkg
Pkg.activate(pwd())
using Flux, Printf, Zygote
using Flux.Optimise: update!
using BSON
using Statistics: mean, std, median
using DataStructures: CircularBuffer
using Distributions: sample, Normal, logpdf
using Random
using Reinforce
using Dates
using Plots
using CSV, DataFrames
gr()



#------------ local machine ----------
#Job_ID = 1006241 #1149869
#Task_ID = 1 #1149869-1 #ENV["SGE_TASK_ID"]
#seed_run = 1 # run inference over all seeds 
#num_seeds = 1 #40

#--------cluster jobs------------
#Job_ID = ENV["JOB_ID"]
#Task_ID = ENV["SGE_TASK_ID"] #LU: original for SGE cluster approach
#seed_run = parse(Int, Task_ID)

#--------bash scheduler jobs------------
Job_ID = ENV["JOB_ID"]
Task_ID = ENV["TASK_ID"]
seed_run = parse(Int, Task_ID)

Charger_ID = "Charger" * lpad((parse(Int, Job_ID) ÷ 100) % 100, 2, '0')

#-------------------------------- INPUTS --------------------------------------------

#------------------- Setting -------------------
train = 1 # 0 1 
plot_result = 0 #0 1
plot_all = 1 #0 1
render = 0 #0 1
track = 1 #-0.7  # 0 - off, 1 - DRL, , rule-based percentage of start Soc e.g. 70% -> -0.7 (has to be negative)
run = "eval" # "test", "eval"
WAIT_SEC = 300 # 60 * (1 + (parse(Int, Job_ID) % 10) * (parse(Int, Job_ID) % 10))
num_seeds = 40  # always make sure this matches the highest Task_ID in the bash scheduler!


# Function to parse the last two digits of the JOB_ID and set hyperparameters

#------------------- Grid search function -------------------

function set_hyperparameters(job_id)
  # Extract the last two digits of the JOB_ID
  last_two_digits = parse(Int, job_id[end-1:end])

  # Define the default values for the hyperparameters
  params = Dict(
      :DISCOMFORT_WEIGHT_EV => 0.01, # REMEMBER TO ADJUST THIS IN ENV
      :penalty => 0.1, # REMEMBER TO ADJUST THIS IN ENV
      :TRAIN_EP_LENGTH => 72,
      :NUM_EP => 1_001,
      :BATCH_SIZE => 120,
      :MEM_SIZE => 24_000,
      #------------- Network tuning ----------------
      :L1 => 300,
      :L2 => 600,
      :γ => 0.99f0,     	# discount rate for future rewards 
      :τ => 1f-3, 		# Parameter for soft target network updates Fudji: 5f-3 
      :η_act => 1f-4,   	# Learning rate actor YFudjiu: 1f-3
      :η_crit => 1f-3,  	# Learning rate critic
      #------------- Noise (exploration vs exploitation) -----
      :noise_type => "gn",
      :σ => 0.1f0, #0.2f0 #sigma
      :θ => 0.15f0, #theta
      :noise_act => 0.1f0,
      :noise_trg => 0.2f0,
  )

  # Define the alternative values for the hyperparameters
  alt_values = Dict(
      1 => (0.04, :DISCOMFORT_WEIGHT_EV),# REMEMBER TO ADJUST THIS IN ENV
      2 => ("lin1", :DISCOMFORT_WEIGHT_EV),# REMEMBER TO ADJUST THIS IN ENV
      3 => (100, :BATCH_SIZE),
      4 => (150, :BATCH_SIZE),
      5 => (20_000, :MEM_SIZE),
      6 => (30_000, :MEM_SIZE),
      7 => ((200, 400), :L1_L2),
      8 => ((400, 800), :L1_L2),
      9 => (5f-3, :τ),
      10 => (5f-4, :τ),
      11 => ((5f-4, 5f-3), :η_act_η_crit),
      12 => ((5f-5, 5f-4), :η_act_η_crit),
      13 => (0.2f0, :noise_act), #theta
      14 => ("ou", :noise_type), #theta
      15 => (1, :penalty),
      16 => (0, :penalty)
      )

  # Set the hyperparameter based on the last two digits
  if last_two_digits in keys(alt_values)
      value, param = alt_values[last_two_digits]
      if param == :L1_L2
          params[:L1], params[:L2] = value  # Unpack the tuple for L1 and L2
      elseif param == :η_act_η_crit
          params[:η_act], params[:η_crit] = value
      else
          params[param] = value
      end
  end

  return params
end

# Set the hyperparameters
params = set_hyperparameters(Job_ID)

# Assign the values to variables
DISCOMFORT_WEIGHT_EV = params[:DISCOMFORT_WEIGHT_EV]
TRAIN_EP_LENGTH = params[:TRAIN_EP_LENGTH]
NUM_EP = params[:NUM_EP]
BATCH_SIZE = params[:BATCH_SIZE]
MEM_SIZE = params[:MEM_SIZE]
noise_type = params[:noise_type]
L1 = params[:L1]
L2 = params[:L2]
penalty = params[:penalty]
γ = params[:γ]     	# discount rate for future rewards 
τ = params[:τ] 		# Parameter for soft target network updates Fudji: 5f-3 
η_act = params[:η_act]   	# Learning rate actor YFudjiu: 1f-3
η_crit = params[:η_crit]  	# Learning rate critic
σ = params[:σ]
θ = params[:θ]
noise_act = params[:noise_act]
noise_trg = params[:noise_trg]

#------------------- Other paramters -------------------

# Optimizers
opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)
#L2_DECAY = 0.01f0

season = "all" # "all" "both" "summer" "winter"
price= "fix" # "fix", "TOU"
idx=NUM_EP
test_every = 100
test_runs = 100
seed_ini = 123
rng_run = parse(Int, string(seed_ini)*string(seed_run))
start_time = now()
current_episode = 0
MIN_EXP_SIZE = MEM_SIZE 
memory = CircularBuffer{Any}(MEM_SIZE)

# Define case (for network and result storage identification)
if track < 0
  case = "$(Charger_ID)_rule_based_$(track)"
else
  case = "$(Charger_ID)_dw$(DISCOMFORT_WEIGHT_EV)_p$(penalty)_BT$(BATCH_SIZE)_MEM$(MEM_SIZE)_$(noise_type)_nsact$(noise_act)_nstrg$(noise_trg)_om$(σ)_th$(θ)_Y$(γ)_tau$(τ)_nact$(η_act)_ncrit$(η_crit)"
end

#--------------------------------- Game environment ---------------------------

include("../../RL_environments/envs/shems_LU1.jl")
using .ShemsEnv_LU1: Shems

EP_LENGTH = Dict("train" => TRAIN_EP_LENGTH,
					("summer", "eval") => 359, ("summer", "test") => 767,
					("winter", "eval") => 359, ("winter", "test") => 719,
					("both", "eval") => 719,   ("both", "test") => 1487,
          #("all", "eval") => 100,   ("all", "test") => 200) # length of whole evaluation set (different)
					("all", "eval") => 1439,   ("all", "test") => 2999) # length of whole evaluation set (different)


env_dict = Dict("train" => Shems(EP_LENGTH["train"], "data/$(Charger_ID)_$(season)_train_$(price).csv"),
				"eval" => Shems(EP_LENGTH[season, "eval"], "data/$(Charger_ID)_$(season)_eval_$(price).csv"),
				"test" => Shems(EP_LENGTH[season, "test"], "data/$(Charger_ID)_$(season)_test_$(price).csv"))


WAIT = Dict(
          ("summer", "DDPG") => 150, ("summer", "TD3") => 1500, ("summer", "SAC") => 5000,
          ("winter", "DDPG") => 200, ("winter", "TD3") => 2000, ("winter", "SAC") => 7000,
          ("both", "DDPG") => 300,   ("both", "TD3") => 3000,  ("both", "SAC") => 10000,
          ("all", "DDPG") => WAIT_SEC,    ("all", "TD3") => 5000,   ("all", "SAC") => 20_000) 
          #("summer", "DDPG") => 1500, ("summer", "TD3") => 1500, ("summer", "SAC") => 5000,  #LU: original code
          #("winter", "DDPG") => 2000, ("winter", "TD3") => 2000, ("winter", "SAC") => 7000,
          #("both", "DDPG") => 3000,   ("both", "TD3") => 3000,  ("both", "SAC") => 10000,
          #("all", "DDPG") => 4000,    ("all", "TD3") => 5000,   ("all", "SAC") => 20_000) 



# ----------------------------- Environment Parameters -------------------------
STATE_SIZE = length(env_dict["train"].state)
ACTION_SIZE = length(env_dict["train"].a)
ACTION_BOUND_HI = maximum(env_dict["train"].a)
ACTION_BOUND_LO = minimum(env_dict["train"].a)
# ACTION_BOUND_HI = Float32[1f0, 1f0, 1f0] #Float32(actions(env, env.state).hi[1])
# ACTION_BOUND_LO = Float32[-1f0, -1f0, -1f0] #Float32(actions(env, env.state).lo[1])
# ACTION_BOUND_HI = Float32[4.6f0, 3.0f0] #Float32(actions(env, env.state).hi[1])
# ACTION_BOUND_LO = Float32[-4.6f0, -3.0f0] #Float32(actions(env, env.state).lo[1])

#------------------------------- Action Noise --------------------------------
struct OUNoise
  μ
  σ
  θ
  dt
  X
end

struct GNoise
  μ
  σ_act
  σ_trg
end

mutable struct EpsNoise
  ζ
  ξ
  ξ_min
end

mutable struct ParamNoise
	μ
  σ_current
	σ_target
	adoption
end

# Ornstein-Uhlenbeck / Gaussian Noise params
# based on: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
μ = 0f0 #mu
#σ = 0.1f0 #0.2f0 #sigma
#θ = 0.15f0 #theta
dt = 1f-2

# Epsilon Noise parameters based on Yu et al. 2019
ζ = 0.0005f0
ξ_0 = 0.5f0
ξ_min = 0.1f0

# Noise actor
#noise_act = 0.1f0 #0.1f0 #2f-1
#noise_trg = 0.2f0 #3f-1

# Fill struct with values
ou = OUNoise(μ, σ, θ, dt, zeros(Float32, ACTION_SIZE))
gn = GNoise(μ, noise_act, noise_trg)
en = EpsNoise(ζ, ξ_0, ξ_min)
pn = ParamNoise(μ, σ, noise_act, 1.01)
