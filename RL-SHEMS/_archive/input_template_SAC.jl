algo="SAC_sep"
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
# Job_ID=10000002
# seed_run=1
# Task_ID=10
#--------cluster jobs------------
Job_ID = ENV["JOB_ID"]
Task_ID = ENV["SGE_TASK_ID"]
seed_run = parse(Int, Task_ID)
#-------------------------------- INPUTS --------------------------------------------
train = 1
plot_result = 0
plot_all = 1
render = 0
track = 1  # 0 - off, 1 - DRL, , rule-based percentage of start Soc e.g. 70% -> -0.7 (has to be negative)

season = "all"
price="fix"  # "fix" "TOU"

using Reinforce.ShemsEnv_H8: Shems
case = "$(season)_$(algo)_$(price)_auto_Env-H8"
run = "eval" # "eval", "test"
NUM_EP = 3_001 #50_000
L1 = 256 #300
L2 = 256 #600
idx=NUM_EP
test_every = 100
test_runs = 100
num_seeds = 10

#-------------------------------------
seed_ini = 123
# individual random seed for each run
rng_run = parse(Int, string(seed_ini)*string(seed_run))

start_time = now()
current_episode = 0

#--------------------------------- Memory ------------------------------------
BATCH_SIZE = 120 #256 #100
MEM_SIZE = 20_000
MIN_EXP_SIZE = 20_000

########################################################################################
memory = CircularBuffer{Any}(MEM_SIZE)

#--------------------------------- Game environment ---------------------------
EP_LENGTH = Dict("train" => 24,
					("summer", "eval") => 359, ("summer", "test") => 767,
					("winter", "eval") => 359, ("winter", "test") => 719,
					("both", "eval") => 719,   ("both", "test") => 1487,
					("all", "eval") => 1439,   ("all", "test") => 2999) # length of whole evaluation set (different)

env_dict = Dict("train" => Shems(EP_LENGTH["train"], "data/$(season)_train_$(price).csv"),
				"eval" => Shems(EP_LENGTH[season, "eval"], "data/$(season)_eval_$(price).csv"),
				"test" => Shems(EP_LENGTH[season, "test"], "data/$(season)_test_$(price).csv"))

WAIT = Dict(
          ("summer", "DDPG") => 1500, ("summer", "TD3") => 1500, ("summer", "SAC_sep") => 5000,
          ("winter", "DDPG") => 2000, ("winter", "TD3") => 2000, ("winter", "SAC_sep") => 7000,
          ("both", "DDPG") => 3000,   ("both", "TD3") => 3000,  ("both", "SAC_sep") => 10000,
          ("all", "DDPG") => 4000,    ("all", "TD3") => 5000,   ("all", "SAC_sep") => 20_000) 

# ----------------------------- Environment Parameters -------------------------
STATE_SIZE = length(env_dict["train"].state)
ACTION_SIZE = length(env_dict["train"].a)
ACTION_BOUND_HI = maximum(env_dict["train"].a)
ACTION_BOUND_LO = minimum(env_dict["train"].a)
#ACTION_BOUND_HI = Float32[1f0, 1f0] #Float32(actions(env, env.state).hi[1])
#ACTION_BOUND_LO = Float32[-1f0, -1f0] #Float32(actions(env, env.state).lo[1])
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
σ = 0.1f0 #sigma
θ = 0.15f0 #theta
dt = 1f-2
# Epsilon Noise parameters based on Yu et al. 2019
ζ = 0.0005f0
ξ_0 = 0.5f0
ξ_min = 0.1f0

# Noise actor
noise_act = 1f-1 #2f-1
noise_trg = 2f-1 #3f-1

# Fill struct with values
ou = OUNoise(μ, σ, θ, dt, zeros(Float32, ACTION_SIZE))
gn = GNoise(μ, noise_act, noise_trg)
en = EpsNoise(ζ, ξ_0, ξ_min)
pn = ParamNoise(μ, σ, noise_act, 1.01)
