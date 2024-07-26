module ShemsEnv_LU1
# Ported from: https://github.com/openai/gym/blob/996e5115621bf57b34b9b79941e629a36a709ea1/gym/envs/classic_control/pendulum.py
#              https://github.com/openai/gym/wiki/Pendulum-v0
# add DataFrames to dependencies
# add shems environment to Reinforce import

using Reinforce: AbstractEnvironment
using LearnBase: IntervalSet
using RecipesBase
using Distributions: Uniform
using Random
using DataFrames, CSV

#-------------- EXTERNAL VARIABLES--------
Job_ID = ENV["JOB_ID"]

if parse(Int, Job_ID[end-1:end]) == 1
	DISCOMFORT_WEIGHT_EV = 2f0
	DISC_POT = 1
elseif parse(Int, Job_ID[end-1:end]) == 2
	DISCOMFORT_WEIGHT_EV = 0.5f0
	DISC_POT = 2f0
else
	DISCOMFORT_WEIGHT_EV = 1f0
	DISC_POT = 1f0
end

if parse(Int, Job_ID[end-1:end]) == 15
	penalty_weight = 1f0
else
	penalty_weight = 0.1f0
end

#DISCOMFORT_WEIGHT_EV = 1f0

#penalty_weight = 0.1f0

#1 + (parse(Int, Job_ID) % 10) # last digid from the Job_ID

charger_id = ((parse(Int, Job_ID) ÷ 100) % 100) # third and fourth last digid form JOB_ID

capacities = Dict{Int, Tuple{Float32, Float32, Float64}}(
    1 => (48.250f0, 7.5f0 * 0.9f0, 3.3),
    2 => (36.271f0, 10f0 * 0.9f0, 3.3),
    3 => (45.508f0, 10f0 * 0.9f0, 3.3),
    4 => (78.993f0, 11f0 * 0.9f0, 4.6),
    5 => (37.207f0, 10f0 * 0.9f0, 4.6),
    6 => (35.816f0, 15f0 * 0.9f0, 4.6),
    7 => (36.521f0, 12f0 * 0.9f0, 3.3),
    8 => (45.728f0, 10f0 * 0.9f0, 3.3),
    9 => (21.935f0, 7.5f0 * 0.9f0, 3.3),
    98 => (35.816f0, 7.5f0 * 0.9f0, 3.3)
)


import Reinforce: reset!, action, finished, step!, state

export
  Shems,  reset!,  step!,  action,  finished,  state,  track

struct PV
    eta::Float32
end

struct Battery
    eta::Float32
    soc_min::Float32
    soc_max::Float32
    rate_max::Float64
    loss::Float32
end

struct ElectricVehicle
    soc_min::Float32
    soc_max::Float32
    rate_max::Float32
end

struct Market
    sell_discount::Float64
	discomfort_weight_ev::Float64
	disc_pot::Float64
end

# PV(eta)
pv = PV(1f0); # PV(0.95f0); # 

# Battery(eta, soc_min, soc_max, rate_max, loss)
b = Battery(0.95f0, 0f0, capacities[1][2], capacities[1][3], 0.00003f0); # Battery(0.98f0, 0f0, 10f0, 4.6f0, 0.00003f0);

# ElectricVehicle(soc_min, soc_max, rate_max)
ev = ElectricVehicle(0f0, capacities[1][1], 11f0);

# Market(price, discomfort_weight)
m = Market(0.3f0, DISCOMFORT_WEIGHT_EV, DISC_POT) #10f0)		# Adjust here the penalty for not charging the full amount

mutable struct ShemsState{T<:AbstractFloat} <: AbstractVector{T}
  Soc_b::T
  Soc_ev::T  # Endogenous, whenever count-down starts or 1 when it's is off. Exogenous at all other times. When soc < 100% at departure: comfort violation
  c_ev::T  # count-down till ev departure. Endogenous. 0 (or -1?) when EV absent.
  d_e::T
  g_e::T
  p_buy::T
  h_cos::T
  h_sin::T
  season::T
end

# Soc_b is in kWh, but Soc_ev in %. Is that too confusing?

ShemsState() = ShemsState(0f0, 0f0, -1f0, 0f0, 0f0, 0f0, 1f0, 0f0, 1f0)

Base.size(::ShemsState) = (9,)

function Base.getindex(s::ShemsState, i::Int)
  (i > length(s)) && throw(BoundsError(s, i))
  	ifelse(i == 1, s.Soc_b,
  	ifelse(i == 2, s.Soc_ev,
	ifelse(i == 3, s.c_ev,
	ifelse(i == 4, s.d_e,
	ifelse(i == 5, s.g_e,
	ifelse(i == 6, s.p_buy,  #TBC
	ifelse(i == 7, s.h_cos,
	ifelse(i == 8, s.h_sin,
	s.season))))))))
end

function Base.setindex!(s::ShemsState, x, i::Int)
  (i > length(s)) && throw(BoundsError(s, i))
  setproperty!(s, ifelse(i == 1, :Soc_b,
  	ifelse(i == 1, s.Soc_b,
  	ifelse(i == 2, s.Soc_ev,
	ifelse(i == 3, s.c_ev,
	ifelse(i == 4, s.d_e,
	ifelse(i == 5, s.g_e,
	ifelse(i == 6, s.p_buy,
	ifelse(i == 7, s.h_cos,
	ifelse(i == 8, s.h_sin,
	:season))))))))), x)
	end

mutable struct ShemsAction{T<:AbstractFloat} <: AbstractVector{T}
  B::T
  EV::T
end

ShemsAction() = ShemsAction(0.7f0, 1f0)

Base.size(::ShemsAction) = (2,)
Base.minimum(::ShemsAction) = (0f0, 0f0)
Base.maximum(::ShemsAction) = (1f0, 1f0)

function Base.getindex(a::ShemsAction, i::Int)
  (i > length(a)) && throw(BoundsError(a, i))
  ifelse(i == 1, a.B,
  	a.EV)
end

function Base.setindex!(a::ShemsAction, x, i::Int)
  (i > length(a)) && throw(BoundsError(a, i))
  setproperty!(a, ifelse(i == 1, :B,
  	:EV), x)
end

mutable struct Shems{V<:AbstractVector, W<:AbstractVector} <: AbstractEnvironment
  state::V
  reward::Float64
  a::W
  step::Int
  maxsteps::Int
  idx::Int
  path::String
end

Base.size(::Shems) = (7,)

function Base.getindex(env::Shems, i::Int)
  (i > length(env)) && throw(BoundsError(env, i))
  	ifelse(i == 1, env.state,
  	ifelse(i == 2, env.reward,
	ifelse(i == 3, env.a,
	ifelse(i == 4, env.step,
	ifelse(i == 5, env.maxsteps,
	ifelse(i == 6, env.idx,
	env.path))))))
end

function Base.setindex!(env::Shems, x, i::Int)
  (i > length(env)) && throw(BoundsError(env, i))
  setproperty!(env, ifelse(i == 1, :state,
	  				  ifelse(i == 2, :reward,
					  ifelse(i == 3, :a,
					  ifelse(i == 4, :step,
					  ifelse(i == 5, :maxsteps,
					  ifelse(i == 6, :idx,
					  :path)))))), x)
end

Shems(maxsteps, path) = Shems(ShemsState(), 0.0, ShemsAction(), 0, maxsteps, 1, path)


function reset!(env::Shems; rng=0)
  idx = reset_state!(env, rng=rng)
  env.reward = 0.0
  env.a = ShemsAction()
  env.step = 0
  env.idx = idx
  env.path = env.path
  return env
end

function reset_state!(env::Shems; rng=0)
	df = CSV.read(env.path, DataFrame)

	# random components
	if rng == -1 #tracking/evaluation/testing always the same
		env.state.Soc_b = 0.5 * (b.soc_min + b.soc_max)
		idx = 1
    else #training/inference mean random
		env.state.Soc_b = rand(MersenneTwister(rng), Uniform(b.soc_min, b.soc_max))
		idx = rand(MersenneTwister(rng), 1:(nrow(df) - env.maxsteps))

		c_ev_end = df[(idx + env.maxsteps), :h_countdown]
		counter = 0
		max_iterations = 100  # Set your limit here

		while c_ev_end > -1 && idx < (nrow(df) - env.maxsteps)
			#println("reset_state problem. idx: $(idx), maxsteps: $(env.maxsteps), c_ev at $(idx+env.maxsteps) : $c_ev_end.")
			idx += Int(c_ev_end + 1)

			# If idx out of bound, draw new idx.
			if idx > (nrow(df) - env.maxsteps) 
				idx = rand(MersenneTwister(rng), 1:(nrow(df) - env.maxsteps))
			end

			#println("new index: $idx")
			c_ev_end = df[(idx + env.maxsteps), :h_countdown]
			#println("new idx: $idx, new c_ev_end = $c_ev_end.")
			
			counter += 1
			if counter > max_iterations
				println("Loop has exceeded maximum iterations, while trying to extend Training Episode in reset_state. Breaking...")
				break
			end
		end

	end

	
	env.state.Soc_ev = df[idx, :soc_ev]

	# endogenous states
	env.state.c_ev = df[idx, :h_countdown]
	env.state.d_e = df[idx, :electkwh]
	env.state.g_e = df[idx,:PV_generation]
	env.state.p_buy = df[idx,:p_buy]
	env.state.season = df[idx,:season]
	env.state.h_cos = df[idx,:hour_cos]
	env.state.h_sin = df[idx,:hour_sin]
	return idx
end

function next_state!(env::Shems)	# determining endogenous states for the next step
	df = CSV.read(env.path, DataFrame)
	idx = env.idx + 1

	env.state.c_ev = df[idx, :h_countdown]

	if env.state.c_ev >= 0 && df[env.idx, :h_countdown] == -1 # if EV is NEWLY connected
		env.state.Soc_ev = df[idx, :soc_ev] # load soc of newly arrived EV from the data
	end # else soc_ev was already connected at the previous step and the soc remains unchanged until an action is taken
	
	env.state.d_e = df[idx, :electkwh]
	env.state.g_e = df[idx,:PV_generation]
	env.state.p_buy = df[idx,:p_buy]
	env.state.season = df[idx,:season]
	env.state.h_cos = df[idx,:hour_cos]
	env.state.h_sin = df[idx,:hour_sin]
	return nothing
end

function action(env::Shems, a::ShemsAction)
		Soc_b, Soc_ev, c_ev, d_e, g_e, p_buy, h_cos, h_sin, season = env.state
		B_target, EV_target = a
		B, EV = zeros(2)

		Soc_b_perc = (Soc_b - b.soc_min) / (b.soc_max - b.soc_min)
        
		############################# Electric Vehicle ###############################
		# charge EV to max if SOC_ev less than EV_target%
		if c_ev > -1 && Soc_ev < EV_target
			# Fill EV up to target, or iwth max charge rate
			EV = min(ev.rate_max, (EV_target - Soc_ev) * (ev.soc_max - ev.soc_min))
		else
			EV = 0
		end

		############################# Battery ###############################
		# charge battery when surplus PV is available
		pv_ = g_e - d_e - EV  # or no -EV? in original code theres no -HP

		# charge to max if SOC less than B%
		if pv_ > 0 && Soc_b_perc < B_target
			# Fill up to target state
			B_target_value = B_target * (b.soc_max - b.soc_min) + b.soc_min
			B = clamp(pv_, 0, min(b.rate_max, (B_target_value - Soc_b + b.loss)))
		# discharge at max if no PV available (level regulated in step!)
		elseif Soc_b > 1f-3
			B = -min(b.rate_max,  ((1 - b.loss) * Soc_b))
		else
			B = 0
		end

	return Float32.([B, EV])
end

function action(env::Shems, track=-1)
	Soc_b, Soc_ev, c_ev, d_e, g_e, p_buy, h_cos, h_sin, season = env.state
    
	############################# Electric Vehicle ###############################
    # No smart charging here  (...?)
    EV = min(ev.rate_max, (1 - Soc_ev) * (ev.soc_max - ev.soc_min) )

	############################# Battery ###############################
	# charge battery when PV is available (substracting electr. demand)
	pv_ = g_e - d_e - EV

	# charge to max if SOC less than 95%
	if pv_ > 0 && Soc_b < (0.95 * b.soc_max)
		B = clamp(pv_, 0, min(b.rate_max, b.soc_max - Soc_b + b.loss))
	# discharge at max if no PV available (level regulated in step!)
	elseif Soc_b > 1f-3
		B = -min(b.rate_max,  ((1 - b.loss) * Soc_b))
	else
		B = 0
	end

return Float32.([B, EV])
end


function step!(env::Shems, s, a; track=0)
	Soc_b, Soc_ev, c_ev, d_e, g_e, p_buy, h_cos, h_sin, season = env.state

	if track >= 0
		B_target, EV_target = a  # soc targets are taken from the actors actions
		B, EV = action(env, ShemsAction(B_target, EV_target))  # actual charge amounts determined
		env.a = ShemsAction(B_target, EV_target) 
	elseif track < 0
		B_target, EV_target = zeros(Float32,2)
		B, EV = a
		env.a = ShemsAction(B_target, EV_target)
	end

  	pv_, BD, BC, EVC, abort, discomfort, profit, penalty = zeros(8)
	PV_DE, PV_B, PV_EV, PV_GR, B_DE, B_EV, B_GR, GR_DE, GR_EV, GR_B, EX_EV = zeros(11)

	############# DETERMINE FLOWS ###################################
	# fill only up to max level
	
	if B < -0.01 # battery discharging, restrictions discharging rate and soc
		BD = clamp(-B, 0.001, min(b.rate_max,  ((1 - b.loss - 1f-7) *Soc_b)) )
	end

	#---------------- PV generation greater than electricity demand -------------
	
	if (g_e * pv.eta) > d_e
		PV_DE = d_e
  		pv_ = (g_e * pv.eta) - PV_DE # PV left
		if  pv_ > EV
			PV_EV = EV
			pv_ -= PV_EV
		elseif pv_ <= EV
			PV_EV = pv_
			pv_ = 0
			if BD > (EV - PV_EV) / b.eta # EV from battery?
				B_EV = (EV - PV_EV)
				BD -= B_EV / b.eta
			elseif BD <= (EV - PV_EV) / b.eta
				B_EV = BD * b.eta
				BD = 0
				GR_EV = (EV - PV_EV) - B_EV 		# slack variable EV
			end
		end	
		
	# -------------- not enough PV for electr. demand --------------------------
	elseif (g_e * pv.eta) <= d_e # electr. demand
		PV_DE = g_e * pv.eta
		pv_ = 0
		d_e -= PV_DE
		if BD > (d_e / b.eta) # from battery?
			B_DE = d_e
			BD -= B_DE / b.eta
			if BD > (EV / b.eta)
				B_EV = EV
				BD -= B_EV / b.eta
			elseif BD <= (EV / b.eta)
				B_EV = BD * b.eta
				BD = 0
				GR_EV = EV - B_EV		# slack variable EV
			end
		elseif BD <= (d_e / b.eta)
			B_DE = BD * b.eta
			BD = 0
			GR_DE = d_e - B_DE						# slack variable demand
			GR_EV = EV
		end
	end

	# battery charging
	if B > 0.01
		BC = clamp(B, 0.001, min(b.rate_max, b.soc_max - Soc_b) )
		if  pv_ > (BC / b.eta)
			PV_B = BC
			pv_ -= (BC / b.eta)
		elseif pv_ <= (BC / b.eta)
			PV_B = pv_ * b.eta
			pv_ = 0
			GR_B = 0 #(BC - PV_B) / b.eta #-----------> #no grid charging
		end
	end

	PV_GR = pv_ 								# slack variable PV generation
	B_GR = 0 # BD * b.eta #..................> no grid discharging

	################### DETERMINE NEXT STATE ############################

	################### Next states ############################

	# Battery
	env.state.Soc_b = (1 - b.loss) * (Soc_b + PV_B + GR_B - ( (B_DE + B_EV + B_GR) / b.eta ) )  # new Soc_b in kWh

	# Electric Vehicle
	env.state.Soc_ev = Soc_ev + (PV_EV + B_EV + GR_EV) / (ev.soc_max - ev.soc_min) # new Soc_ev in %


	discomfort = 0
	penalty = 0
	EX_EV = 0

	if c_ev == 0 && env.state.Soc_ev < 1  # <0: disconnect/end of a transaction
		# not charged to potential (full, or what could have been)
		discomfort = (1 - env.state.Soc_ev) * 100
		EX_EV = (1 - env.state.Soc_ev) * (ev.soc_max - ev.soc_min) # kWh that was not charged into the EV and needs to be charged elsewhere
		env.state.Soc_ev = 1 # Electric Vehicle is disconnected and its soc is set to 1 for the duration of being disconnected.
	elseif c_ev < 0 && EV_target < 0.99
		penalty = (1 - EV_target) * penalty_weight
	end

	#penalty = 0 # testing runs without penalty!

	# Set uncertain parts of next state
	next_state!(env)
	env.step += 1
	env.idx += 1


	################### DETERMINE REWARD ############################

	b_degr = 0 #- 0.01 * (abs(B) > 0.01)   # abort penalty when discomfort abort
	abort = - 0 * finished(env, env.state)  # abort penalty when discomfort abort

	profit = (m.sell_discount * p_buy * (PV_GR + B_GR)) - (p_buy * (GR_DE + GR_B + GR_EV + EX_EV))

	if track < 0
		env.reward =  profit - (discomfort * m.discomfort_weight_ev) ^ (m.disc_pot) #+ b_degr + abort
		penalty = 0
	else
		env.reward =  profit - (discomfort * m.discomfort_weight_ev) ^ (m.disc_pot) - penalty #+ b_degr + abort
	end

	#results = hcat(Soc_b, Soc_ev, env.reward, comfort, b_degr+abort, PV_DE, B_DE, GR_DE,
	#				PV_B, PV_GR, PV_EV, B_EV, GR_EV, EX_EV, GR_B, B_GR, env.idx, B, B_target,
	#				EV, EV_target)
	results = hcat(env.idx, c_ev, EV_target, EV, Soc_ev, env.reward, profit, discomfort, penalty, PV_DE, B_DE, GR_DE,
					PV_B, PV_GR, PV_EV, B_EV, GR_EV, EX_EV, GR_B, B_GR, B, B_target, Soc_b
					)

	if track == 0
		return env.reward, Vector{Float32}(env.state)
	else
		return env.reward, Vector{Float32}(env.state), results
	end
end

function finished(env::Shems, s′)
	# indicate failure state / premature abort
	if env.step == env.maxsteps # only 24 time steps
	 	return false
	# elseif env.state.V_hw > hw.volume # tank volume 200l max
	# 	return true
	# elseif env.state.V_hw < 0 # tank volume 0l max
	# 	return true
	# elseif env.state.T_fh > fh.t_supply # heating can't exceed supply
	# 	return true
	# elseif env.state.T_fh < (0.8*fh.soc_min) # heating can't fall below 15.2°C
	# 	return true
	else
		return false
	end
end

# ------------------------------------------------------------------------

@recipe function f(env::Shems)
  legend := false
  link := :x
  xlims := (0, 2)
  #grid := false
  xticks := nothing
  layout := (3, 1)

  # battery state
  @series begin
	subplot := 1
	seriestype := :bar
	ylims := (b.soc_min, b.soc_max)
	fillcolor := :purple
    return [1], [env.state.Soc_b]
  end

  # battery range
  @series begin
	subplot := 1
	seriestype := :path
	ylims := (b.soc_min, b.soc_max)
	linecolor := :purple
	annotations := [(0.3, (b.soc_max - 0.5), "B: $(round(env.a[1], digits=3))", :top),
					(1.7, (b.soc_max - 0.5), "Hour: $(mod(env.idx-1, 24))", :top),
					(1.7, (b.soc_max - 3), "Reward: $(round(env.reward, digits=2))", :top),
					(1.7, (b.soc_max - 5.5), "Over?: $(finished(env, env.state))", :top)]
	return [0 0; 2 2], [b.soc_min b.soc_max; b.soc_min b.soc_max]
  end



end

end
