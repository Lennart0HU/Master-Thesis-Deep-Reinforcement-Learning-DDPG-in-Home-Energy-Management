# Parameters and architecture based on:
# https://github.com/fabio-4/JuliaRL/blob/master/algorithms/sac.jl
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/algorithms/policy_gradient/sac.jl
# Does not work on gpu!
using CUDA
CUDA.allowscalar(true)
#----------------------------- Model Architecture -----------------------------
γ = 0.995f0 #0.99f0     # discount rate for future rewards #Yu

τ = 5f-3 #1f-3       # Parameter for soft target network updatess
η_act = 3f-4 #1f-4   # Learning rate actor 10^(-4)
η_crit = 3f-4 #1f-3  # Learning rate critic
η_α = 3f-3  # learning rate of tuning entropy.

α = 0.2f0		# temperature trade-off entropy/rewards
target_entropy = -2f0 # dim actionspace

#L2_DECAY = 0.01f0

init = Flux.glorot_uniform(MersenneTwister(rng_run))
w_init(dims...) = 6f-3rand(MersenneTwister(rng_run), Float32, dims...) .- 3f-3

noise = CuArray{Float32}(undef, (ACTION_SIZE, 1));
noise_batch = CuArray{Float32}(undef, (ACTION_SIZE, BATCH_SIZE));

# Optimizers
opt_crit = ADAM(η_crit)
opt_act = ADAM(η_act)

struct Actor{S, A1, A2}
    model::S
    μ::A1
    logσ::A2
end

Flux.@functor Actor
(m::Actor)(s) = (l = m.model(s); (m.μ(l), m.logσ(l)))

actor = Actor(
    		Chain(Dense(STATE_SIZE, L1, relu), Dense(L1, L2, relu)) |> gpu,
    		Dense(L2, ACTION_SIZE, init=init) |> gpu,
    		Dense(L2, ACTION_SIZE, x -> clamp(x, typeof(x)(-10f0), typeof(x)(2f0)), init=w_init) |> gpu
)

critic1 = Chain(
			Dense(STATE_SIZE + ACTION_SIZE, L1, relu, init=init) |> gpu,
			Dense(L1, L2, relu, init=init) |> gpu,
  			Dense(L2, 1, init=w_init) |> gpu)

critic_target1 = deepcopy(critic1) |> gpu

critic2 = Chain(
			Dense(STATE_SIZE + ACTION_SIZE, L1, relu, init=init) |> gpu,
			Dense(L1, L2, relu, init=init) |> gpu,
  			Dense(L2, 1, init=w_init) |> gpu)

critic_target2 = deepcopy(critic2) |> gpu

# ---------------------- Param Update Functions --------------------------------
function soft_update!(target, model; τ = 1f0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t .= (1f0 - τ) * p_t .+ τ * p_m
  end
end

function update_model!(model, opt, loss, inp...)
  grads = gradient(() -> loss(inp...), params(model))
  update!(opt, params(model), grads)
end

# ---------------------------------- Training ----------------------------------
Flux.Zygote.@nograd Flux.params

function loss_crit(y, s_norm, a)
  q1 = critic1(vcat(s_norm, a))
  q2 = critic2(vcat(s_norm, a))
  return Flux.mse(q1, y) + Flux.mse(q2, y) |> gpu
end

function loss_act(s_norm, noise)
  actions, log_π = act(s_norm, noise, train=true) |> gpu
  Q_min = min.(critic1(vcat(s_norm, actions)), critic2(vcat(s_norm, actions)))
  return mean(α .* log_π .- Q_min) |> gpu
end

function update_alpha!(α, log_π)
	global α -= η_α * mean(-log_π .- target_entropy)
	return nothing
end

function replay(;train= true, rng_rpl=0)
  # retrieve minibatch from replay buffer
  s, a, r, s′ = getData(BATCH_SIZE, rng_dt=rng_rpl) |> gpu
  Random.seed!(rng_rpl)
  randn!(noise_batch)
  a′, log_π  = act(normalize(s′), noise_batch, train=train) |> gpu
  # update networks
  q′_min = min.(critic_target1(vcat(normalize(s′), a′)), critic_target2(vcat(normalize(s′), a′)))
  y = r .+ γ .* (q′_min .- α .* log_π)

  # update critic
  update_model!(critic1, opt_crit, loss_crit, y, normalize(s), a)
  update_model!(critic2, opt_crit, loss_crit, y, normalize(s), a)
  # update actor
  Random.seed!(rng_rpl+1)
  randn!(noise_batch)
  update_model!(actor, opt_act, loss_act, normalize(s), noise_batch)
#   grads = gradient(Flux.params(actor)) do
# 	actions, log_π = act(normalize(s), noise_batch, train=train) |> gpu
# 	Q_min = min.(critic1(vcat(normalize(s), actions)), critic2(vcat(normalize(s), actions)))
# 	α * mean(log_π) - mean(Q_min)
#   end
#   println(grads)
#   update!(opt_act, params(actor), grads)
  # update target networks
  soft_update!(critic_target1, critic1; τ = τ)
  soft_update!(critic_target2, critic2; τ = τ)
  # update alpha
  update_alpha!(α, log_π)
  return nothing
end

function loglikelihood(x, mu, logσ)
    return -5f-1 .* (((x .- mu) ./ (exp.(logσ) .+ 1f-8)) .^ 2f0 .+ 2f0 .* logσ .+ log(2f0*Float32(π)))
end

function act(s_norm, noise; train::Bool=true)
	μ, logσ = actor(s_norm) |> gpu
	if train == true
	  σ = exp.(logσ) |> gpu
	  ã = μ .+ σ .* noise
	  logpã = sum(loglikelihood(ã, μ, logσ), dims = 1) 
	  logpã -= sum((2.0f0 .* (log(2.0f0) .- ã .- softplus.(-2.0f0 .* ã))), dims = 1)
	  return tanh.(ã), logpã
	 elseif train == false
		return μ, logσ
	end
end

function scale_action(action)
	#scale action [-1, 1] to action bounds
	scaled_action = Float32.(ACTION_BOUND_LO .+
						(action .+ ones(ACTION_SIZE)) .*
							0.5 .* (ACTION_BOUND_HI .- ACTION_BOUND_LO))
	return scaled_action
end

function episode!(env::Shems; NUM_STEPS=EP_LENGTH["train"], train=true, render=false,
					track=0, rng_ep=0)
  reset!(env, rng=rng_ep) # rng = -1 sets evaluation/test initials
  local reward_eps=0f0
  local noise_eps=0f0
  local last_step = 1
  local results = Array{Float64}(undef, 0, 25)
  for step=1:NUM_STEPS
	# create individual random seed
	rng_step = parse(Int, string(abs(rng_ep))*string(step))
	# determine action
	s = copy(env.state)
	Random.seed!(rng_step)
	randn!(noise)
	a = act(normalize(s |> gpu), noise, train=train)[1] |> cpu
	scaled_action = scale_action(a)

	# execute action in RL environment
	if track == 0
		r, s′ = step!(env, s, scaled_action)
	elseif track == 1 # DRL
		r, s′, results_new = step!(env, s, scaled_action, track=track)
		results = vcat(results, results_new)
	elseif track < 0 # rule-based
		a = action(env, track)
		r, s′, results_new = step!(env, s, a, track=track)
		results = vcat(results, results_new)
	end

	# render step
	if render == true
		#sleep(1f-10)
		gui(plot(env))
		#plot(env) # for gif creation
		#frame(anim) # for gif creation
	end
	reward_eps += r
	#noise_eps += noise
	last_step = step

	# save step in replay buffer
	if train == true
      remember(s, a, r, s′)  #, finished(env, s′)) # for final state
	  # update network weights
	  replay(train=true, rng_rpl=rng_step)
	  # break episode in training
	  finished(env, s′) && break
    end
  end

  if track == 0
	  return (reward_eps / last_step), (noise_eps / last_step)
  else
      return (reward_eps / last_step), results
  end
end

function run_episodes(env_train::Shems, env_eval::Shems, total_reward, score_mean, best_run,
						noise_mean, test_every, render, rng; track=0)
	best_score = -1000
	for i=1:NUM_EP
		score=0f0
		score_all=0f0
		global current_episode = i
		# set individual seed per episode
		rng_ep = parse(Int, string(rng)*string(i))

		# Train set
		total_reward[i], noise_mean[i] = episode!(env_train, train=true, render=render,
													track=track, rng_ep=rng_ep)
		print("Episode: $i | Mean Step Score: $(@sprintf "%9.3f" total_reward[i]) |  $(α)  |  ")

		# test on eval data set
		if i % test_every == 1
			idx = ceil(Int32, i/test_every)
			# Evaluation set
			for test_ep in 1:test_runs
				rng_test= parse(Int, string(seed_ini)*string(test_ep))
				score, _ = episode!(env_eval, train=false, render=false,
										NUM_STEPS=EP_LENGTH["train"], track=track, rng_ep=rng_test)
				score_all += score
			end
			score_mean[idx] = score_all/test_runs
			print("Eval score $(@sprintf "%9.3f" score_mean[idx]) | ")
			#save weights for early stopping
			if score_mean[idx] > best_score
				# save network weights
				saveBSON(actor,	total_reward, score_mean, best_run, noise_mean,
									idx=i, path="temp", rng=rng_run)
				# set new best score
				best_score = score_mean[idx]
				# save best run
				global best_run = i
			end
		end
		t_elap = round(now()-t_start, Dates.Minute)
		println("Time elapsed: $(t_elap)")
	end
	return nothing
end
