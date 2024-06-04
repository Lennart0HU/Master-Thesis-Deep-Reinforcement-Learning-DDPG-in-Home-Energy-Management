
# Parameters and architecture based on:
# https://github.com/msinto93/DDPG/blob/master/train.py
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl
# https://github.com/FluxML/model-zoo/blob/master/contrib/games/differentiable-programming/pendulum/DDPG.jl

#--------------------------------- MEMORY -------------------------------------
# Populate memory with random actions
function populate_memory(env::Shems; rng=0)
	while length(memory) < MIN_EXP_SIZE
		reset!(env, rng=rng)
		for step=1:EP_LENGTH["train"]
		  # create individual random seeds
		  rng2 = parse(Int, string(rng)*string(step))
		  s = copy(env.state)
		  # random values between -1 and 1
		  a = Float32.(rand(MersenneTwister(rng2), ACTION_SIZE) .* 2 .- 1)
		  # sclae to action bounds
		  scaled_action = scale_action(a)
		  # execute action in RL environment
		  r, s′ = step!(env, s, scaled_action, track=0)
		  # store tuple in experience buffer
		  remember(s, a, r, s′, finished(env, s′))
		  finished(env, s′) && break
		end
		rng += 1
	end
	return nothing
end

function getData(batch_size; rng_dt=0)
  # Getting data in shape
  minibatch = sample(MersenneTwister(rng_dt), memory, batch_size)

  s       =   reduce(hcat,getindex.(minibatch,1))  |> gpu
  a       =   reduce(hcat,getindex.(minibatch,2))  |> gpu
  r       =   reduce(hcat,getindex.(minibatch,3))  |> gpu
  s′ 	  =   reduce(hcat,getindex.(minibatch,4))  |> gpu
  done    =   reduce(hcat,getindex.(minibatch,5))  |> gpu

  return s, a, r, s′, done
end

#---------------------------- Helper MEMORY --------------------------------
# Stores tuple of state, action, reward, next_state, and done
remember(state, action, reward, next_state, done) =
	push!(memory, [state, action, reward, next_state, done])

# --------------------------------- Data preprocessing -------------------------
function min_max_buffer(MIN_EXP_SIZE; rng_mm=rng)
	s, a, r, s′, done = getData(MIN_EXP_SIZE, rng_dt=rng_mm) 
	return minimum(s, dims=2), maximum(s, dims=2)
end

function normalize(s; s_min=s_min, s_max=s_max)
	return (s .- s_min) ./ (s_max .- s_min .+ 1f-8)
end

# -------------------------------- Testing -------------------------------------

# runs steps through data set without reset, rendering decisions
function inference(env::Shems; render=false, track=0, idx=NUM_EP, rng_inf=rng_run, best=false)
	local reward = 0f0
	local noise=0f0
	
	# tracking flows
	if track != 0
	  reward, results = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false,
								  render=render, track=track, rng_ep=-1)
	  write_to_results_file(results, idx=idx, rng=rng_inf, best=best)
	  return nothing
  
	# rendering flows
	elseif render == true && track == 0
	  reward, noise = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false,
								  render=true, track=track, rng_ep=-1)
	  return nothing
	  
	# return mean reward over 100 eps
	else
	  local runs=100
	  local reward = zeros(runs)
	  for e = 1:runs
		  reward[e], noise = episode!(env, NUM_STEPS=EP_LENGTH[season, run], train=false,
										  render=render, track=track, rng_ep=e)
	  end
	  return (reward / runs)
	end
  end

#------------------------------ PLOTTING --------------------------------------
function plot_scores(;ymin=Inf, total_reward=total_reward, score_mean=score_mean, noise_mean=noise_mean,
						rng=rng_run)
	# plot training results
	scatter(1:NUM_EP, [total_reward], label="train",
			markershape=:circle, markersize=2, markerstrokewidth=0.1,
			legend=:bottomright, ylim=(ymin, 1.5), colour = :turquoise,
			left_margin=7Plots.mm, bottom_margin=6Plots.mm);
	# plot training moving average
	plot!(1:NUM_EP, [mean(total_reward[max(1, i-50):i]) for i in 1:length(total_reward)],
			label="train (average last 50 episodes)",
			colour = :teal, markerstrokewidth=0.2, alpha=0.4);
	# plot training moving average
	plot!(1:NUM_EP, [noise_mean], label="noise",
					colour = :orange, markerstrokewidth=0.2, alpha=0.8);
	# plot test/evaluation results mean
	plot!(1:test_every:NUM_EP, score_mean[:], label="$(run) (mean)",
			markershape=:circle, colour =:indigo, markersize=3, markerstrokewidth=0.2);

	yaxis!("Average score per time step [€] / noise", font(10, "serif"));
	xaxis!("Training episodes", font(10, "serif"));

	savefig("out/fig/$(Job_ID)-$(Task_ID)_DDPG_Shems_Charger_v1_$(run)_$(EP_LENGTH["train"])"*
				"_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_$(ymin).png");
end

function plot_all_scores(;ymin=Inf, score_mean=score_mean)
	all_score_mean = mean(score_mean, dims=2);
	all_score_std = std(score_mean, dims=2);
	all_score_min = minimum(score_mean, dims=2);
	all_score_max = maximum(score_mean, dims=2);

	println()
	println("Final score over $(num_seeds) random seeds: $(all_score_mean[end])");
	println("Final standard deviation over $(num_seeds) random seeds: $(all_score_std[end])");
	println("Final minimum over $(num_seeds) random seeds: $(all_score_min[end])");
	println("Final maximum over $(num_seeds) random seeds: $(all_score_max[end])");
	println(" $(all_score_mean[end]) $(all_score_std[end]) $(all_score_min[end]) $(all_score_max[end])")

	# plot test/evaluation results mean -last
	plot(1:test_every:NUM_EP, [all_score_mean], label="$(run) (mean)",
			markersize=2, markerstrokewidth=0.1, ylim=(ymin, 0.5),
			legend=:bottomright, markershape=:circle, colour =:indigo,
			left_margin=7Plots.mm, bottom_margin=6Plots.mm);

	# plot test/evaluation 95% confodence interval (n=100) -last
	plot!(1:test_every:NUM_EP, [all_score_mean .+ 1.96 .* all_score_std],
			fillrange = [all_score_mean .- 1.96 .* all_score_std],
			label="eval (95% confidence)", colour =:darkmagenta, alpha=0.4);

	# # plot test/evaluation 95% confodence inteval (n=100) -last
	# plot!(1:test_every:NUM_EP, [all_score_max],
	# fillrange = [all_score_min],
	# label="$(run)  (95% confidence interval)", colour =:darkmagenta, alpha=0.4);

	# plot test/evaluation 95% confodence inteval (n=100) -last
	scatter!(1:test_every:NUM_EP, [score_mean],
	markersize=2, label="", colour =:magenta, alpha=0.3);

	# plot test/evaluation 95% confodence inteval (n=100) -last
	scatter!(1:test_every:NUM_EP, [all_score_max],
				markersize=4, label="$(run) max", colour =:green, alpha=0.4);

	# plot test/evaluation 95% confodence inteval (n=100) -last
	scatter!(1:test_every:NUM_EP, [all_score_min],
				markersize=4, label="$(run) min", colour =:red, alpha=0.4);


	yaxis!("Average score per episode [€]", font(10, "serif"));
	xaxis!("Training episodes", font(10, "serif"));

	savefig("out/fig/$(Job_ID)_DDPG_Shems_Charger_v1_$(run)_$(EP_LENGTH["train"])_"*
				"$(NUM_EP)_$(L1)_$(L2)_$(case)_all_$(ymin).png");
end

#------------------------------- SAVING ---------------------------------------
function write_to_results_file(results; idx=NUM_EP, rng=seed_run, best=false)
    date=Date(now());
	if best == true
		CSV.write("out/tracker/$(Job_ID)_$(run)_results_charger_v1_$(EP_LENGTH["train"])_"*
		"$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_best.csv",
			DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
			"T_FH_plus", "T_FH_minus", "V_HW_plus", "V_HW_minus",
			"rewards", "comfort", "rewART", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
			"PV_HP","GR_HP", "GR_B", "B_HP","B_GR", "HP_FH", "HP_HW","index", 
			"B", "HP", "B_tar", "FH_tar", "HW_tar"]);

	elseif idx == NUM_EP
    CSV.write("out/tracker/$(Job_ID)_$(run)_results_charger_v1_$(EP_LENGTH["train"])_"*
				"$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_$(idx).csv",
					DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
					"T_FH_plus", "T_FH_minus", "V_HW_plus", "V_HW_minus",
					"rewards", "comfort", "rewART", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
					"PV_HP","GR_HP", "GR_B", "B_HP", "B_GR", "HP_FH", "HP_HW","index", 
					"B", "HP", "B_tar", "FH_tar", "HW_tar"]);
	elseif idx < 0
	CSV.write("out/tracker/$(Job_ID)_$(run)_results_$(case)_rule_$(idx).csv",
				DataFrame(results, :auto), header=["Temp_FH", "Vol_HW", "Soc_B",
				"T_FH_plus", "T_FH_minus", "V_HW_plus", "V_HW_minus",
				"rewards", "comfort", "rewART", "COP_FH","COP_HW","PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR",
				"PV_HP","GR_HP", "GR_B", "B_HP","B_GR", "HP_FH", "HP_HW","index", 
				"B", "HP", "B_tar", "FH_tar", "HW_tar"]);
	end
    return nothing
end

function write_to_tracker_file(;idx=NUM_EP, rng=rng_run, best=false)
	time=now();
	date=Date(now());
	if best == true
		results = CSV.read("out/tracker/$(Job_ID)_$(run)_results_charger_v1_$(EP_LENGTH["train"])_$(NUM_EP)_"*
								"$(L1)_$(L2)_$(case)_$(rng)_best.csv", DataFrame)
	elseif idx == NUM_EP
		results = CSV.read("out/tracker/$(Job_ID)_$(run)_results_charger_v1_$(EP_LENGTH["train"])_$(NUM_EP)_"*
								"$(L1)_$(L2)_$(case)_$(rng)_$(idx).csv", DataFrame)
	else
		results = CSV.read("out/tracker/$(Job_ID)_$(run)_results_$(case)_rule_$(idx).csv", DataFrame)
	end
	# Overall tracker (rewards without comfort costs)
	Tracker = CSV.read("out/Tracker.csv", DataFrame, header=true);
	Tracker = vcat(Matrix(Tracker), [time NUM_EP L1  L2  BATCH_SIZE  MEM_SIZE  MIN_EXP_SIZE season  run Job_ID rng case  best idx [
								sum(results[!, :T_FH_plus])]  sum(results[!, :T_FH_minus]) [
								sum(results[!, :V_HW_plus])]  sum(results[!, :V_HW_minus]) [
								sum(results[!, :rewards] .-results[!, :comfort] .-results[!, :rewART])] sum(results[!, :comfort]) sum(results[!, :rewART]) ]);

    CSV.write("out/Tracker.csv", DataFrame(Tracker,:auto), header=["time", "NUM_EP", "L1", "L2", "BATCH_SIZE", "MEM_SIZE",
								"MIN_EXP_SIZE","season", "run", "Job_ID", "seed", "case", "best", "idx", "T_FH_plus", "T_FH_minus", "V_HW_plus",
								"V_HW_minus", "rewards", "comfort", "rewART"]);

	# # Cost tracker
	# Costs = Matrix(CSV.read("out/Costs_$(run).csv", DataFrame, header=true))
	# Costs = hcat(Costs, [time NUM_EP L1  L2  BATCH_SIZE  MEM_SIZE  MIN_EXP_SIZE season  run Job_ID s_rng case idx transpose(results[!, :profits] .- results[!, :comfort])])
    # CSV.write("out/Costs_$(run).csv", DataFrame(Costs,:auto), header=vcat(["time", "NUM_EP", "L1", "L2", "BATCH_SIZE", "MEM_SIZE",
	# 							"MIN_EXP_SIZE","season", "run", "Job_ID", "seed", "case", "idx"], string.([i  for i=1:EP_LENGTH[season, run]])));

	# # Cost tracker
	# Cost = CSV.read("out/Costs_$(run)_v2.csv", DataFrame, header=true)
	# Costs = hcat(Matrix(Cost), results[!, :profits] .- results[!, :comfort])
    # CSV.write("out/Costs_$(run)_v2.csv", DataFrame(Costs,:auto), header=[names(Cost); "$(Job_ID)_$(rng)_$(idx)"]);
	#
	# # Comfort tracker
	# Comfort = CSV.read("out/Comfort_$(run)_v2.csv", DataFrame, header=1)
	# Comforts = hcat(Matrix(Comfort), results[!, :comfort])
	# CSV.write("out/Comfort_$(run)_v2.csv", DataFrame(Comforts,:auto), header=[names(Comfort); "$(Job_ID)-$(rng)-$(idx)"]);

	# # Comfort tracker
	# Comfort = Matrix(CSV.read("out/Comfort_$(run).csv", DataFrame, header=true))
	# Comfort = vcat(Comfort, [time NUM_EP L1  L2  BATCH_SIZE  MEM_SIZE  MIN_EXP_SIZE season  run Job_ID s_rng case idx results[!, :comfort]'])
    # CSV.write("out/Comfort_$(run).csv", DataFrame(Comfort,:auto), header=vcat(["time", "NUM_EP", "L1", "L2", "BATCH_SIZE", "MEM_SIZE",
	# 							"MIN_EXP_SIZE","season", "run", "Job_ID", "seed", "case", "idx"], string.([i  for i=1:EP_LENGTH[season, run]])));
    return nothing
end

function saveBSON(actor, total_reward, score_mean, best_run, noise_mean;
					idx=NUM_EP, path="", rng=rng_run)
	actor = cpu(actor)

	BSON.@save "out/bson/$(path)/DDPG_Shems_Charger_v1_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_actor_$(idx).bson" actor
	BSON.@save "out/bson/$(path)/DDPG_Shems_Charger_v1_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_scores_$(idx).bson" total_reward score_mean best_run noise_mean
	return nothing
end

function loadBSON(;idx=NUM_EP, scores_only=false, path="", rng=seed_run)
	if scores_only==true
		BSON.@load "out/bson/$(path)/DDPG_Shems_Charger_v1_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_scores_$(idx).bson" total_reward score_mean best_run noise_mean
		return total_reward |> gpu, score_mean |> gpu, best_run |> gpu, noise_mean |> gpu
	else
		BSON.@load "out/bson/$(path)/DDPG_Shems_Charger_v1_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_actor_$(idx).bson" actor
		BSON.@load "out/bson/$(path)/DDPG_Shems_Charger_v1_$(EP_LENGTH["train"])_$(NUM_EP)_$(L1)_$(L2)_$(case)_$(rng)_scores_$(idx).bson" total_reward score_mean best_run noise_mean
		return actor |> gpu, total_reward |> gpu, score_mean |> gpu, best_run |> gpu, noise_mean |> gpu
	end
end
