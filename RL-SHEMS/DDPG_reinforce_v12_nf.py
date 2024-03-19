import time
import numpy as np
import os
from datetime import datetime

try:
    print("Use cluster setup.")
    exec(open("out/input/" + os.environ["JOB_ID"] + "--input.py").read())
    time.sleep(400)  # Wait for input to load before calling algo
except:
    print("Use single run setup.")
    exec(open("input.py").read())

exec(open("algorithms/" + algo + ".py").read())  # Training functions
exec(open("src/memory_plotting_saving.py").read())  # Plotting and rendering functions

populate_memory(env_dict["train"], rng=rng_run)
s_min, s_max = min_max_buffer(MIN_EXP_SIZE, rng_mm=rng_run)

total_reward = np.zeros(NUM_EP, dtype=np.float32)
noise_mean = np.zeros(NUM_EP, dtype=np.float32)
best_run = 0
score_mean = np.zeros((int(np.ceil(NUM_EP / test_every)),), dtype=np.int32)

if train:
    t_start = datetime.now()
    print(f"Max steps: {EP_LENGTH['train']}, Max episodes: {NUM_EP}, Layer 1: {L1} nodes, Layer 2: {L2} nodes, Case: {case}, Time to start: {(t_start - start_time).minutes}")
    run_episodes(env_dict["train"], env_dict["eval"], total_reward, score_mean, best_run, noise_mean, test_every, render, rng_run, track=0)
    saveBSON(actor, total_reward, score_mean, best_run, noise_mean, rng=rng_run)

if render and not train:
    actor, total_reward, score_mean, best_run, noise_mean = loadBSON(idx=idx, rng=rng_run)
    inference(env_dict[run], render=True, track=0, idx=idx)

if plot_result:
    total_reward, score_mean, best_run, noise_mean = loadBSON(scores_only=True, rng=rng_run)
    print(f"train (last {int(len(total_reward) / 20) + 1})= {np.mean(total_reward[-int(len(total_reward) / 20):])}")
    print(f"eval (last {int(score_mean.shape[0] / 10) + 1})= {np.mean(score_mean[-int(score_mean.shape[0] / 10):, 0])}")
    plot_scores(ymin=-2, rng=rng_run)

if plot_all:
    if seed_run == num_seeds:
        time.sleep(WAIT[season][algo])
        score_mean_all = np.zeros((int(np.ceil(NUM_EP / test_every)), num_seeds), dtype=np.float32)
        for i in range(1, num_seeds + 1):
            test_rng_run = int(str(seed_ini) + str(i))
            score_mean_all[:, i - 1] = loadBSON(scores_only=True, rng=test_rng_run)[1]
        plot_all_scores(ymin=-50, score_mean=score_mean_all)

if track == 1:
    if seed_run == num_seeds:
        for i in range(1, num_seeds + 1):
            test_rng_run = int(str(seed_ini) + str(i))
            ac, tr, sm, best_eval, nm = loadBSON(rng=test_rng_run)
            actor = deepcopy(ac)
            inference(env_dict[run], render=False, track=track, rng_inf=test_rng_run)
            write_to_tracker_file(rng=test_rng_run)
            ac = loadBSON(idx=best_eval, path="temp", rng=test_rng_run)[0]
            actor = deepcopy(ac)
            inference(env_dict[run], render=False, track=track, idx=best_eval, rng_inf=test_rng_run, best=True)
            write_to_tracker_file(idx=best_eval, rng=test_rng_run, best=True)
elif track < 0:
    inference(env_dict[run], render=False, track=track, idx=track)
    write_to_tracker_file(idx=track, rng=track)
