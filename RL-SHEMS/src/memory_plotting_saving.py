import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

# Parameters
MIN_EXP_SIZE = 1000
EP_LENGTH = {"train": 200}
ACTION_SIZE = 1  # Adjust according to your action space size

# Memory initialization
memory = []

# --------------------------------- MEMORY -------------------------------------
def populate_memory(env, rng=0):
    global memory
    while len(memory) < MIN_EXP_SIZE:
        env.reset(rng=rng)
        for step in range(EP_LENGTH["train"]):
            rng2 = int(str(rng) + str(step))
            s = env.state.copy()
            a = np.random.rand(ACTION_SIZE) * 2 - 1
            scaled_action = scale_action(a)
            r, s_prime = env.step(s, scaled_action, track=0)
            remember(s, a, r, s_prime, env.finished(s_prime))
            if env.finished(s_prime):
                break
        rng += 1

def getData(batch_size, rng_dt=0):
    minibatch = np.random.choice(memory, batch_size)
    s = np.hstack([m[0] for m in minibatch])
    a = np.hstack([m[1] for m in minibatch])
    r = np.hstack([m[2] for m in minibatch])
    s_prime = np.hstack([m[3] for m in minibatch])
    done = np.hstack([m[4] for m in minibatch])
    return s, a, r, s_prime, done

#---------------------------- Helper MEMORY --------------------------------
def remember(state, action, reward, next_state, done):
    global memory
    memory.append([state, action, reward, next_state, done])

# --------------------------------- Data preprocessing -------------------------
def min_max_buffer(rng_mm):
    s, a, r, s_prime, done = getData(MIN_EXP_SIZE, rng_dt=rng_mm)
    scaler = MinMaxScaler()
    scaler.fit(s.T)
    s_min = scaler.data_min_
    s_max = scaler.data_max_
    return scaler.transform(s.T).T, s_min, s_max

def normalize(s, s_min, s_max):
    return (s - s_min) / (s_max - s_min + 1e-8)

# -------------------------------- Testing -------------------------------------

def inference(env, render=False, track=0, idx=NUM_EP, rng_inf=rng_run, best=False):
    reward = 0.0
    noise = 0.0
    if track != 0:
        reward, results = episode(env, NUM_STEPS=EP_LENGTH[season, run], train=False, render=render, track=track, rng_ep=-1)
        write_to_results_file(results, idx=idx, rng=rng_inf, best=best)
    elif render == True and track == 0:
        reward, noise = episode(env, NUM_STEPS=EP_LENGTH[season, run], train=False, render=True, track=track, rng_ep=-1)
    else:
        runs = 100
        rewards = np.zeros(runs)
        for e in range(runs):
            rewards[e], noise = episode(env, NUM_STEPS=EP_LENGTH[season, run], train=False, render=render, track=track, rng_ep=e)
        reward = np.mean(rewards)
    return reward

#------------------------------ PLOTTING --------------------------------------
def plot_scores(ymin, total_reward, score_mean, noise_mean, rng=rng_run):
    plt.scatter(range(1, NUM_EP + 1), total_reward, label="train", marker='o', markersize=2, alpha=1.0, color='turquoise')
    plt.plot(range(1, NUM_EP + 1), [np.mean(total_reward[max(1, i - 50):i]) for i in range(1, len(total_reward) + 1)], label="train (average last 50 episodes)", color='teal', alpha=0.4)
    plt.plot(range(1, NUM_EP + 1, test_every), score_mean, label=f"{run} (mean)", marker='o', markersize=3, color='indigo')
    plt.xlabel("Training episodes")
    plt.ylabel("Average score per time step [€] / noise")
    plt.legend(loc='lower right')
    plt.ylim(ymin, 1.5)
    plt.savefig(f"out/fig/{Job_ID}-{Task_ID}_DDPG_Shems_v12_{run}_{EP_LENGTH['train']}_{NUM_EP}_{L1}_{L2}_{case}_{rng}_{ymin}.png")
    plt.show()

def plot_all_scores(ymin, score_mean):
    all_score_mean = np.mean(score_mean, axis=1)
    all_score_std = np.std(score_mean, axis=1)
    all_score_min = np.min(score_mean, axis=1)
    all_score_max = np.max(score_mean, axis=1)
    print()
    print(f"Final score over {num_seeds} random seeds: {all_score_mean[-1]}")
    print(f"Final standard deviation over {num_seeds} random seeds: {all_score_std[-1]}")
    print(f"Final minimum over {num_seeds} random seeds: {all_score_min[-1]}")
    print(f"Final maximum over {num_seeds} random seeds: {all_score_max[-1]}")
    plt.plot(range(1, NUM_EP + 1, test_every), all_score_mean, label=f"{run} (mean)", marker='o', markersize=2, color='indigo')
    plt.fill_between(range(1, NUM_EP + 1, test_every), all_score_mean - 1.96 * all_score_std, all_score_mean + 1.96 * all_score_std, label="eval (95% confidence)", color='darkmagenta', alpha=0.4)
    plt.scatter(range(1, NUM_EP + 1, test_every), score_mean, label="", color='magenta', alpha=0.3)
    plt.scatter(range(1, NUM_EP + 1, test_every), all_score_max, label=f"{run} max", color='green', alpha=0.4)
    plt.scatter(range(1, NUM_EP + 1, test_every), all_score_min, label=f"{run} min", color='red', alpha=0.4)
    plt.xlabel("Training episodes")
    plt.ylabel("Average score per episode [€]")
    plt.legend(loc='lower right')
    plt.ylim(ymin, 0.5)
    plt.savefig(f"out/fig/{Job_ID}_DDPG_Shems_v12_{run}_{EP_LENGTH['train']}_{NUM_EP}_{L1}_{L2}_{case}_all_{ymin}.png")
    plt.show()

#------------------------------- SAVING ---------------------------------------
def write_to_results_file(results, idx=NUM_EP, rng=seed_run, best=False):
    date = datetime.now().date()
    if best == True:
        results.to_csv(f"out/tracker/{Job_ID}_{run}_results_v12_{EP_LENGTH['train']}_{NUM_EP}_{L1}_{L2}_{case}_{rng}_best.csv", index=False)
    elif
