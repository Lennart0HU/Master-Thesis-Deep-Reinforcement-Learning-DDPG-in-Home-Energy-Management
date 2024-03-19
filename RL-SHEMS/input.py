import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import datetime

# Setting up the parameters and configurations
algo = "DDPG"

# Paths to data files or directories might need to be adjusted
data_path = "./data/"  # Path to data files
model_path = "./models/"  # Path to save models

# Training and plotting flags
train_flag = False
plot_result_flag = False
plot_all_flag = False
render_flag = False

# Percentage of starting SOC (State of Charge)
track = -0.7  # 0 - off, 1 - DRL, , rule-based percentage of start SOC e.g. 70% -> -0.7 (has to be negative)

# Season and price type
season = "all"  # "all", "both", "summer", "winter"
price = "fix"  # "fix", "TOU"

# Noise type for exploration
noise_type = "gn"  # "ou", "pn", "gn", "en"

# Training parameters
NUM_EP = 3001
L1 = 300
L2 = 600
test_every = 100
test_runs = 100
num_seeds = 40

# Random seeds and current time
seed_ini = 123
current_time = datetime.datetime.now()

# Memory parameters
BATCH_SIZE = 120
MEM_SIZE = 24000
MIN_EXP_SIZE = 24000

# Length of episodes for different scenarios
EP_LENGTH = {
    "train": 24,
    ("summer", "eval"): 359, ("summer", "test"): 767,
    ("winter", "eval"): 359, ("winter", "test"): 719,
    ("both", "eval"): 719, ("both", "test"): 1487,
    ("all", "eval"): 1439, ("all", "test"): 2999
}

# Wait time between actions for different scenarios
WAIT = {
    ("summer", "DDPG"): 1500, ("summer", "TD3"): 1500, ("summer", "SAC"): 5000,
    ("winter", "DDPG"): 2000, ("winter", "TD3"): 2000, ("winter", "SAC"): 7000,
    ("both", "DDPG"): 3000, ("both", "TD3"): 3000, ("both", "SAC"): 10000,
    ("all", "DDPG"): 4000, ("all", "TD3"): 5000, ("all", "SAC"): 20000
}

# Environment parameters
STATE_SIZE =  # Define the state size based on the environment
ACTION_SIZE =  # Define the action size based on the environment
ACTION_BOUND_HI =  # Define the upper bound for actions
ACTION_BOUND_LO =  # Define the lower bound for actions

# Noise parameters
mu = 0.0
sigma = 0.1
theta = 0.15
dt = 1e-2

epsilon = 0.0005
xi_0 = 0.5
xi_min = 0.1

noise_act = 0.1
noise_trg = 0.2

# Structures for different noise types
class OUNoise:
    def __init__(self, mu, sigma, theta, dt, action_size):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.X = np.zeros(action_size)

class GNoise:
    def __init__(self, mu, sigma_act, sigma_trg):
        self.mu = mu
        self.sigma_act = sigma_act
        self.sigma_trg = sigma_trg

class EpsNoise:
    def __init__(self, zeta, xi, xi_min):
        self.zeta = zeta
        self.xi = xi
        self.xi_min = xi_min

class ParamNoise:
    def __init__(self, mu, sigma_current, sigma_target, adoption):
        self.mu = mu
        self.sigma_current = sigma_current
        self.sigma_target = sigma_target
        self.adoption = adoption

# Initialize noise structures
ou = OUNoise(mu, sigma, theta, dt, ACTION_SIZE)
gn = GNoise(mu, noise_act, noise_trg)
en = EpsNoise(epsilon, xi_0, xi_min)
pn = ParamNoise(mu, sigma, noise_act, 1.01)

# Save settings for later use
settings = {
    "algo": algo,
    "data_path": data_path,
    "model_path": model_path,
    "train_flag": train_flag,
    "plot_result_flag": plot_result_flag,
    "plot_all_flag": plot_all_flag,
    "render_flag": render_flag,
    "track": track,
    "season": season,
    "price": price,
    "noise_type": noise_type,
    "NUM_EP": NUM_EP,
    "L1": L1,
    "L2": L2,
    "test_every": test_every,
    "test_runs": test_runs,
    "num_seeds": num_seeds,
    "seed_ini": seed_ini,
    "current_time": current_time,
    "BATCH_SIZE": BATCH_SIZE,
    "MEM_SIZE": MEM_SIZE,
    "MIN_EXP_SIZE": MIN_EXP_SIZE,
    "EP_LENGTH": EP_LENGTH,
    "WAIT": WAIT,
    "STATE_SIZE": STATE_SIZE,
    "ACTION_SIZE": ACTION_SIZE,
    "ACTION_BOUND_HI": ACTION_BOUND_HI,
    "ACTION_BOUND_LO": ACTION_BOUND_LO,
    "ou": ou,
    "gn": gn,
    "en": en,
    "pn": pn
}
