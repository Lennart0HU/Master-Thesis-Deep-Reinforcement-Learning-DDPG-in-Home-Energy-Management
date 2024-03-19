import numpy as np
import pandas as pd
from scipy.stats import uniform

# Define constants
p_concr = 2400.0  # kg/m^3
c_concr = 1.0      # kJ/(kg*°C)
p_water = 997.0    # kg/m^3
c_water = 4.184    # kJ/(kg*°C)

class HeatPump:
    def __init__(self, rate_max):
        self.rate_max = rate_max

class PV:
    def __init__(self, eta):
        self.eta = eta

class Battery:
    def __init__(self, eta, soc_min, soc_max, rate_max, loss):
        self.eta = eta
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.rate_max = rate_max
        self.loss = loss

class ThermalStorage:
    def __init__(self, volume, loss, t_supply, soc_min, soc_max):
        self.volume = volume
        self.loss = loss
        self.t_supply = t_supply
        self.soc_min = soc_min
        self.soc_max = soc_max

class Market:
    def __init__(self, sell_discount, comfort_weight_hw, comfort_weight_fh):
        self.sell_discount = sell_discount
        self.comfort_weight_hw = comfort_weight_hw
        self.comfort_weight_fh = comfort_weight_fh

class ShemsState(np.ndarray):
    def __new__(cls):
        obj = np.zeros(12, dtype=np.float32)
        obj[10] = 1.0  # Set default season value to 1.0
        return obj

class ShemsAction(np.ndarray):
    def __new__(cls):
        obj = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        return obj

class ShemsEnvironment:
    def __init__(self, maxsteps, path):
        self.state = ShemsState()
        self.reward = 0.0
        self.a = ShemsAction()
        self.step = 0
        self.maxsteps = maxsteps
        self.idx = 1
        self.path = path

    def reset(self, rng=0):
        self.idx = self.reset_state(rng=rng)
        self.reward = 0.0
        self.a = ShemsAction()
        self.step = 0
        return self

    def reset_state(self, rng=0):
        df = pd.read_csv(self.path)
        if rng == -1:  # Tracking/evaluation/testing always the same
            self.state[0] = 0.5 * (b.soc_min + b.soc_max)
            self.state[1] = 0.5 * (fh.soc_min + fh.soc_max)
            self.state[2] = 0.5 * (hw.soc_min + hw.soc_max)
            idx = 1
        else:  # Training/inference mean random
            self.state[0] = uniform.rvs(b.soc_min, b.soc_max)
            self.state[1] = uniform.rvs(fh.soc_min, fh.soc_max)
            self.state[2] = uniform.rvs(hw.soc_min, hw.soc_max)
            idx = np.random.randint(1, df.shape[0] - self.maxsteps + 1)
        self.state[3] = df.loc[idx, 'electkwh']
        self.state[4] = df.loc[idx, 'heatingkwh']
        self.state[5] = df.loc[idx, 'hotwaterkwh']
        self.state[6] = df.loc[idx, 'PV_generation']
        self.state[7] = df.loc[idx, 'Temperature']
        self.state[8] = df.loc[idx, 'p_buy']
        self.state[10] = df.loc[idx, 'season']
        self.state[11] = df.loc[idx, 'hour_sin']
        return idx

    def next_state(self):
        df = pd.read_csv(self.path)
        self.idx += 1
        self.state[3] = df.loc[self.idx, 'electkwh']
        self.state[4] = df.loc[self.idx, 'heatingkwh']
        self.state[5] = df.loc[self.idx, 'hotwaterkwh']
        self.state[6] = df.loc[self.idx, 'PV_generation']
        self.state[7] = df.loc[self.idx, 'Temperature']
        self.state[8] = df.loc[self.idx, 'p_buy']
        self.state[10] = df.loc[self.idx, 'season']
        self.state[11] = df.loc[self.idx, 'hour_sin']

    def cop_calc(self, ts):
        return max(5.8 - (1 / 14 * abs(ts.t_supply - self.state[7])), 0)

    def is_hot(self):
        return self.state[7] > ts.t_supply

    def reward(self, ts, hp):
        cost = 0
        reward = 0
        if self.is_hot():
            reward -= self.state[4] * (ts.t_supply - self.state[7]) * p_water * c_water
            cost += self.state[5] * (ts.t_supply - self.state[7]) * p_water * c_water
        else:
            reward -= self.state[4] * (self.state[7] - ts.t_supply) * p_water * c_water
            cost += self.state[5] * (self.state[7] - ts.t_supply) * p_water * c_water

        reward += self.state[3] * hp.rate_max * self.cop_calc(ts)
        cost += self.state[3] * hp.rate_max / self.cop_calc(ts)

        reward -= self.state[8] * self.state[3]
        cost += self.state[8] * self.state[3]

        return reward - cost
