import numpy as np

def COPcalc(ts, t_outside):
    # Calculate coefficients of performance for every time period (1:h_predict)
    cop = np.maximum((5.8 * np.ones_like(t_outside)) - (1.0 / 14) * np.abs((ts.t_supply * np.ones_like(t_outside)) - t_outside), 0)
    return cop

import pandas as pd
from numpy import maximum, abs, sum
import gurobipy
from pulp import LpMaximize, LpProblem, LpVariable, LpConstraint, LpStatus, value
import pulp.solvers as solvers

def SHEMS_optimizer(sh, hp, fh, hw, b, m, pv):
    flows = ['PV_DE', 'B_DE', 'GR_DE', 'PV_B', 'PV_GR', 'PV_HP', 'GR_HP', 'B_HP', 'HP_FH', 'HP_HW']
    p_concr = 2400.0  # kg/m^3
    c_concr = 1.0     # kJ/(kg*°C)
    p_water = 997.0    # kg/m^3
    c_water = 4.184    # kJ/(kg*°C)

    # Input data
    df = pd.read_csv(f"single_building/data/{m.season}_{m.run}_{m.price}.csv")
    h_last = sh.h_start + m.h_predict - 1  # optimization horizon

    # read input data
    d_e = df.loc[sh.h_start:h_last, 'electkwh'].values
    d_fh = df.loc[sh.h_start:h_last, 'heatingkwh'].values
    d_hw = df.loc[sh.h_start:h_last, 'hotwaterkwh'].values
    g_e = df.loc[sh.h_start:h_last, 'PV_generation'].values
    t_outside = df.loc[sh.h_start:h_last, 'Temperature'].values

    # Calculate coefficients of performance for every time period (1:h_predict)
    cop_fh = COPcalc(fh, t_outside)
    cop_hw = COPcalc(hw, t_outside)

    # Model start
    model = LpProblem(name="SHEMS_optimizer", sense=LpMaximize)

    # Define variables
    X = LpVariable.dicts("X", ((i, j) for i in range(1, m.h_predict + 1) for j in range(1, len(flows) + 1)), lowBound=0)
    SOC_b = LpVariable.dicts("SOC_b", (i for i in range(1, m.h_predict + 2)), lowBound=0)
    T_fh = LpVariable.dicts("T_fh", (i for i in range(1, m.h_predict + 2)), lowBound=0)
    V_hw = LpVariable.dicts("V_hw", (i for i in range(1, m.h_predict + 2)), lowBound=0)
    Mod_fh = LpVariable.dicts("Mod_fh", (i for i in range(1, m.h_predict + 1)), lowBound=0)
    Mod_hw = LpVariable.dicts("Mod_hw", (i for i in range(1, m.h_predict + 1)), lowBound=0)
    T_fh_plus = LpVariable.dicts("T_fh_plus", (i for i in range(1, m.h_predict + 1)), lowBound=0)
    T_fh_minus = LpVariable.dicts("T_fh_minus", (i for i in range(1, m.h_predict + 1)), lowBound=0)
    V_hw_plus = LpVariable.dicts("V_hw_plus", (i for i in range(1, m.h_predict + 1)), lowBound=0)
    V_hw_minus = LpVariable.dicts("V_hw_minus", (i for i in range(1, m.h_predict + 1)), lowBound=0)
    HP_switch = LpVariable.dicts("HP_switch", (i for i in range(1, m.h_predict + 1)), cat='Binary')
    Hot = LpVariable.dicts("Hot", (i for i in range(1, m.h_predict + 1)), cat='Binary')

    # Fix start SoCs
    for var in [SOC_b[1], T_fh[1], V_hw[1]]:
        var.varValue = sh.SOC_b if var == SOC_b[1] else sh.T_fh if var == T_fh[1] else sh.V_hw

    # Objective function
    model += sum((sh.p_sell * X[h][5]) - sum(sh.p_buy * X[h][i] for i in [3, 7]) -
                 (sh.costfactor * (T_fh_plus[h] + T_fh_minus[h] + V_hw_plus[h] + V_hw_minus[h])) for h in range(1, m.h_predict + 1)), "Objective"

    # Electricity demand, generation
    for h in range(1, m.h_predict + 1):
        model += sum(X[h][i] for i in range(1, 4)) == d_e[h - 1], f"Electricity_demand_{h}"
        model += sum(X[h][i] for i in [1, 4, 5, 6]) == g_e[h - 1] * pv.eta, f"PV_generation_{h}"

    # Battery
    for h in range(1, m.h_predict + 1):
        model += SOC_b[h + 1] == ((1 - b.loss) * SOC_b[h]) + (b.eta * X[h][4]) - sum((1.0 / b.eta) * X[h][i] for i in [2, 8]), f"Battery_SOC_{h}"
        model += b.soc_min <= SOC_b[h] <= b.soc_max, f"Battery_limits_{h}"
        model += sum(X[h][i] for i in [2, 4, 8]) <= b.rate_max, f"Battery_charging_discharging_{h}"

    # Heat pump
    for h in range(1, m.h_predict + 1):
        model += sum(X[h][i] for i in range(9, 11)) == sum(X[h][i] for i in range(6, 9)), f"Heat_pump_power_{h}"
        model += X[h][9] == Mod_fh[h] * hp.rate_max, f"Heating_energy_FH_{h}"
        model += X[h][10] == Mod_hw[h] * hp.rate_max, f"Heating_energy_HW_{h}"
        model += Mod_fh[h] <= 1 - HP_switch[h], f"Heat_pump_switch_FH_HW_{h}"
        model += Mod_hw[h] <= HP_switch[h], f"Heat_pump_switch_HW_FH_{h}"

    # Floor heating
    for h in range(1, m.h_predict + 1):
        model += T_fh[h + 1] == T_fh[h] + (60 * 60) / (p_concr * fh.volume * c_concr) * (
                    (cop_fh[h - 1] * X[h][9]) - d_fh[h - 1] - ((1 - Hot[h]) * fh.loss) + (Hot[h] * fh.loss)), f"Floor_heating_temperature_{h}"
        model += T_fh[h] - ((1 - Hot[h]) * m.big) <= t_outside[h - 1], f"Floor_heating_temp_binary_{h}"
        model += t_outside[h - 1] - (Hot[h] * m.big) <= T_fh[h], f"Floor_heating_temp_outside_{h}"
        model += T_fh[h] <= fh.soc_max + T_fh_plus[h], f"Floor_heating_temp_max_{h}"
        model += fh.soc_min - T_fh_minus[h] <= T_fh[h], f"Floor_heating_temp_min_{h}"

    # Hot water
    for h in range(1, m.h_predict + 1):
        model += V_hw[h + 1] == V_hw[h] + (60 * 60) / ((p_water * hw.t_supply * c_water) / 1000) * (
                    (cop_hw[h - 1] * X[h][10]) - d_hw[h - 1] - hw.loss), f"Hot_water_volume_{h}"
        model += V_hw[h] <= hw.soc_max + V_hw_plus[h], f"Hot_water_volume_max_{h}"
        model += hw.soc_min - V_hw_minus[h] <= V_hw[h], f"Hot_water_volume_min_{h}"

    # Solve the problem
    model.solve(solver=solvers.GUROBI_CMD())

    # Collect returns
    profits = sum((sh.p_sell * value(X[h][5])) - sh.p_buy * sum(value(X[h][i]) for i in [3, 7]) for h in range(1, m.h_control + 1))
    results = [[value(T_fh[h]), value(V_hw[h]), value(SOC_b[h]), value(V_hw_plus[h]), value(V_hw_minus[h]),
                value(T_fh_plus[h]), value(T_fh_minus[h]), profits, cop_fh[h - 1], cop_hw[h - 1], [value(X[h][i]) for i in range(1, len(flows) + 1)],
                df.loc[sh.h_start:(sh.h_start + m.h_control - 1), [6, 7, 8]].values.tolist()] for h in range(1, m.h_control + 1)]

    if m.rolling_flag == 1:
        return [value(SOC_b[m.h_control + 1]), value(T_fh[m.h_control + 1]), value(V_hw[m.h_control + 1]), results]
    else:
        return results
