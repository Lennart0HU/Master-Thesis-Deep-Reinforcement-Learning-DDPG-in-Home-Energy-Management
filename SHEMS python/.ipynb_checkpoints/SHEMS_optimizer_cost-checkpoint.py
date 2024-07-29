import numpy as np

def COPcalc(ts, t_outside):
    # Calculate coefficients of performance for every time period (1:h_predict)
    cop = np.maximum((5.8 * np.ones_like(t_outside)) - (1.0 / 14) * np.abs((ts.t_supply * np.ones_like(t_outside)) - t_outside), 0)
    return cop

import pandas as pd
from numpy import maximum, abs, sum
import gurobipy
#from pulp import LpMaximize, LpProblem, LpVariable, LpConstraint, LpStatus, value
import pulp as pl


def SHEMS_optimizer(sh, hp, fh, hw, b, m, pv):
    flows = ['PV_DE', 'B_DE', 'GR_DE', 'PV_B', 'PV_GR', 'PV_HP', 'GR_HP', 'B_HP', 'HP_FH', 'HP_HW']
    p_concr = 2400.0  # kg/m^3
    c_concr = 1.0     # kJ/(kg*°C)
    p_water = 997.0    # kg/m^3
    c_water = 4.184    # kJ/(kg*°C)
    
    #print("h_predict: ", m.h_predict)
    
    # Input data
    df = pd.read_csv(f"single_building/data/{m.season}_{m.run}_{m.price}.csv")
    h_last = sh.h_start + m.h_predict   # optimization horizon
    print(df.iloc[:, 6].head())
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
    model = pl.LpProblem(name="SHEMS_optimizer", sense=pl.LpMaximize)

    # Define variables
    X = pl.LpVariable.dicts("X", ((i, j) for i in range(0, m.h_predict) for j in range(0, len(flows))), lowBound=0)
    SOC_b = pl.LpVariable.dicts("SOC_b", (i for i in range(0, m.h_predict + 1)), lowBound=0)
    T_fh = pl.LpVariable.dicts("T_fh", (i for i in range(0, m.h_predict + 1)), lowBound=0)
    V_hw = pl.LpVariable.dicts("V_hw", (i for i in range(0, m.h_predict + 1)), lowBound=0)
    Mod_fh = pl.LpVariable.dicts("Mod_fh", (i for i in range(0, m.h_predict)), lowBound=0)
    Mod_hw = pl.LpVariable.dicts("Mod_hw", (i for i in range(0, m.h_predict)), lowBound=0)
    T_fh_plus = pl.LpVariable.dicts("T_fh_plus", (i for i in range(0, m.h_predict)), lowBound=0)
    T_fh_minus = pl.LpVariable.dicts("T_fh_minus", (i for i in range(0, m.h_predict)), lowBound=0)
    V_hw_plus = pl.LpVariable.dicts("V_hw_plus", (i for i in range(0, m.h_predict)), lowBound=0)
    V_hw_minus = pl.LpVariable.dicts("V_hw_minus", (i for i in range(0, m.h_predict)), lowBound=0)
    HP_switch = pl.LpVariable.dicts("HP_switch", (i for i in range(0, m.h_predict)), cat='Binary')
    Hot = pl.LpVariable.dicts("Hot", (i for i in range(0, m.h_predict)), cat='Binary')

    # Fix start SoCs
    for var in [SOC_b[0], T_fh[0], V_hw[0]]:
        var.varValue = sh.SOC_b if var == SOC_b[0] else sh.T_fh if var == T_fh[0] else sh.V_hw
    
    print("sh.p_sell: ", sh.p_sell, "sh.p_buy: ", sh.p_buy)
    for key, value in list(X.items())[:5]:
        print(f'{key}: {value}')
    print("printing X(1,3 5 and 7): ", X[(0,0)], X[(0,2)], X[(0,4)], X[(0,6)])
    
    # Objective function
    model += sum((sh.p_sell * X[(h,4)]) - sum(sh.p_buy * X[(h,i)] for i in [2, 6]) -
                 (sh.costfactor * (T_fh_plus[h] + T_fh_minus[h] + V_hw_plus[h] + V_hw_minus[h])) for h in range(0, m.h_predict)), "Objective"

    # Electricity demand, generation
    for h in range(0, m.h_predict):
        
       # model += sum(X[(h,i)] for i in range(1, 4)) == d_e[h - 1], f"Electricity_demand_{h}"
        model += sum(X[(h,i)] for i in range(0, 3)) == d_e[h], f"Electricity_demand_{h}"

        model += sum(X[(h,i)] for i in [0, 3, 4, 5]) == g_e[h] * pv.eta, f"PV_generation_{h}"

    # Battery
    for h in range(0, m.h_predict):
        model += SOC_b[h + 1] == ((1 - b.loss) * SOC_b[h]) + (b.eta * X[(h,3)]) - sum((1.0 / b.eta) * X[(h,i)] for i in [1, 7]), f"Battery_SOC_{h}"
        model += b.soc_min <= SOC_b[h] <= b.soc_max, f"Battery_limits_{h}"
        model += sum(X[(h,i)] for i in [1, 3, 7]) <= b.rate_max, f"Battery_charging_discharging_{h}"

    # Heat pump
    for h in range(0, m.h_predict):
        model += sum(X[(h,i)] for i in range(8, 10)) == sum(X[(h,i)] for i in range(5, 8)), f"Heat_pump_power_{h}"
        model += X[(h,8)] == Mod_fh[h] * hp.rate_max, f"Heating_energy_FH_{h}"
        model += X[(h,9)] == Mod_hw[h] * hp.rate_max, f"Heating_energy_HW_{h}"
        model += Mod_fh[h] <= 1 - HP_switch[h], f"Heat_pump_switch_FH_HW_{h}"
        model += Mod_hw[h] <= HP_switch[h], f"Heat_pump_switch_HW_FH_{h}"

    # Floor heating
    for h in range(0, m.h_predict):
        model += T_fh[h + 1] == T_fh[h] + (60 * 60) / (p_concr * fh.volume * c_concr) * (
                    (cop_fh[h] * X[(h,8)]) - d_fh[h] - ((1 - Hot[h]) * fh.loss) + (Hot[h] * fh.loss)), f"Floor_heating_temperature_{h}"
        model += T_fh[h] - ((1 - Hot[h]) * m.big) <= t_outside[h], f"Floor_heating_temp_binary_{h}"
        model += t_outside[h] - (Hot[h] * m.big) <= T_fh[h], f"Floor_heating_temp_outside_{h}"
        model += T_fh[h] <= fh.soc_max + T_fh_plus[h], f"Floor_heating_temp_max_{h}"
        model += fh.soc_min - T_fh_minus[h] <= T_fh[h], f"Floor_heating_temp_min_{h}"

    # Hot water
    for h in range(0, m.h_predict):
        model += V_hw[h + 1] == V_hw[h] + (60 * 60) / ((p_water * hw.t_supply * c_water) / 1000) * (
                    (cop_hw[h] * X[(h,9)]) - d_hw[h] - hw.loss), f"Hot_water_volume_{h}"
        model += V_hw[h] <= hw.soc_max + V_hw_plus[h], f"Hot_water_volume_max_{h}"
        model += hw.soc_min - V_hw_minus[h] <= V_hw[h], f"Hot_water_volume_min_{h}"

    # Solve the problem
    #model.solve(solver=pl.GUROBI_CMD())
    model.solve(solver=pl.PULP_CBC_CMD())


    # Collect returns
    profits = sum((sh.p_sell * pl.value(X[(h,4)])) - sh.p_buy * sum(pl.value(X[(h,i)]) for i in [2, 6]) for h in range(0, m.h_control))
    results = [[pl.value(T_fh[h]), pl.value(V_hw[h]), pl.value(SOC_b[h]), pl.value(V_hw_plus[h]), pl.value(V_hw_minus[h]),
                pl.value(T_fh_plus[h]), pl.value(T_fh_minus[h]), profits, cop_fh[h], cop_hw[h], (pl.value(X[(h,i)]) for i in range(0, len(flows))),
                df.iloc[h, [5, 6, 7]].values.tolist()] for h in range(0, m.h_control)
               ]
    for h in range (0,m.h_control):
        results_h1 = [pl.value(T_fh[h]), pl.value(V_hw[h]), pl.value(SOC_b[h]), pl.value(V_hw_plus[h]), pl.value(V_hw_minus[h]),
                pl.value(T_fh_plus[h]), pl.value(T_fh_minus[h]), profits, cop_fh[h], cop_hw[h]]
        results_h2 = [pl.value(X[(h,i)]) for i in range(0, len(flows))]
        results_h3 = df.iloc[h, [5, 6, 7]].values.tolist()
        results[h] = results_h1 + results_h2 + results_h3
    print(results)
    if m.rolling_flag == 1:
        return [pl.value(SOC_b[m.h_control + 1]), pl.value(T_fh[m.h_control + 1]), pl.value(V_hw[m.h_control + 1]), results]
    else:
        return results
