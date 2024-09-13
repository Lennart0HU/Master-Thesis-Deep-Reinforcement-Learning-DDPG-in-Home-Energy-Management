import numpy as np
import pandas as pd
from numpy import maximum, abs, sum
import gurobipy
#from pulp import LpMaximize, LpProblem, LpVariable, LpConstraint, LpStatus, value
import pulp as pl


def SHEMS_optimizer(sh, ev, b, m):
    flows = ['PV_DE', 'B_DE', 'GR_DE', 'PV_B', 'PV_GR', 'PV_EV', 'GR_EV', 'B_EV', 'EX_EV']    
    #print("h_predict: ", m.h_predict)
    
    # Input data
    df = pd.read_csv(f"single_building/data/{m.chargerID}_{m.season}_{m.run}_{m.price}.csv")
    h_last = sh.h_start + m.h_predict   # optimization horizon

    # read input data
    d_e = df.loc[sh.h_start:h_last, 'electkwh'].values
    c_ev = df.loc[sh.h_start:h_last, 'h_countdown'].values
    g_e = df.loc[sh.h_start:h_last, 'PV_generation'].values
    soc_ev = df.loc[sh.h_start:h_last, 'soc_ev'].values * ev.soc_max

    # Model start
    model = pl.LpProblem(name="SHEMS_optimizer", sense=pl.LpMaximize)

    # Define variables
    X = pl.LpVariable.dicts("X", ((i, j) for i in range(0, m.h_predict) for j in range(0, len(flows))), lowBound=0)
    SOC_b = pl.LpVariable.dicts("SOC_b", (i for i in range(0, m.h_predict + 1)), lowBound=0)
    SOC_ev = pl.LpVariable.dicts("SOC_ev", (i for i in range(0, m.h_predict + 1)), lowBound=0)


    # Fix start SoCs
    for var in [SOC_b[0], SOC_ev[0]]:
        var.varValue = sh.SOC_b if var == SOC_b[0] else (sh.SOC_ev * ev.soc_max)
    print("Fixed starting soc_b: ", SOC_b[0].varValue)

    if c_ev[0] > -1:    
        SOC_ev[0].varValue = soc_ev[0]    # If starting during a transaction soc = data soc (x%)
    
    print("SOC EV START: ", SOC_ev[0].varValue)
    print("C EV start: ", c_ev[0])
    print("soc ev start: ", soc_ev[0])

    # Objective function
    model += sum((sh.p_sell * X[(h,4)]) - sum(sh.p_buy * X[(h,i)] for i in [2, 6]) - sum(sh.costfactor * sh.p_buy * X[(h,8)])
              for h in range(0, m.h_predict)), "Objective"


    # Electricity demand, generation
    for h in range(0, m.h_predict):
        
       # model += sum(X[(h,i)] for i in range(1, 4)) == d_e[h - 1], f"Electricity_demand_{h}"
        model += sum(X[(h,i)] for i in range(0, 3)) == d_e[h], f"Electricity_demand_{h}"

        model += sum(X[(h,i)] for i in [0, 3, 4, 5]) == g_e[h], f"PV_generation_{h}"

    # Battery
    for h in range(0, m.h_predict):
        model += SOC_b[0] == sh.SOC_b
        model += SOC_b[h + 1] == ((1 - b.loss) * SOC_b[h]) + (b.eta * X[(h,3)]) - sum((1.0 / b.eta) * X[(h,i)] for i in [1, 7]), f"Battery_SOC_{h}"
        model += b.soc_min <= SOC_b[h] <= b.soc_max, f"Battery_limits_{h}"
        model += sum(X[(h,i)] for i in [1, 3, 7]) <= b.rate_max, f"Battery_charging_discharging_{h}"

    # EV
    for h in range(0, m.h_predict):

        model += SOC_ev[0] == soc_ev[0]

        if c_ev[h] > 0:
            model += SOC_ev[h + 1] == (SOC_ev[h] + sum(X[(h,i)] for i in range(5, 8))), f"EV_SOC_{h}"       # during transaction: next soc = current soc + charge
        elif c_ev[h] == 0: 
            model += SOC_ev[h + 1] == (SOC_ev[h] + sum(X[(h,i)] for i in range(5, 9))), f"EV_SOC_{h}"       # in last transaction step: next soc = current soc + charge + external charge
        else:
            model += SOC_ev[h] == soc_ev[h], f"EV_SOC_{h}"                                                  # outside of transaction: current soc = data soc (=100%)

            if c_ev[h + 1] > -1:                                                                                # Before start of transaction: next-soc = data next-soc (x%)
                model += SOC_ev[h + 1] == soc_ev[h + 1], f"EV_SOC_start_{h}"
    
        model += ev.soc_min <= X[(h,8)] <= ev.soc_max
        model += ev.soc_min <= SOC_ev[h] <= ev.soc_max, f"EV_limits_{h}"
        model += sum(X[(h,i)] for i in range(5, 8)) <= ev.rate_max, f"EV_charging_{h}"


    # Solve the problem
    model.solve(solver=pl.GUROBI_CMD())
    #model.solve(solver=pl.PULP_CBC_CMD())


    # Collect returns
    profits = sum((sh.p_sell * pl.value(X[(h,4)])) - sh.p_buy * sum(pl.value(X[(h,i)]) for i in [2, 6]) for h in range(0, m.h_control))
    results = [[pl.value(SOC_b[h]), pl.value(SOC_ev[h]), c_ev[h], profits, (pl.value(X[(h,i)]) for i in range(0, len(flows))),
                [df.iloc[h, i] for i in [5, 6, 7]]] for h in range(0, m.h_control)
               ]
    for h in range (0,m.h_control):
        results_h1 = [pl.value(SOC_b[h]), pl.value(SOC_ev[h]), c_ev[h], profits]
        results_h2 = [pl.value(X[(h,i)]) for i in range(0, len(flows))]
        #results_h3 = df.iloc[h, [5, 6, 7]].values.tolist()
        results_h3 = [df.iloc[h, i] for i in [5, 6, 7]]
        results[h] = results_h1 + results_h2 + results_h3

    return results

