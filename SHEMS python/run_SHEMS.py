import main
import numpy as np
import pandas as pd
from SHEMS_optimizer_cost import SHEMS_optimizer

def set_SHEMS_parameters(h_start, h_end, h_predict, h_control, rolling_flag, case=1, costfactor=1.0, outputflag=0,
                          season="all", run="all", price="fix"):
    # Initialize technical setup
    m = main.Model_SHEMS(h_start, h_end, h_predict, h_control, 60, rolling_flag, "Cbc", 0.05, outputflag, -1,
                    season, run, price)
    # PV(eta)
    pv = main.PV(0.95)
    # HeatPump(eta, rate_max)
    hp = main.HeatPump(1.0, 3.0)

    if case == 1:  # base case
        # Battery(eta, soc_min, soc_max, rate_max, loss)
        b = main.Battery(0.95, 0.0, 13.5, 3.3, 0.00003)
        # SHEMS(costfactor, p_buy, p_sell, soc_b, soc_fh, soc_hw, h_start)
        sh = main.SHEMS(costfactor, 0.3, 0.1, 13.5, 22.0, 180.0, h_start)
        # ThermalStorage(eta, volume, loss, t_supply, soc_min, soc_max)
        fh = main.ThermalStorage(1.0, 10.0, 0.045, 30.0, 20.0, 22.0)
        hw = main.ThermalStorage(1.0, 180.0, 0.035, 45.0, 20.0, 180.0)
    elif case == 2:  # no battery
        # set soc_max, rate_max to zero for no battery
        b = main.Battery(0.95, 0.0, 0.0, 0.0, 0.00003)
        # soc_b zero for no battery
        sh = main.SHEMS(costfactor, 0.3, 0.1, 0.0, 22.0, 180.0, h_start)
        # ThermalStorage(eta, volume, loss, t_supply, soc_min, soc_max)
        fh = main.ThermalStorage(1.0, 10.0, 0.045, 30.0, 20.0, 22.0)
        hw = main.ThermalStorage(1.0, 180.0, 0.035, 45.0, 20.0, 180.0)
    elif case == 3:  # no grid feed-in compensation
        # Battery(eta, soc_min, soc_max, rate_max, loss)
        b = main.Battery(0.95, 0.0, 13.5, 3.3, 0.00003)
        # set p_sell to zero for no feedin tariff
        sh = main.SHEMS(costfactor, 0.3, 0.0, 13.5, 22.0, 180.0, h_start)
        # ThermalStorage(eta, volume, loss, t_supply, soc_min, soc_max)
        fh = main.ThermalStorage(1.0, 10.0, 0.045, 30.0, 20.0, 22.0)
        hw = main.ThermalStorage(1.0, 180.0, 0.035, 45.0, 20.0, 180.0)
    elif case == 4:  # no battery and no grid feed-in compensation
        # Battery(eta, soc_min, soc_max, rate_max, loss)
        b = main.Battery(0.95, 0.0, 0.0, 0.0, 0.00003)
        # set p_sell to zero for no feedin tariff
        sh = main.SHEMS(costfactor, 0.3, 0.0, 0.0, 22.0, 180.0, h_start)
        # ThermalStorage(eta, volume, loss, t_supply, soc_min, soc_max)
        fh = main.ThermalStorage(1.0, 10.0, 0.045, 30.0, 20.0, 22.0)
        hw = main.ThermalStorage(1.0, 180.0, 0.035, 45.0, 20.0, 180.0)
    elif case == 5:  # RL case study
        # Battery(eta, soc_min, soc_max, rate_max, loss)
        b = main.Battery(0.98, 0.0, 10.0, 4.6, 0.00003)
        # ThermalStorage(eta, volume, loss, t_supply, soc_min, soc_max)
        fh = main.ThermalStorage(1.0, 10.0, 0.045, 30.0, 19.0, 24.0)
        hw = main.ThermalStorage(1.0, 200.0, 0.035, 45.0, 20.0, 180.0)
        # SHEMS(costfactor, p_buy, p_sell, soc_b, soc_fh, soc_hw, h_start)
        sh = main.SHEMS(costfactor, 0.3, 0.1, 0.5 * (b.soc_min + b.soc_max), 0.5 * (fh.soc_min + fh.soc_max),
                    0.5 * (hw.soc_min + hw.soc_max), h_start)

    return sh, hp, fh, hw, b, m, pv


def roll_SHEMS(h_start, h_end, h_predict, h_control, case=1, costfactor=1.0, outputflag=0):
    # Initialize technical setup
    sh, hp, fh, hw, b, m, pv = set_SHEMS_parameters(h_start, h_end, h_predict, h_control, True, costfactor, case, outputflag)

    # Initial run
    sh.soc_b, sh.soc_fh, sh.soc_hw, results_new = SHEMS_optimizer(sh, hp, fh, hw, b, m)
    results = results_new
    sh.h_start = m.h_start + m.h_control

    # Loop runs for rest of the horizon
    h = sh.h_start
    while h <= (m.h_end - m.h_predict):
        sh.h_start = h
        sh.soc_b, sh.soc_fh, sh.soc_hw, results_new = SHEMS_optimizer(sh, hp, fh, hw, b, m, pv)
        results += results_new
        h += m.h_control

    # Write to results folder
    write_to_results_file(np.hstack((results, np.ones((results.shape[0], 1)) * m.h_predict)), m, 1, case, costfactor)
    
    return None

# def write_to_results_file(results, m, objective=1, case=1, costfactor=1.0):
#     date = 211116
#     headers = ["Temp_FH", "Vol_HW", "Soc_B", "V_HW_plus", "V_HW_minus", "T_FH_plus", "T_FH_minus", "profits", "COP_FH",
#                "COP_HW", "PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR", "PV_HP", "GR_HP", "B_HP", "HP_FH", "HP_HW",
#                "month", "day", "hour", "horizon"]

#     # Create a DataFrame from the results and write it to a CSV file
#     pd.DataFrame(results, columns=headers).to_csv(
#         f"single_building/results/{date}_results_{m.h_predict}_{m.h_control}_{m.h_start}-{m.h_end}_{objective}_{case}_{costfactor}_{m.season}_{m.run}_{m.price}.csv",
#         index=False)

#     return None


def write_to_results_file(results, m, objective=1, case=1, costfactor=1.0):
    date = 211116
    headers = ["Temp_FH", "Vol_HW", "Soc_B", "V_HW_plus", "V_HW_minus", "T_FH_plus", "T_FH_minus", "profits", "COP_FH",
               "COP_HW", "PV_DE", "B_DE", "GR_DE", "PV_B", "PV_GR", "PV_HP", "GR_HP", "B_HP", "HP_FH", "HP_HW",
               "month", "day", "hour", "horizon"]

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=headers)

    # Add the 'horizon' column to the DataFrame
    #df_results['horizon'] = m.h_predict

    # Write the DataFrame to a CSV file
    df_results.to_csv(
        f"single_building/results/{date}_results_{m.h_predict}_{m.h_control}_{m.h_start}-{m.h_end}_{objective}_{case}_{costfactor}_{m.season}_{m.run}_{m.price}.csv",
        index=False)

    return None


def yearly_SHEMS(h_start=0, objective=1, case=1, costfactor=1.0, outputflag=1,
                 bc_violations=79, season="all", run="all", price="fix"):
    # Initialize technical setup according to case
    sh, hp, fh, hw, b, m, pv = set_SHEMS_parameters(
        h_start, main.H_LENGTH[season, run], (main.H_LENGTH[season, run]-h_start),
        (main.H_LENGTH[season, run]-h_start), False, case, costfactor, outputflag,
        season=season, run=run, price=price)
    #print("SHEMS parameters: ", sh, hp, fh, hw, b, m, pv)

    # Perform optimization based on the objective
    if objective == 1:  # minimize costs (default)
        results = SHEMS_optimizer(sh, hp, fh, hw, b, m, pv)
    # elif objective == 2:  # maximize self-consumption
    #     results = SHEMS_optimizer_seco(sh, hp, fh, hw, b, m, bc_violations)
    # elif objective == 3:  # maximize self-sufficiency
    #     results = SHEMS_optimizer_sesu(sh, hp, fh, hw, b, m, bc_violations)
    
    
    import numpy as np
    
    # Convert results to a NumPy array
    results_array = np.array(results)
    
    # Use the shape attribute of the NumPy array
    num_rows, num_cols = results_array.shape
    
    # Then you can use num_rows to access the number of rows in the array
    write_to_results_file(np.hstack((results_array, np.ones((num_rows, 1)) * m.h_predict)), m, objective, case, costfactor)



    # Write results to a file
    #write_to_results_file(np.hstack((results, np.ones((results.shape[0], 1)) * m.h_predict)), m, objective, case, costfactor)

    return None

# %%

yearly_SHEMS(0, 1, 5, 1.0, 1, season="all", run="all", price="fix")

#=
# # Calling model runs:
# yearly_SHEMS(1, 1, 5, 1.0, 1, season="all", run="eval", price="fix")
# yearly_SHEMS(1, 1, 5, 1.0, 1, run="test")
# yearly_SHEMS(1, 1, 5, 1.0, 1, season="summer", run="eval", price="fix")
# yearly_SHEMS(1, 1, 5, 1.0, 1, season="winter", run="eval", price="fix")
# =#