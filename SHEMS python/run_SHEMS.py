import main
import numpy as np
import pandas as pd
from SHEMS_optimizer_cost import SHEMS_optimizer


def set_SHEMS_parameters(h_start, h_end, h_predict, h_control, rolling_flag, case=1, costfactor=10.0, outputflag=0,
                          season="all", run="all", price="fix", chargerID="Charger01"):
    
    capacities = {
    'Charger01': (48.250, 7.5 * 0.9, 3.3),
    'Charger02': (36.271, 10 * 0.9, 3.3),
    'Charger03': (45.508, 10 * 0.9, 3.3),
    'Charger04': (78.993, 11 * 0.9, 4.6),
    'Charger05': (37.207, 10 * 0.9, 4.6),
    'Charger06': (35.816, 15 * 0.9, 4.6),
    'Charger07': (36.521, 12 * 0.9, 3.3),
    'Charger08': (45.728, 10 * 0.9, 3.3),
    'Charger09': (21.935, 7.5 * 0.9, 3.3),
    'Charger98': (35.816, 7.5 * 0.9, 3.3)
}

    # Get capacities from dictionary
    ev_capacity, battery_capacity, battery_power = capacities[chargerID]

    # Initialize technical setup
    m = main.Model_SHEMS(h_start, h_end, h_predict, h_control, 60, rolling_flag, "Cbc", 0.05, outputflag, -1,
                    season, run, price, chargerID)

    # Battery(eta, soc_min, soc_max, rate_max, loss)
    b = main.Battery(0.95, 0.0, battery_capacity, battery_power, 0.00003)
    ev = main.EV(0.0, ev_capacity, 11.0)
    # SHEMS(costfactor, p_buy, p_sell, soc_b, soc_ev,  h_start)
    sh = main.SHEMS(costfactor, 0.4, 0.2*0.4, 0.5 * (b.soc_min + b.soc_max), 1,
                h_start)
    print(f'ChargerID: ', chargerID)
    print('initialized battery soc_b: ', sh.SOC_b)

    return sh, ev, b, m



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


def write_to_results_file(results, m, objective=1, case=1, costfactor=10.0):
    date = 260724
    headers = ["Soc_B", "Soc_Ev", "C_EV", "profits", 'PV_DE', 'B_DE', 'GR_DE', 'PV_B', 'PV_GR', 'PV_EV', 'GR_EV', 'B_EV', 'EXT_EV',
               "month", "day", "hour", "horizon"]

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=headers)

    # Add the 'horizon' column to the DataFrame
    df_results['horizon'] = m.h_predict

    # Write the DataFrame to a CSV file
    df_results.to_csv(
        f"single_building/results/{date}_results_{m.h_predict}_{m.h_control}_{m.h_start}-{m.h_end}_{objective}_{case}_{costfactor}_{m.season}_{m.run}_{m.price}_{m.chargerID}.csv",
        index=False)

    return None


def yearly_SHEMS(h_start=0, objective=1, case=1, costfactor=10.0, outputflag=1,
                 season="all", run="all", price="fix", chargerID="Charger01"):
    print(f"Running SHEMS optimization for case {case}, costfactor {costfactor}, run {run}, chargerID {chargerID}.")
    # Initialize technical setup according to case
    sh, ev, b, m = set_SHEMS_parameters(
        h_start, main.H_LENGTH[season, run], (main.H_LENGTH[season, run]-h_start),
        (main.H_LENGTH[season, run]-h_start), False, case, costfactor, outputflag,
        season=season, run=run, price=price, chargerID=chargerID)
    #print("SHEMS parameters: ", sh, hp, fh, hw, b, m, pv)

    # Perform optimization based on the objective
    if objective == 1:  # minimize costs (default)
        results = SHEMS_optimizer(sh, ev, b, m)
    # elif objective == 2:  # maximize self-consumption
    #     results = SHEMS_optimizer_seco(sh, hp, fh, hw, b, m, bc_violations)
    # elif objective == 3:  # maximize self-sufficiency
    #     results = SHEMS_optimizer_sesu(sh, hp, fh, hw, b, m, bc_violations)
    
        
    results_array = np.array(results)
    num_rows, num_cols = results_array.shape
    print(f"Optimization for case {case}, costfactor {costfactor}, run {run}, chargerID {chargerID} done!")
    print(f"Results non array: {results[0][3]}")

    write_to_results_file(np.hstack((results_array, np.ones((num_rows, 1)) * m.h_predict)), m, objective, case, costfactor)



    # Write results to a file
    #write_to_results_file(np.hstack((results, np.ones((results.shape[0], 1)) * m.h_predict)), m, objective, case, costfactor)

    return results


#yearly_SHEMS(0, 1, 5, 10.0, 1, season="all", run="test", price="fix", chargerID='Charger04')


chargerIDs = ['Charger01', 'Charger02', 'Charger03', 'Charger04', 'Charger05', 'Charger06', 'Charger07', 'Charger08', 'Charger09', 'Charger98']

profits = [('profits', '', 0)]

for chargerID in chargerIDs:
    results = yearly_SHEMS(0, 1, 5, 10.0, 1, season="all", run="train", price="fix", chargerID=chargerID)
    profits.append((chargerID, 'train', results[0][3]))

    results = yearly_SHEMS(0, 1, 5, 10.0, 1, season="all", run="eval", price="fix", chargerID=chargerID)
    profits.append((chargerID, 'eval', results[0][3]))

    results = yearly_SHEMS(0, 1, 5, 10.0, 1, season="all", run="test", price="fix", chargerID=chargerID)
    profits.append((chargerID, 'test', results[0][3]))

print(profits)
df_profits = pd.DataFrame(profits, columns=['ChargerID', 'Run', 'Result'])
df_profits.to_excel(
        "single_building/results/profits.xlsx",
        index=False)

