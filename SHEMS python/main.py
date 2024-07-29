class PV:
    def __init__(self, eta):
        self.eta = eta  # Efficiency of the PV

class HeatPump:
    def __init__(self, eta, rate_max):
        self.eta = eta  # Efficiency of the heat pump
        self.rate_max = rate_max  # Maximum rate of the heat pump

class ThermalStorage:
    def __init__(self, eta, volume, loss, t_supply, soc_min, soc_max):
        self.eta = eta  # Efficiency of the thermal storage
        self.volume = volume  # Volume of the thermal storage
        self.loss = loss  # Loss factor of the thermal storage
        self.t_supply = t_supply  # Supply temperature of the thermal storage
        self.soc_min = soc_min  # Minimum state of charge of the thermal storage
        self.soc_max = soc_max  # Maximum state of charge of the thermal storage

class Battery:
    def __init__(self, eta, soc_min, soc_max, rate_max, loss):
        self.eta = eta  # Efficiency of the battery
        self.soc_min = soc_min  # Minimum state of charge of the battery
        self.soc_max = soc_max  # Maximum state of charge of the battery
        self.rate_max = rate_max  # Maximum charging/discharging rate of the battery
        self.loss = loss  # Loss factor of the battery

class EV:
    def __init__(self, soc_min, soc_max, rate_max):
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.rate_max = rate_max

class SHEMS:
    def __init__(self, costfactor, p_buy, p_sell, SOC_b, SOC_ev, h_start):
        self.costfactor = costfactor  # Cost factor of the SHEMS
        self.p_buy = p_buy  # Buying price of electricity
        self.p_sell = p_sell  # Selling price of electricity
        self.SOC_b = SOC_b  # State of charge of the battery
        self.SOC_ev = SOC_ev  # 
        self.h_start = h_start  # Start hour of the SHEMS

class Model_SHEMS:
    def __init__(self, h_start, h_end, h_predict, h_control, big, rolling_flag, solver, mip_gap,
                 output_flag, presolve_flag, season, run, price, chargerID):
        self.h_start = h_start  # Start hour of the SHEMS model
        self.h_end = h_end  # End hour of the SHEMS model
        self.h_predict = h_predict  # Prediction horizon of the SHEMS model
        self.h_control = h_control  # Control horizon of the SHEMS model
        self.big = big  # Big parameter of the SHEMS model
        self.rolling_flag = rolling_flag  # Rolling flag of the SHEMS model
        self.solver = solver  # Solver used in the SHEMS model
        self.mip_gap = mip_gap  # MIP gap of the SHEMS model
        self.output_flag = output_flag  # Output flag of the SHEMS model
        self.presolve_flag = presolve_flag  # Presolve flag of the SHEMS model
        self.season = season  # Season of the SHEMS model
        self.run = run  # Run type of the SHEMS model
        self.price = price  # Price type of the SHEMS model
        self.chargerID = chargerID
        
H_LENGTH = {
    ("all", "all"): 8760,
    ("summer", "eval"): 360, 
    ("summer", "test"): 768,
    ("winter", "eval"): 360, 
    ("winter", "test"): 720,
    ("both", "eval"): 720,   
    ("both", "test"): 1488,
    ("all", "eval"): 1439,   
    ("all", "test"): 2999
}

