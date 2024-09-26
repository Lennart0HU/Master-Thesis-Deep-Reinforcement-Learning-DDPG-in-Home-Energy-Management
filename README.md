# Efficient Energy Management for Prosumer Households

## Abstract
Efficient energy management on the level of prosumer households is key to alleviating grid stress in an energy transition marked by electrification, a switch to renewable generation, and decentralization. With millions of new private EV charging stations estimated to be installed by 2030, smart EV charging is one strategy to balance production and consumption locally, as well as regionally. Model-free deep reinforcement learning has emerged as a promising method to deal with the dynamic and uncertain environment of home energy management optimization problems. In this thesis, a deep deterministic policy gradient algorithm is applied to a new real-life dataset of prosumer households with PV, battery, and EV chargers and benchmarked against a power-mode approach (REPHRASE) commonly used in the markets today, as well as the optimal solution under perfect information with model predictive control.

## Model Set-Up
The code for the Deep Reinforcement Learning model was adapted from Langer, 2022 in the programming language Julia. Python was used for the descriptive analysis and most of the pre-processing of the data. The code for the Model Predictive Control Benchmark, where the approach is based on Langer, 2020, was also written in Python. The linear programming model was written using the `pulp` package (ADD SOURCE) and solved using the Gurobi solver under an academic license (LINK?).

## Data Availability
Please note that the company data used in this thesis is not available in this repository.

## Repository Structure
- `data/`: ...
- `src/`: ...
- `notebooks/`: ...
- `results/`: ...

## How to Run
Scheduler...

## References
- Langer, 2022:
- Langer, 2020:

