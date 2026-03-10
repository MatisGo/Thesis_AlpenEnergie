"""
Constraints
===========
All physical and operational constraints of the hydro system.

Constraint                  Description
────────────────────────────────────────────────────────────────────
1. Initial reservoir level  Fix starting water level from data
2. Reservoir balance        Water level evolves: in - out = change
3. Reservoir limits         Min and max safe water level
4. Turbine discharge limits Max water flow through each turbine
5. Power production         P[i,t] = k[i] * Q[i,t]
6. Electricity balance      Production + Buy = Demand + Sell
7. Winter contract          Minimum production in winter months
"""

from pyomo.environ import Constraint
from model.sets import TURBINE_RESERVOIR


def add_constraints(model, sets, params):
    """
    Add all constraints to the Pyomo model.

    Parameters
    ----------
    model  : pyomo.environ.ConcreteModel
    sets   : dict   output of get_sets()
    params : dict   output of get_parameters()
    """

    T = sets['T']   # [0, 1, 2, ..., N-1]
    R = sets['R']   # ['Bidmi', 'Haselholz']
    I = sets['I']   # ['M1', 'M2']

    # -----------------------------------------------------------------------
    # 1. INITIAL RESERVOIR LEVEL
    # -----------------------------------------------------------------------
    # The first timestep must match the observed starting water level.
    # This anchors the model to reality.

    def initial_level_rule(model, r):
        return model.R[r, 0] == params['R0'][r]

    model.initial_level = Constraint(R, rule=initial_level_rule)

    # -----------------------------------------------------------------------
    # 2. RESERVOIR WATER BALANCE
    # -----------------------------------------------------------------------
    # For each reservoir r at each timestep t:
    #
    #   R[r, t+1] = R[r, t]  +  Inflow[r, t]  -  Q[i, t]
    #       ↑           ↑              ↑                ↑
    #   next level  current      natural water      water released
    #               level        inflow (rain)      through turbine
    #
    # The turbine that draws from reservoir r is defined in TURBINE_RESERVOIR.
    # Bidmi → M2,  Haselholz → M1

    def reservoir_balance_rule(model, r, t):
        if t == T[-1]:
            return Constraint.Skip   # No "t+1" exists at the last step

        # Find which turbine(s) draw from reservoir r
        turbines_of_r = [i for i in I if TURBINE_RESERVOIR[i] == r]

        # Total water discharged from reservoir r at time t
        total_discharge = sum(model.Q[i, t] for i in turbines_of_r)

        # Water balance equation
        return (
            model.R[r, t + 1]
            == model.R[r, t]
            + params['Inflow'][r, t]
            - total_discharge
        )

    model.reservoir_balance = Constraint(R, T, rule=reservoir_balance_rule)

    # -----------------------------------------------------------------------
    # 3. RESERVOIR STORAGE LIMITS
    # -----------------------------------------------------------------------
    # The reservoir level must stay within safe operating bounds.
    # Too high → risk of overflow.  Too low → turbines cannot operate.

    def reservoir_max_rule(model, r, t):
        return model.R[r, t] <= params['Rmax'][r]

    def reservoir_min_rule(model, r, t):
        return model.R[r, t] >= params['Rmin'][r]

    model.reservoir_max = Constraint(R, T, rule=reservoir_max_rule)
    model.reservoir_min = Constraint(R, T, rule=reservoir_min_rule)

    # -----------------------------------------------------------------------
    # 4. TURBINE DISCHARGE LIMITS
    # -----------------------------------------------------------------------
    # Each turbine has a physical maximum water flow it can handle.
    # Discharge cannot exceed this limit.

    def turbine_max_discharge_rule(model, i, t):
        return model.Q[i, t] <= params['Qmax'][i]

    model.turbine_max_discharge = Constraint(I, T, rule=turbine_max_discharge_rule)

    # -----------------------------------------------------------------------
    # 5. POWER PRODUCTION EQUATION
    # -----------------------------------------------------------------------
    # Electricity production is directly proportional to water discharged.
    # The conversion factor k is specific to each turbine:
    #
    #   P[i, t] = k[i] * Q[i, t]
    #
    # Examples:
    #   M2 (Bidmi):     P = 3.330 kWh/mm × Q_M2  [kWh/step]
    #   M1 (Haselholz): P = 0.455 kWh/mm × Q_M1  [kWh/step]

    def power_production_rule(model, i, t):
        return model.P[i, t] == params['K_turbine'][i] * model.Q[i, t]

    model.power_production = Constraint(I, T, rule=power_production_rule)

    # -----------------------------------------------------------------------
    # 6. ELECTRICITY BALANCE
    # -----------------------------------------------------------------------
    # At every timestep, supply must equal demand.
    #
    # SpotTrade is a single signed variable:
    #   SpotTrade > 0  →  selling surplus to grid   (subtract from supply side)
    #   SpotTrade < 0  →  buying from grid           (add to supply side)
    #
    # Rearranged:
    #   Production + IntradayBuy - SpotTrade = Demand
    #
    # This guarantees local demand is always satisfied.

    def electricity_balance_rule(model, t):
        total_production = sum(model.P[i, t] for i in I)

        return (
            total_production
            + model.IntradayBuy[t]
            - model.SpotTrade[t]
            == params['Demand'][t]
        )

    model.electricity_balance = Constraint(T, rule=electricity_balance_rule)

    # -----------------------------------------------------------------------
    # 7. WINTER CONTRACT CONSTRAINT
    # -----------------------------------------------------------------------
    # During winter months, a minimum electricity production is committed.
    # If hydro production falls below the contract, the model must buy
    # electricity on the market to compensate.

    def winter_contract_rule(model, t):
        if params['Contract'][t] == 0:
            return Constraint.Skip   # No contract outside winter months

        total_production = sum(model.P[i, t] for i in I)
        return total_production >= params['Contract'][t]

    model.winter_contract = Constraint(T, rule=winter_contract_rule)

    print(f"  Constraints added:")
    print(f"    - Initial reservoir levels")
    print(f"    - Reservoir water balance  (Bidmi + Haselholz)")
    print(f"    - Reservoir storage limits (min / max)")
    print(f"    - Turbine discharge limits (M1 + M2)")
    print(f"    - Power production equation  P = k * Q")
    print(f"    - Electricity balance  (production + buy = demand + sell)")
    print(f"    - Winter contract constraint")
