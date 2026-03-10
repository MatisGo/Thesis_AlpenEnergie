"""
Variables
=========
Defines all decision variables for the optimisation model.

Variable          Unit          Description
─────────────────────────────────────────────────────────────────────
R[r, t]           mm            Reservoir water level at timestep t
Q[i, t]           mm/step       Water discharged through turbine i at t
P[i, t]           kWh/step      Electricity produced by turbine i at t
SpotTrade[t]      kWh/step      Net spot market position:
                                  > 0  →  selling electricity (revenue)
                                  < 0  →  buying electricity  (cost)
IntradayBuy[t]    kWh/step      Electricity bought on the intraday market

SpotTrade is a FREE variable (positive or negative).
All other variables are non-negative.
"""

from pyomo.environ import Var, NonNegativeReals, Reals


def add_variables(model, sets):
    """
    Add all decision variables to the Pyomo model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
    sets  : dict   output of get_sets()
    """

    T = sets['T']   # Time steps
    R = sets['R']   # Reservoirs
    I = sets['I']   # Turbines

    # -----------------------------------------------------------------------
    # STATE VARIABLE: Reservoir water level  [mm]
    # -----------------------------------------------------------------------
    # R[r, t] is what the reservoir looks like at each point in time.
    # It is determined by the balance equation (inflow - discharge).
    model.R = Var(R, T,
                  domain=NonNegativeReals,
                  doc='Reservoir water level [mm]')

    # -----------------------------------------------------------------------
    # DECISION VARIABLE: Water discharge through each turbine  [mm/step]
    # -----------------------------------------------------------------------
    # Q[i, t] is the main operational decision: how much water to release
    # through each turbine at each timestep.
    # More discharge → more electricity, but empties the reservoir faster.
    model.Q = Var(I, T,
                  domain=NonNegativeReals,
                  doc='Water discharged through turbine i at timestep t [mm/step]')

    # -----------------------------------------------------------------------
    # DERIVED VARIABLE: Electricity produced  [kWh/step]
    # -----------------------------------------------------------------------
    # P[i, t] is linked to Q[i, t] via the turbine conversion factor:
    #   P[i, t] = K[i] * Q[i, t]
    # It is a separate variable (not just K*Q) so it appears cleanly
    # in the electricity balance constraint.
    model.P = Var(I, T,
                  domain=NonNegativeReals,
                  doc='Electricity produced by turbine i at timestep t [kWh/step]')

    # -----------------------------------------------------------------------
    # MARKET VARIABLES  [kWh/step]
    # -----------------------------------------------------------------------
    # SpotTrade is a single FREE variable combining buy and sell:
    #   SpotTrade[t] > 0  →  net seller   (revenue: reduces cost)
    #   SpotTrade[t] < 0  →  net buyer    (cost: increases total cost)
    #   SpotTrade[t] = 0  →  not on market
    #
    # This is simpler than having two separate SpotBuy / SpotSell variables
    # and avoids the model "buying and selling at the same time".

    model.SpotTrade = Var(T,
                          domain=Reals,
                          doc='Net spot market position [kWh/step]  (+sell / -buy)')

    model.IntradayBuy = Var(T,
                            domain=NonNegativeReals,
                            doc='Electricity bought on intraday market [kWh/step]')

    print(f"  Variables added: R, Q, P, SpotTrade, IntradayBuy")
