"""
Objective Function
==================
Minimise the total electricity cost over the optimisation horizon.

Cost = Σ_t  [ - Price_spot[t]     × SpotTrade[t]
             +   Price_intraday[t] × IntradayBuy[t]  ]

SpotTrade sign convention:
  SpotTrade > 0  →  selling  →  negative cost (revenue)   → model wants this HIGH
  SpotTrade < 0  →  buying   →  positive cost             → model wants this LOW

So minimising  -Price_spot × SpotTrade  makes the model:
  → sell as much as possible when prices are high
  → avoid buying (or buy as little as possible)
"""

from pyomo.environ import Objective, minimize


def add_objective(model, sets, params):
    """
    Add the cost minimisation objective to the Pyomo model.

    Parameters
    ----------
    model  : pyomo.environ.ConcreteModel
    sets   : dict   output of get_sets()
    params : dict   output of get_parameters()
    """

    T = sets['T']

    def total_cost_rule(model):
        cost = sum(
            # SpotTrade: negative when selling (revenue), positive when buying (cost)
            - params['Price_spot'][t]     * model.SpotTrade[t]
            # Intraday buy always costs money
            + params['Price_intraday'][t] * model.IntradayBuy[t]
            for t in T
        )
        return cost

    model.objective = Objective(rule=total_cost_rule, sense=minimize)

    print(f"  Objective added: minimise total electricity cost  [€]")
