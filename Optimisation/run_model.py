"""
Run Model
=========
Main script to build and solve the hydro-electricity optimisation model.

What this script does:
  1. Load operational data from the Excel file
  2. Build index sets (time steps, reservoirs, turbines)
  3. Build model parameters (physical constants + time-series data)
  4. Create a Pyomo LP model with variables, constraints, objective
  5. Solve the model with the chosen solver
  6. Save results to results.csv for analysis

Usage:
  python run_model.py

To change the time window or solver, edit the CONFIGURATION section below.
"""

import sys
import pandas as pd
from pyomo.environ import ConcreteModel, SolverFactory, value
from pyomo.opt import TerminationCondition

# --- Our own modules ---
from data.load_data       import load_data
from model.sets           import get_sets
from model.parameters     import get_parameters
from model.variables      import add_variables
from model.constraints    import add_constraints
from model.objective      import add_objective


# ===========================================================================
# CONFIGURATION  ← edit this section to change runs
# ===========================================================================

# Time window to optimise
# Tip: start with 1 week to test quickly, then extend to full year
START_DATE = '2025-06-01'
END_DATE   = '2025-06-07'

# Solver to use:
#   'glpk'   → free, good for testing  (install: conda install -c conda-forge glpk)
#   'highs'  → free, fast open-source  (install: pip install highspy)
#   'gurobi' → commercial, very fast   (needs a Gurobi license)
SOLVER = 'glpk'

# Print the solver log to the screen? (True = verbose, False = silent)
VERBOSE_SOLVER = True


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("  HYDRO OPTIMISATION MODEL")
    print("  AlpenEnergie — Bidmi & Haselholz System")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # STEP 1: Load Data
    # -----------------------------------------------------------------------
    print("\n[1] Loading data ...")
    df = load_data(start_date=START_DATE, end_date=END_DATE)

    # -----------------------------------------------------------------------
    # STEP 2: Build Sets
    # -----------------------------------------------------------------------
    print("\n[2] Building sets ...")
    sets = get_sets(df)

    # -----------------------------------------------------------------------
    # STEP 3: Build Parameters
    # -----------------------------------------------------------------------
    print("\n[3] Building parameters ...")
    params = get_parameters(df, sets)

    # -----------------------------------------------------------------------
    # STEP 4: Build Pyomo Model
    # -----------------------------------------------------------------------
    print("\n[4] Building Pyomo model ...")
    model = ConcreteModel(name='HydroOptimisation')

    add_variables   (model, sets)
    add_constraints (model, sets, params)
    add_objective   (model, sets, params)

    print(f"\n  Model built successfully!")

    # -----------------------------------------------------------------------
    # STEP 5: Solve
    # -----------------------------------------------------------------------
    print(f"\n[5] Solving with '{SOLVER}' ...")
    solver = SolverFactory(SOLVER)
    result = solver.solve(model, tee=VERBOSE_SOLVER)

    # Check whether the solver found an optimal solution
    if result.solver.termination_condition == TerminationCondition.optimal:
        print("\n  ✓ Optimal solution found!")
    else:
        print(f"\n  ✗ No optimal solution.")
        print(f"    Termination condition: {result.solver.termination_condition}")
        print(f"    Solver status:         {result.solver.status}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 6: Print Quick Summary
    # -----------------------------------------------------------------------
    T = sets['T']
    I = sets['I']

    total_cost       = value(model.objective)
    total_prod       = sum(value(model.P[i, t]) for i in I for t in T)
    total_demand     = sum(params['Demand'][t] for t in T)
    spot_trade_vals  = [value(model.SpotTrade[t]) for t in T]
    total_sold       = sum(v for v in spot_trade_vals if v > 0)
    total_bought     = sum(-v for v in spot_trade_vals if v < 0)
    total_intra      = sum(value(model.IntradayBuy[t]) for t in T)

    print("\n" + "─" * 45)
    print("  RESULTS SUMMARY")
    print("─" * 45)
    print(f"  Period:             {START_DATE}  →  {END_DATE}")
    print(f"  Timesteps solved:   {len(T)}")
    print(f"  Total cost:         {total_cost:>10.2f}  €")
    print(f"  Total production:   {total_prod:>10.1f}  kWh")
    print(f"  Total demand:       {total_demand:>10.1f}  kWh")
    print(f"  Spot sold  (+):     {total_sold:>10.1f}  kWh")
    print(f"  Spot bought (-):    {total_bought:>10.1f}  kWh")
    print(f"  Intraday buy:       {total_intra:>10.1f}  kWh")
    print("─" * 45)

    # -----------------------------------------------------------------------
    # STEP 7: Save Results to CSV
    # -----------------------------------------------------------------------
    print("\n[6] Saving results to results.csv ...")

    rows = []
    for t in T:
        rows.append({
            'DateTime':        df['DateTime'].iloc[t],
            'R_Bidmi_mm':      value(model.R['Bidmi',     t]),
            'R_Haselholz_mm':  value(model.R['Haselholz', t]),
            'Q_M1_mm':         value(model.Q['M1', t]),
            'Q_M2_mm':         value(model.Q['M2', t]),
            'P_M1_kWh':        value(model.P['M1', t]),
            'P_M2_kWh':        value(model.P['M2', t]),
            'P_total_kWh':     value(model.P['M1', t]) + value(model.P['M2', t]),
            'Demand_kWh':      params['Demand'][t],
            'SpotTrade_kWh':   value(model.SpotTrade[t]),   # + = sell, - = buy
            'IntradayBuy_kWh': value(model.IntradayBuy[t]),
            'Price_spot':      params['Price_spot'][t],
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv('results.csv', index=False)
    print("  Saved: results.csv")
    print("\n  Run  python analysis.py  to see the plots.")

    return model, results_df


if __name__ == '__main__':
    main()
