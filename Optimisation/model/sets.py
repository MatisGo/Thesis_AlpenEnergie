"""
Sets
====
Defines the index sets used in the optimisation model.

Sets:
  T  : time steps  (0, 1, 2, ..., N-1)   one per 5-minute interval
  R  : reservoirs  ('Bidmi', 'Haselholz')
  I  : turbines    ('M1', 'M2')

Turbine → Reservoir mapping:
  M1  draws water from  Haselholz  (k = 0.455 kWh/mm)
  M2  draws water from  Bidmi      (k = 3.330 kWh/mm)
"""

# Which turbine uses which reservoir
# This mapping is used in the reservoir balance constraints
TURBINE_RESERVOIR = {
    'M1': 'Haselholz',
    'M2': 'Bidmi',
}


def get_sets(df):
    """
    Build index sets from the loaded data.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_data().

    Returns
    -------
    sets : dict
        'T'  : list of time indices [0, 1, ..., N-1]
        'R'  : list of reservoir names
        'I'  : list of turbine names
    """

    # Time steps: one integer per 5-minute row in the data
    T = list(range(len(df)))

    # Reservoir names (must match column names used in parameters.py)
    R = ['Bidmi', 'Haselholz']

    # Turbine names
    I = ['M1', 'M2']

    print(f"  Sets: {len(T)} timesteps | Reservoirs: {R} | Turbines: {I}")

    return {'T': T, 'R': R, 'I': I}
