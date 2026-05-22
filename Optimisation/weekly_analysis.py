"""
weekly_analysis.py
==================
Interactive WEEKLY dashboard for hydro optimisation results.
For a single-day view use day_analysis.py instead.

Layout
------
  [File selector + Controls (6 curves)] | [Graph]
  ──────────────────────────────────────────────
  [Week selector — bottom]

Single panel. The graph has 6 curve slots with colour-coded Y-axes.
Reservoir-level columns automatically show Min/Max bounds in red.
Y-axis range can be overridden per curve with the Min/Max inputs.

Each selected week starts on Monday and spans 7 days (Mon → Sun).

Usage
-----
  python weekly_analysis.py  ← weekly view
  python day_analysis.py     ← daily view
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')

# Reservoir columns → automatic Min / Max horizontal lines in red
LEVEL_BOUNDS = {
    'Bidmi_mm':         (1000, 2200),
    'Haselholz_mm':     ( 600, 2800),
    'Opt_Bidmi_mm':     (1000, 2200),
    'Opt_Haselholz_mm': ( 600, 2800),
}

CURVE_COLORS = ['#2166ac', '#d6604d', '#4dac26', '#8073ac', '#e08214', '#01665e']
BOUND_COLOR  = '#b2182b'
N_CURVES     = 6

# Outward px offsets for the 5 right-side twin axes (curves 2..6)
AXIS_OFFSETS = [0, 60, 120, 180, 240]

# Group prefixes shown in the curve dropdowns. The mapping decides where each
# raw column name lands; unknown columns fall under 'Other'.
COLUMN_GROUPS = [
    ('Hydro',    ['Opt_Production_kW', 'Opt_M1_kW', 'Opt_M2_kW',
                  'Forecast_kW', 'Forecast_Drift_kW',
                  'Ref_M1_kW', 'Ref_M2_kW',
                  'Opt_Spill_Bidmi_kWh', 'Opt_Spill_Haselholz_kWh',
                  'Opt_Spill_Bidmi_kWh_Cum', 'Opt_Spill_Haselholz_kWh_Cum']),
    ('Reservoir',['Opt_Bidmi_mm', 'Opt_Haselholz_mm',
                  'Bidmi_mm', 'Haselholz_mm',
                  'Opt_Target_Bidmi_mm', 'Opt_Target_Haselholz_mm',
                  'Bidmi_Inflow_ls', 'Haselholz_Inflow_ls']),
    ('Load',     ['Consumption_kW', 'Forecast_Consumption_kW']),
    ('Grid',     ['DA_Import_kW', 'DA_Export_kW',
                  'Imbalance_Import_kW',
                  'P_Import_kW', 'P_Import_15min_kW', 'P_Exchange_kW']),
    ('Battery',  ['Batt_SOC_kWh', 'Batt_Charge_kW', 'Batt_Discharge_kW',
                  'Batt_Net_kW', 'Batt_Revenue_EUR']),
    ('Money',    ['Opt_DA_Trading_EUR', 'Opt_ID_Imbalance_EUR',
                  'Opt_Energy_Trading_EUR', 'Opt_Total_Energy_Cost_EUR',
                  'Day_Ahead_Price_EUR_MWh',
                  'BG_Long_EUR_MWh', 'BG_Short_EUR_MWh']),
    ('Mode',     ['Opt_Dispatch_Mode', 'Opt_Recovery_Bidmi',
                  'Opt_Recovery_Haselholz', 'Opt_Forecast_Scale']),
]


def _grouped_label(col: str) -> str:
    """Prefix each column name with its group ('Battery: Batt_SOC_kWh')."""
    for grp, cols in COLUMN_GROUPS:
        if col in cols:
            return f'{grp}: {col}'
    return f'Other: {col}'


def _label_to_col(label: str) -> str:
    """Strip the group prefix to recover the raw column name."""
    return label.split(': ', 1)[1] if ': ' in label else label


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class WeeklyAnalysisApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Weekly Analysis — AlpenEnergie")
        self.root.geometry("1280x780")
        self.root.minsize(960, 540)
        self.panel = None
        self._week_lookup = {}    # label → Monday timestamp

        self._build_ui()
        self._auto_load_panel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _available_files(self):
        """Return list of results .xlsx filenames (excludes summary/analysis files)."""
        if not os.path.isdir(OUTPUT_DIR):
            return []
        RESULT_PREFIXES = ('FC_', 'OD_', 'RBD_', 'WV_')
        return sorted(
            f for f in os.listdir(OUTPUT_DIR)
            if f.endswith('.xlsx') and any(f.startswith(p) for p in RESULT_PREFIXES))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Bottom bar (week selector) ────────────────────────────────
        bar = tk.Frame(self.root, relief='ridge', bd=2, pady=6, bg='#e8e8e8')
        bar.pack(side='bottom', fill='x')

        tk.Label(bar, text="Week starting:", font=('Arial', 11, 'bold'),
                 bg='#e8e8e8').pack(side='left', padx=(15, 5))

        self.week_var   = tk.StringVar()
        self.week_combo = ttk.Combobox(
            bar, textvariable=self.week_var,
            width=42, state='readonly', font=('Arial', 10))
        self.week_combo.pack(side='left')
        self.week_combo.bind('<<ComboboxSelected>>', lambda _: self._redraw())

        tk.Label(bar, text="  ← select a week (Mon → Sun)",
                 font=('Arial', 9, 'italic'), fg='#666666',
                 bg='#e8e8e8').pack(side='left')

        # ── Main area ─────────────────────────────────────────────────
        top = tk.Frame(self.root)
        top.pack(side='top', fill='both', expand=True)

        self.panel = self._make_panel(top)
        self.panel['outer'].pack(side='left', fill='both', expand=True,
                                 padx=3, pady=3)

    def _make_panel(self, parent):
        """Build the single (controls + figure) panel."""
        outer = tk.Frame(parent, bd=2, relief='groove')

        # ── Left control strip ────────────────────────────────────────
        ctrl = tk.Frame(outer, width=235, bg='#f0f0f0')
        ctrl.pack(side='left', fill='y', padx=2, pady=2)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="  Graph",
                 font=('Arial', 10, 'bold'), bg='#f0f0f0',
                 anchor='w').pack(fill='x', pady=(10, 2))

        # ── File selector ─────────────────────────────────────────────
        tk.Label(ctrl, text="File:", font=('Arial', 8, 'bold'),
                 bg='#f0f0f0', anchor='w').pack(fill='x', padx=6)

        file_var = tk.StringVar()
        file_combo = ttk.Combobox(ctrl, textvariable=file_var,
                                  width=22, state='readonly', font=('Arial', 8))
        file_combo['values'] = self._available_files()
        file_combo.pack(fill='x', padx=6, pady=(1, 0))

        tk.Button(ctrl, text="Load file",
                  command=self._load_panel,
                  width=14, font=('Arial', 8)).pack(pady=(3, 0))

        loaded_label = tk.Label(ctrl, text="(no file loaded)",
                                font=('Arial', 7, 'italic'), fg='#888888',
                                bg='#f0f0f0', anchor='w', wraplength=210)
        loaded_label.pack(fill='x', padx=6, pady=(1, 4))

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', padx=6, pady=4)

        sel_vars   = []
        sel_combos = []
        ymin_vars  = []
        ymax_vars  = []

        for j in range(N_CURVES):
            color = CURVE_COLORS[j]

            # ── Selector row ──────────────────────────────────────────
            top_row = tk.Frame(ctrl, bg='#f0f0f0')
            top_row.pack(fill='x', padx=6, pady=(4, 0))

            tk.Label(top_row, text='●', fg=color,
                     bg='#f0f0f0', font=('Arial', 12)).pack(side='left')

            var   = tk.StringVar(value='— none —')
            combo = ttk.Combobox(top_row, textvariable=var,
                                 width=16, state='readonly')
            combo.pack(side='left', padx=3, fill='x', expand=True)
            combo.bind('<<ComboboxSelected>>', lambda _: self._redraw())

            sel_vars.append(var)
            sel_combos.append(combo)

            # ── Per-curve Y-axis override ─────────────────────────────
            minmax_row = tk.Frame(ctrl, bg='#f0f0f0')
            minmax_row.pack(fill='x', padx=22, pady=(1, 0))

            ymin_v = tk.StringVar()
            ymax_v = tk.StringVar()

            tk.Label(minmax_row, text='Min', fg=color, bg='#f0f0f0',
                     font=('Arial', 7), width=3).pack(side='left')
            e_min = tk.Entry(minmax_row, textvariable=ymin_v, width=6,
                             font=('Arial', 7))
            e_min.pack(side='left', padx=1)
            e_min.bind('<Return>', lambda _: self._redraw())

            tk.Label(minmax_row, text='Max', fg=color, bg='#f0f0f0',
                     font=('Arial', 7), width=3).pack(side='left', padx=(4, 0))
            e_max = tk.Entry(minmax_row, textvariable=ymax_v, width=6,
                             font=('Arial', 7))
            e_max.pack(side='left', padx=1)
            e_max.bind('<Return>', lambda _: self._redraw())

            ymin_vars.append(ymin_v)
            ymax_vars.append(ymax_v)

        # Apply button
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', padx=6, pady=8)
        tk.Button(ctrl, text="Apply Y-axes",
                  command=self._redraw,
                  width=14).pack(pady=4)

        # ── Matplotlib figure ─────────────────────────────────────────
        fig = plt.figure(figsize=(9, 5))
        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        return {
            'outer':        outer,
            'file_var':     file_var,
            'file_combo':   file_combo,
            'loaded_label': loaded_label,
            'sel_vars':     sel_vars,
            'sel_combos':   sel_combos,
            'ymin_vars':    ymin_vars,
            'ymax_vars':    ymax_vars,
            'fig':          fig,
            'canvas':       canvas,
            'df':           None,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _auto_load_panel(self):
        """On startup, pre-load the first available file."""
        files = self._available_files()
        if not files:
            return
        self.panel['file_var'].set(files[0])
        self._load_panel()

    def _load_panel(self):
        """Load the selected file and refresh the graph."""
        p     = self.panel
        fname = p['file_var'].get()
        if not fname:
            return

        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"'{fname}' not found in Output/.")
            return

        try:
            df = pd.read_excel(path, sheet_name='Results',
                               parse_dates=['DateTime'], engine='calamine')
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        p['df'] = df
        p['loaded_label'].config(text=fname)

        # Build dropdown values: group-prefixed labels, ordered by COLUMN_GROUPS.
        numeric_cols = [c for c in df.columns
                        if c != 'DateTime' and pd.api.types.is_numeric_dtype(df[c])]
        ordered = []
        for grp, cols in COLUMN_GROUPS:
            for c in cols:
                if c in numeric_cols:
                    ordered.append(f'{grp}: {c}')
        # Anything not in a group lands in "Other"
        grouped_cols = {c for _, cols in COLUMN_GROUPS for c in cols}
        for c in numeric_cols:
            if c not in grouped_cols:
                ordered.append(f'Other: {c}')
        labels = ['— none —'] + ordered

        DEFAULTS = [
            'Opt_Production_kW',
            'Forecast_kW',
            'DA_Import_kW',
            'Batt_SOC_kWh',
            'Batt_Charge_kW',
            'Batt_Discharge_kW',
        ]
        for j, combo in enumerate(p['sel_combos']):
            combo['values'] = labels
            current_col = _label_to_col(combo.get())
            if current_col in numeric_cols:
                combo.set(_grouped_label(current_col))
            else:
                default = DEFAULTS[j] if j < len(DEFAULTS) else None
                if default and default in numeric_cols:
                    combo.set(_grouped_label(default))
                else:
                    combo.set('— none —')

        self._update_weeks()
        self._redraw()

    def _update_weeks(self):
        """Build the week dropdown from the loaded dataset (Monday-anchored)."""
        p = self.panel
        if p['df'] is None:
            return

        dt = p['df']['DateTime']
        # Anchor each timestamp to the Monday of its ISO week
        mondays = (dt - pd.to_timedelta(dt.dt.weekday, unit='D')).dt.normalize()
        unique_mondays = sorted(pd.Timestamp(d) for d in mondays.unique())

        labels = [
            f"{m.strftime('%Y-%m-%d')}  →  {(m + pd.Timedelta(days=6)).strftime('%Y-%m-%d')}"
            for m in unique_mondays
        ]
        self._week_lookup = dict(zip(labels, unique_mondays))
        self.week_combo['values'] = labels
        if labels and self.week_var.get() not in labels:
            self.week_var.set(labels[0])

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _week_df(self):
        p = self.panel
        if p['df'] is None or self.week_var.get() not in self._week_lookup:
            return None
        start = self._week_lookup[self.week_var.get()]
        end   = start + pd.Timedelta(days=7)
        mask  = (p['df']['DateTime'] >= start) & (p['df']['DateTime'] < end)
        d     = p['df'].loc[mask].copy()
        return d if not d.empty else None

    def _redraw(self):
        p   = self.panel
        fig = p['fig']
        fig.clear()

        week = self._week_df()
        if week is None:
            p['canvas'].draw()
            return

        t = week['DateTime']

        active = []
        for ci, var in enumerate(p['sel_vars']):
            label = var.get()
            if label == '— none —':
                continue
            col = _label_to_col(label)
            if col in week.columns:
                active.append((ci, col))

        if not active:
            p['canvas'].draw()
            return

        n = len(active)

        # ── Adjust figure margins (more right space when many axes) ───
        right_margin = max(0.55, 0.94 - 0.07 * max(0, n - 1))
        fig.subplots_adjust(left=0.08, right=right_margin,
                            top=0.92, bottom=0.22)

        # ── Create axes ───────────────────────────────────────────────
        ax_main = fig.add_subplot(111)
        axes    = [ax_main]

        for k in range(1, n):
            ax_twin = ax_main.twinx()
            offset_idx = k - 1
            offset_px = (AXIS_OFFSETS[offset_idx]
                         if offset_idx < len(AXIS_OFFSETS)
                         else AXIS_OFFSETS[-1] + 60 * (offset_idx - len(AXIS_OFFSETS) + 1))
            ax_twin.spines['right'].set_position(('outward', offset_px))
            axes.append(ax_twin)

        if n > 1:
            ax_main.spines['right'].set_visible(False)

        # ── Plot each curve ───────────────────────────────────────────
        legend_handles = []
        legend_labels  = []
        bound_added    = set()

        for plot_idx, (ci, col) in enumerate(active):
            ax    = axes[plot_idx]
            color = CURVE_COLORS[ci]

            line, = ax.plot(t, week[col], color=color,
                            linewidth=1.1, label=col)

            ax.set_ylabel(col, color=color, fontsize=7, labelpad=3)
            ax.tick_params(axis='y', colors=color, labelsize=7)
            ax.spines['right' if plot_idx > 0 else 'left'].set_color(color)

            try:
                ylo = float(p['ymin_vars'][ci].get())
            except ValueError:
                ylo = None
            try:
                yhi = float(p['ymax_vars'][ci].get())
            except ValueError:
                yhi = None
            if ylo is not None or yhi is not None:
                cur_lo, cur_hi = ax.get_ylim()
                new_lo = ylo if ylo is not None else cur_lo
                new_hi = yhi if yhi is not None else cur_hi
                if new_lo < new_hi:
                    ax.set_ylim(new_lo, new_hi)

            if col in LEVEL_BOUNDS:
                lo, hi = LEVEL_BOUNDS[col]
                ax.axhline(lo, color=BOUND_COLOR, lw=1.0,
                           ls='--', alpha=0.75, zorder=0)
                ax.axhline(hi, color=BOUND_COLOR, lw=1.0,
                           ls='-',  alpha=0.75, zorder=0)
                if 'min' not in bound_added:
                    legend_handles.append(
                        Line2D([0], [0], color=BOUND_COLOR, lw=1.0, ls='--'))
                    legend_labels.append(f'Min  {lo} mm')
                    bound_added.add('min')
                if 'max' not in bound_added:
                    legend_handles.append(
                        Line2D([0], [0], color=BOUND_COLOR, lw=1.0, ls='-'))
                    legend_labels.append(f'Max  {hi} mm')
                    bound_added.add('max')

            legend_handles.append(line)
            legend_labels.append(col)

        # ── X-axis: day boundaries with weekday labels, hour minor ticks ─
        ax_main.xaxis.set_major_locator(mdates.DayLocator())
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d/%m'))
        ax_main.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        ax_main.grid(True, which='major', alpha=0.35, linestyle='-')
        ax_main.grid(True, which='minor', alpha=0.15, linestyle=':')

        title = f"{self.week_var.get()}   —   {p['file_var'].get()}"
        ax_main.set_title(title, fontsize=8, pad=4)

        if legend_handles:
            ax_main.legend(legend_handles, legend_labels,
                           loc='upper center',
                           bbox_to_anchor=(0.5, -0.18),
                           ncol=min(len(legend_handles), 4),
                           fontsize=7,
                           framealpha=0.88, edgecolor='#aaaaaa')

        p['canvas'].draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    WeeklyAnalysisApp(root)

    def _on_close():
        plt.close('all')
        root.quit()
        root.destroy()

    # Window-X button → fully shut down (Tk + matplotlib + process)
    root.protocol('WM_DELETE_WINDOW', _on_close)
    try:
        root.mainloop()
    finally:
        plt.close('all')


if __name__ == '__main__':
    main()