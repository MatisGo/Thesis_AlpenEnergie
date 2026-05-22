"""
day_analysis.py
===============
Interactive daily dashboard for hydro optimisation results — dual-file comparison.

Layout
------
  [File selectors + Controls (6 curves)] | [Graph]
  ────────────────────────────────────────────────
  [Day selector + Prev / Next — bottom]

Two files can be loaded simultaneously:
  File 1 → solid lines
  File 2 → dotted lines

Each curve slot has a Show checkbox (hide/show without removing the selection),
plus independent Min / Max Y-axis overrides.

Usage
-----
  python day_analysis.py
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------

class _Tooltip:
    """Show a popup with full text when hovering over a widget."""

    def __init__(self, widget, text_var):
        self._widget   = widget
        self._text_var = text_var   # callable or StringVar — called at show time
        self._tip      = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, _event=None):
        text = self._text_var() if callable(self._text_var) else self._text_var.get()
        if not text or text in ('(not loaded)', ''):
            return
        x = self._widget.winfo_rootx() + 10
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self._tip, text=text, justify='left',
                 background='#ffffcc', relief='solid', borderwidth=1,
                 font=('Arial', 8), wraplength=600).pack()

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')

LEVEL_BOUNDS = {
    'Bidmi_mm':         (1000, 2200),
    'Haselholz_mm':     ( 600, 2800),
    'Opt_Bidmi_mm':     (1000, 2200),
    'Opt_Haselholz_mm': ( 600, 2800),
}

CURVE_COLORS = ['#2166ac', '#d6604d', '#4dac26', '#8073ac', '#e08214', '#01665e']
BOUND_COLOR  = '#b2182b'
N_CURVES     = 6

AXIS_OFFSETS = [0, 60, 120, 180, 240]

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

DEFAULTS = [
    'Opt_Production_kW',
    'Forecast_kW',
    'DA_Import_kW',
    'Batt_SOC_kWh',
    'Batt_Charge_kW',
    'Batt_Discharge_kW',
]


def _grouped_label(col: str) -> str:
    for grp, cols in COLUMN_GROUPS:
        if col in cols:
            return f'{grp}: {col}'
    return f'Other: {col}'


def _label_to_col(label: str) -> str:
    return label.split(': ', 1)[1] if ': ' in label else label


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DailyAnalysisApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Daily Analysis — AlpenEnergie")
        self.root.geometry("1440x800")
        self.root.minsize(1000, 560)
        self.panel = None
        self._day_lookup = {}

        self._build_ui()
        self._auto_load_panel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _available_files(self):
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
        # ── Bottom bar (day selector) ─────────────────────────────────
        bar = tk.Frame(self.root, relief='ridge', bd=2, pady=6, bg='#e8e8e8')
        bar.pack(side='bottom', fill='x')

        tk.Label(bar, text="Date:", font=('Arial', 11, 'bold'),
                 bg='#e8e8e8').pack(side='left', padx=(15, 5))

        self.day_var   = tk.StringVar()
        self.day_combo = ttk.Combobox(
            bar, textvariable=self.day_var,
            width=28, state='readonly', font=('Arial', 10))
        self.day_combo.pack(side='left')
        self.day_combo.bind('<<ComboboxSelected>>', lambda _: self._redraw())

        tk.Label(bar, text="  ← select a day",
                 font=('Arial', 9, 'italic'), fg='#666666',
                 bg='#e8e8e8').pack(side='left')

        tk.Button(bar, text="◀ Prev", font=('Arial', 9),
                  command=self._prev_day).pack(side='left', padx=(20, 2))
        tk.Button(bar, text="Next ▶", font=('Arial', 9),
                  command=self._next_day).pack(side='left', padx=2)

        # ── Main area ─────────────────────────────────────────────────
        top = tk.Frame(self.root)
        top.pack(side='top', fill='both', expand=True)

        self.panel = self._make_panel(top)
        self.panel['outer'].pack(side='left', fill='both', expand=True,
                                 padx=3, pady=3)

    def _make_panel(self, parent):
        outer = tk.Frame(parent, bd=2, relief='groove')

        # ── Left control strip ────────────────────────────────────────
        ctrl = tk.Frame(outer, width=320, bg='#f0f0f0')
        ctrl.pack(side='left', fill='y', padx=2, pady=2)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="  Graph",
                 font=('Arial', 10, 'bold'), bg='#f0f0f0',
                 anchor='w').pack(fill='x', pady=(8, 2))

        # ── File 1 ────────────────────────────────────────────────────
        tk.Label(ctrl, text="File 1  (solid line):",
                 font=('Arial', 8, 'bold'), bg='#f0f0f0',
                 fg='#333333', anchor='w').pack(fill='x', padx=6, pady=(4, 0))

        f1_row = tk.Frame(ctrl, bg='#f0f0f0')
        f1_row.pack(fill='x', padx=6, pady=(1, 0))

        file_var1 = tk.StringVar()
        file_combo1 = ttk.Combobox(f1_row, textvariable=file_var1,
                                   width=24, state='readonly', font=('Arial', 8))
        file_combo1['values'] = self._available_files()
        file_combo1.pack(side='left', fill='x', expand=True)
        tk.Button(f1_row, text="Load", width=5, font=('Arial', 8),
                  command=self._load_file1).pack(side='left', padx=(3, 0))

        loaded_label1 = tk.Label(ctrl, text="(not loaded)",
                                 font=('Arial', 7, 'italic'), fg='#888888',
                                 bg='#f0f0f0', anchor='w', wraplength=300)
        loaded_label1.pack(fill='x', padx=6, pady=(0, 2))
        _Tooltip(loaded_label1, lambda: loaded_label1.cget('text'))

        # ── File 2 ────────────────────────────────────────────────────
        tk.Label(ctrl, text="File 2  (dotted line):",
                 font=('Arial', 8, 'bold'), bg='#f0f0f0',
                 fg='#555555', anchor='w').pack(fill='x', padx=6, pady=(4, 0))

        f2_row = tk.Frame(ctrl, bg='#f0f0f0')
        f2_row.pack(fill='x', padx=6, pady=(1, 0))

        file_var2 = tk.StringVar()
        file_combo2 = ttk.Combobox(f2_row, textvariable=file_var2,
                                   width=24, state='readonly', font=('Arial', 8))
        file_combo2['values'] = self._available_files()
        file_combo2.pack(side='left', fill='x', expand=True)
        tk.Button(f2_row, text="Load", width=5, font=('Arial', 8),
                  command=self._load_file2).pack(side='left', padx=(3, 0))

        loaded_label2 = tk.Label(ctrl, text="(not loaded)",
                                 font=('Arial', 7, 'italic'), fg='#888888',
                                 bg='#f0f0f0', anchor='w', wraplength=300)
        loaded_label2.pack(fill='x', padx=6, pady=(0, 4))
        _Tooltip(loaded_label2, lambda: loaded_label2.cget('text'))

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', padx=6, pady=4)

        # ── Curve slots ───────────────────────────────────────────────
        sel_vars   = []
        sel_combos = []
        ymin_vars  = []
        ymax_vars  = []
        show_vars  = []

        for j in range(N_CURVES):
            color = CURVE_COLORS[j]

            # Row 1: colour dot + column dropdown + Show checkbox
            top_row = tk.Frame(ctrl, bg='#f0f0f0')
            top_row.pack(fill='x', padx=6, pady=(4, 0))

            tk.Label(top_row, text='●', fg=color,
                     bg='#f0f0f0', font=('Arial', 12)).pack(side='left')

            var   = tk.StringVar(value='— none —')
            combo = ttk.Combobox(top_row, textvariable=var,
                                 width=13, state='readonly', font=('Arial', 8))
            combo.pack(side='left', padx=2, fill='x', expand=True)
            combo.bind('<<ComboboxSelected>>', lambda _: self._redraw())

            show_var = tk.BooleanVar(value=True)
            tk.Checkbutton(top_row, text='Show', variable=show_var,
                           command=self._redraw,
                           bg='#f0f0f0', font=('Arial', 7),
                           activebackground='#f0f0f0').pack(side='left', padx=(2, 0))

            sel_vars.append(var)
            sel_combos.append(combo)
            show_vars.append(show_var)

            # Row 2: Min / Max entries
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

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', padx=6, pady=8)
        tk.Button(ctrl, text="Apply Y-axes",
                  command=self._redraw,
                  width=14).pack(pady=4)

        # ── Matplotlib figure ─────────────────────────────────────────
        fig = plt.figure(figsize=(9, 5))
        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        return {
            'outer':         outer,
            'file_var1':     file_var1,
            'file_combo1':   file_combo1,
            'loaded_label1': loaded_label1,
            'file_var2':     file_var2,
            'file_combo2':   file_combo2,
            'loaded_label2': loaded_label2,
            'sel_vars':      sel_vars,
            'sel_combos':    sel_combos,
            'show_vars':     show_vars,
            'ymin_vars':     ymin_vars,
            'ymax_vars':     ymax_vars,
            'fig':           fig,
            'canvas':        canvas,
            'df1':           None,
            'df2':           None,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _auto_load_panel(self):
        files = self._available_files()
        if not files:
            return
        self.panel['file_var1'].set(files[0])
        self._load_file1()

    def _load_file1(self):
        self._load_file(1)

    def _load_file2(self):
        self._load_file(2)

    def _load_file(self, n: int):
        p     = self.panel
        fname = p[f'file_var{n}'].get()
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

        p[f'df{n}'] = df
        p[f'loaded_label{n}'].config(text=fname)

        self._refresh_combos()
        self._update_days()
        self._redraw()

    def _refresh_combos(self):
        """Rebuild column dropdowns from the union of both loaded DataFrames."""
        p = self.panel
        numeric_cols = set()
        for key in ('df1', 'df2'):
            df = p[key]
            if df is not None:
                for c in df.columns:
                    if c != 'DateTime' and pd.api.types.is_numeric_dtype(df[c]):
                        numeric_cols.add(c)

        ordered = []
        for grp, cols in COLUMN_GROUPS:
            for c in cols:
                if c in numeric_cols:
                    ordered.append(f'{grp}: {c}')
        grouped_cols = {c for _, cols in COLUMN_GROUPS for c in cols}
        for c in sorted(numeric_cols):
            if c not in grouped_cols:
                ordered.append(f'Other: {c}')
        labels = ['— none —'] + ordered

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

    def _update_days(self):
        p  = self.panel
        df = p['df1'] if p['df1'] is not None else p['df2']
        if df is None:
            return

        dates         = df['DateTime'].dt.normalize().unique()
        unique_dates  = sorted(pd.Timestamp(d) for d in dates)
        labels        = [d.strftime('%Y-%m-%d  (%A)') for d in unique_dates]
        self._day_lookup = dict(zip(labels, unique_dates))
        self.day_combo['values'] = labels
        if labels and self.day_var.get() not in labels:
            self.day_var.set(labels[0])

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _prev_day(self):
        labels = list(self.day_combo['values'])
        if not labels:
            return
        cur = self.day_var.get()
        idx = labels.index(cur) if cur in labels else 0
        if idx > 0:
            self.day_var.set(labels[idx - 1])
            self._redraw()

    def _next_day(self):
        labels = list(self.day_combo['values'])
        if not labels:
            return
        cur = self.day_var.get()
        idx = labels.index(cur) if cur in labels else 0
        if idx < len(labels) - 1:
            self.day_var.set(labels[idx + 1])
            self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _day_dfs(self):
        """Return (day_df1, day_df2) filtered to the selected calendar day."""
        p   = self.panel
        sel = self.day_var.get()
        if sel not in self._day_lookup:
            return None, None
        start = self._day_lookup[sel]
        end   = start + pd.Timedelta(days=1)

        def _filter(df):
            if df is None:
                return None
            mask = (df['DateTime'] >= start) & (df['DateTime'] < end)
            d = df.loc[mask].copy()
            return d if not d.empty else None

        return _filter(p['df1']), _filter(p['df2'])

    def _redraw(self):
        p   = self.panel
        fig = p['fig']
        fig.clear()

        day1, day2 = self._day_dfs()
        if day1 is None and day2 is None:
            p['canvas'].draw()
            return

        # Build list of curves to draw (only those with Show ticked)
        active = []
        for ci, var in enumerate(p['sel_vars']):
            if not p['show_vars'][ci].get():
                continue
            label = var.get()
            if label == '— none —':
                continue
            col  = _label_to_col(label)
            in1  = day1 is not None and col in day1.columns
            in2  = day2 is not None and col in day2.columns
            if in1 or in2:
                active.append((ci, col, in1, in2))

        if not active:
            p['canvas'].draw()
            return

        n = len(active)

        right_margin = max(0.55, 0.94 - 0.07 * max(0, n - 1))
        fig.subplots_adjust(left=0.08, right=right_margin,
                            top=0.90, bottom=0.22)

        ax_main = fig.add_subplot(111)
        axes    = [ax_main]

        for k in range(1, n):
            ax_twin    = ax_main.twinx()
            offset_idx = k - 1
            offset_px  = (AXIS_OFFSETS[offset_idx]
                          if offset_idx < len(AXIS_OFFSETS)
                          else AXIS_OFFSETS[-1] + 60 * (offset_idx - len(AXIS_OFFSETS) + 1))
            ax_twin.spines['right'].set_position(('outward', offset_px))
            axes.append(ax_twin)

        if n > 1:
            ax_main.spines['right'].set_visible(False)

        legend_handles = []
        legend_labels  = []
        bound_added    = set()

        for plot_idx, (ci, col, in1, in2) in enumerate(active):
            ax    = axes[plot_idx]
            color = CURVE_COLORS[ci]

            # File 1 — solid line
            if in1:
                line1, = ax.plot(day1['DateTime'], day1[col],
                                 color=color, linewidth=1.2,
                                 linestyle='-')
                legend_handles.append(line1)
                legend_labels.append(f'{col}  [F1]')

            # File 2 — dotted line
            if in2:
                line2, = ax.plot(day2['DateTime'], day2[col],
                                 color=color, linewidth=1.4,
                                 linestyle=':')
                legend_handles.append(line2)
                legend_labels.append(f'{col}  [F2]')

            ax.set_ylabel(col, color=color, fontsize=7, labelpad=3)
            ax.tick_params(axis='y', colors=color, labelsize=7)
            ax.spines['right' if plot_idx > 0 else 'left'].set_color(color)

            # Y-axis overrides (min and max applied independently)
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

            # Reservoir bounds
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

        # ── X-axis: every 2h major, 15-min minor ──────────────────────
        ax_main.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_main.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
        ax_main.grid(True, which='major', alpha=0.35, linestyle='-')
        ax_main.grid(True, which='minor', alpha=0.15, linestyle=':')
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')

        f1_name = p['file_var1'].get() or '—'
        f2_name = p['file_var2'].get() or '—'
        title   = (f"{self.day_var.get()}\n"
                   f"── {f1_name}   ··· {f2_name}")
        ax_main.set_title(title, fontsize=7, pad=4, loc='left')

        if legend_handles:
            ax_main.legend(legend_handles, legend_labels,
                           loc='upper center',
                           bbox_to_anchor=(0.5, -0.22),
                           ncol=min(len(legend_handles), 4),
                           fontsize=7,
                           framealpha=0.88, edgecolor='#aaaaaa')

        p['canvas'].draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    DailyAnalysisApp(root)

    def _on_close():
        plt.close('all')
        root.quit()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', _on_close)
    try:
        root.mainloop()
    finally:
        plt.close('all')


if __name__ == '__main__':
    main()
