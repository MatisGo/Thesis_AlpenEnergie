"""
daily_analysis.py
=================
Interactive daily dashboard for hydro optimisation results.

Layout
------
  [File selector + Controls 1] | [Graph 1] || [File selector + Controls 2] | [Graph 2]
  ──────────────────────────────────────────────────────────────────────────────────────
  [Date selector — bottom, shara ed across both graphs]

Each panel loads its own results file independently.
Each graph has 4 curve slots with colour-coded Y-axes.
Reservoir-level columns automatically show Min/Max bounds in red.
Y-axis range can be overridden per curve with the Min/Max inputs.

Usage
-----
  python daily_analysis.py
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

CURVE_COLORS = ['#2166ac', '#d6604d', '#4dac26', '#8073ac']
BOUND_COLOR  = '#b2182b'
N_CURVES     = 4

# Right-axis offsets (outward px) for curves 2, 3, 4
AXIS_OFFSETS = [0, 60, 120]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DailyAnalysisApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Daily Analysis — AlpenEnergie")
        self.root.geometry("1560x820")
        self.root.minsize(960, 540)
        self.panels = []

        self._build_ui()
        self._auto_load_panels()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _available_files(self):
        """Return list of results .xlsx filenames (excludes summary files without a Results sheet)."""
        if not os.path.isdir(OUTPUT_DIR):
            return []
        EXCLUDE = {'Main_results.xlsx'}
        return sorted(
            f for f in os.listdir(OUTPUT_DIR)
            if f.endswith('.xlsx') and f not in EXCLUDE)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Bottom bar (shared date selector) ─────────────────────────
        bar = tk.Frame(self.root, relief='ridge', bd=2, pady=6, bg='#e8e8e8')
        bar.pack(side='bottom', fill='x')

        tk.Label(bar, text="Date:", font=('Arial', 11, 'bold'),
                 bg='#e8e8e8').pack(side='left', padx=(15, 5))

        self.date_var   = tk.StringVar()
        self.date_combo = ttk.Combobox(
            bar, textvariable=self.date_var,
            width=13, state='readonly', font=('Arial', 10))
        self.date_combo.pack(side='left')
        self.date_combo.bind('<<ComboboxSelected>>', lambda _: self._refresh_all())

        tk.Label(bar, text="  ← select a date to update both graphs",
                 font=('Arial', 9, 'italic'), fg='#666666',
                 bg='#e8e8e8').pack(side='left')

        # ── Main area ─────────────────────────────────────────────────
        top = tk.Frame(self.root)
        top.pack(side='top', fill='both', expand=True)

        for i in range(2):
            p = self._make_panel(top, i)
            p['outer'].pack(side='left', fill='both', expand=True,
                            padx=3, pady=3)
            self.panels.append(p)

    def _make_panel(self, parent, idx):
        """Build one (controls + figure) panel. Returns state dict."""
        outer = tk.Frame(parent, bd=2, relief='groove')

        # ── Left control strip ────────────────────────────────────────
        ctrl = tk.Frame(outer, width=225, bg='#f0f0f0')
        ctrl.pack(side='left', fill='y', padx=2, pady=2)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text=f"  Graph {idx + 1}",
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
                  command=lambda i=idx: self._load_panel(i),
                  width=14, font=('Arial', 8)).pack(pady=(3, 0))

        # Label showing currently loaded file
        loaded_label = tk.Label(ctrl, text="(no file loaded)",
                                font=('Arial', 7, 'italic'), fg='#888888',
                                bg='#f0f0f0', anchor='w', wraplength=200)
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
            combo.bind('<<ComboboxSelected>>',
                       lambda _, i=idx: self._redraw(i))

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
            e_min.bind('<Return>', lambda _, i=idx: self._redraw(i))

            tk.Label(minmax_row, text='Max', fg=color, bg='#f0f0f0',
                     font=('Arial', 7), width=3).pack(side='left', padx=(4, 0))
            e_max = tk.Entry(minmax_row, textvariable=ymax_v, width=6,
                             font=('Arial', 7))
            e_max.pack(side='left', padx=1)
            e_max.bind('<Return>', lambda _, i=idx: self._redraw(i))

            ymin_vars.append(ymin_v)
            ymax_vars.append(ymax_v)

        # Apply button
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', padx=6, pady=8)
        tk.Button(ctrl, text="Apply Y-axes",
                  command=lambda i=idx: self._redraw(i),
                  width=14).pack(pady=4)

        # ── Matplotlib figure ─────────────────────────────────────────
        fig = plt.figure(figsize=(6, 5))
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

    def _auto_load_panels(self):
        """On startup, pre-load the first available file into both panels."""
        files = self._available_files()
        if not files:
            return
        for i, p in enumerate(self.panels):
            # Load first file into panel 0, second (if exists) into panel 1
            fname = files[min(i, len(files) - 1)]
            p['file_var'].set(fname)
            self._load_panel(i)

    def _load_panel(self, idx):
        """Load the selected file into panel idx and refresh."""
        p     = self.panels[idx]
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

        # Update column selectors for this panel
        cols = ['— none —'] + [
            c for c in df.columns
            if c != 'DateTime' and pd.api.types.is_numeric_dtype(df[c])
        ]
        # Panel 0: hydro production view
        # Panel 1: battery / financial / grid-import view
        PANEL_DEFAULTS = [
            [   # Panel 0 — hydro
                'Opt_Production_kW',
                'Forecast_kW',
                'Opt_Bidmi_mm',
                'Opt_Haselholz_mm',
            ],
            [   # Panel 1 — battery & financials
                'Batt_SOC_kWh',
                'Batt_Net_kW',
                'P_Import_15min_kW',
                'Opt_Energy_Trading_EUR',
            ],
        ]
        defaults = PANEL_DEFAULTS[idx] if idx < len(PANEL_DEFAULTS) else PANEL_DEFAULTS[0]
        for j, combo in enumerate(p['sel_combos']):
            current = combo.get()
            combo['values'] = cols
            if current not in cols:
                default = defaults[j] if j < len(defaults) else '— none —'
                combo.set(default if default in cols else '— none —')

        self._update_dates()
        self._redraw(idx)

    def _update_dates(self):
        """Recompute shared date list as union across all loaded panels."""
        all_dates = set()
        for p in self.panels:
            if p['df'] is not None:
                all_dates.update(
                    p['df']['DateTime'].dt.date.unique().astype(str))
        dates = sorted(all_dates)
        self.date_combo['values'] = dates
        if dates and self.date_var.get() not in dates:
            self.date_var.set(dates[0])
            self._refresh_all()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _refresh_all(self):
        self._redraw(0)
        self._redraw(1)

    def _day_df(self, idx):
        p        = self.panels[idx]
        date_str = self.date_var.get()
        if not date_str or p['df'] is None:
            return None
        mask = p['df']['DateTime'].dt.date.astype(str) == date_str
        d    = p['df'][mask].copy()
        return d if not d.empty else None

    def _redraw(self, idx):
        p   = self.panels[idx]
        fig = p['fig']
        fig.clear()

        day = self._day_df(idx)
        if day is None:
            p['canvas'].draw()
            return

        t = day['DateTime']

        # Collect active curve slots
        active = [
            (ci, var.get())
            for ci, var in enumerate(p['sel_vars'])
            if var.get() != '— none —' and var.get() in day.columns
        ]

        if not active:
            p['canvas'].draw()
            return

        n = len(active)

        # ── Adjust figure margins ─────────────────────────────────────
        right_margin = max(0.60, 0.93 - 0.08 * max(0, n - 1))
        fig.subplots_adjust(left=0.13, right=right_margin,
                            top=0.92, bottom=0.28)

        # ── Create axes ───────────────────────────────────────────────
        ax_main = fig.add_subplot(111)
        axes    = [ax_main]

        for k in range(1, n):
            ax_twin = ax_main.twinx()
            ax_twin.spines['right'].set_position(
                ('outward', AXIS_OFFSETS[k - 1]))
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

            line, = ax.plot(t, day[col], color=color,
                            linewidth=1.7, label=col)

            ax.set_ylabel(col, color=color, fontsize=7, labelpad=3)
            ax.tick_params(axis='y', colors=color, labelsize=7)
            ax.spines['right' if plot_idx > 0 else 'left'].set_color(color)

            try:
                ylo = float(p['ymin_vars'][ci].get())
                yhi = float(p['ymax_vars'][ci].get())
                if ylo < yhi:
                    ax.set_ylim(ylo, yhi)
            except ValueError:
                pass

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

        # ── X-axis, grid, title ───────────────────────────────────────
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax_main.grid(True, alpha=0.22, linestyle=':')
        fig.autofmt_xdate(rotation=30)

        title = f"{self.date_var.get()}  —  {p['file_var'].get()}"
        ax_main.set_title(title, fontsize=8, pad=4)

        if legend_handles:
            ax_main.legend(legend_handles, legend_labels,
                           loc='upper center',
                           bbox_to_anchor=(0.5, -0.22),
                           ncol=min(len(legend_handles), 3),
                           fontsize=7,
                           framealpha=0.88, edgecolor='#aaaaaa')

        p['canvas'].draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    DailyAnalysisApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
