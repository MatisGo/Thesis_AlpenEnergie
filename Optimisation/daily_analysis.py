"""
daily_analysis.py
=================
Interactive daily dashboard for hydro optimisation results.

Layout
------
  [Controls 1] | [Graph 1] || [Controls 2] | [Graph 2]
  ──────────────────────────────────────────────────────
  [Date selector — bottom left]

Each graph has 4 curve slots.  Each curve gets its own Y-axis (colour-coded).
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

RESULTS_FILE = 'optimised_results.csv'

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
        self.root.geometry("1560x800")
        self.root.minsize(960, 520)
        self.df     = None
        self.panels = []

        self._build_ui()
        self._load_data()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Bottom bar ────────────────────────────────────────────────
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
        """Build one (controls + figure) panel.  Returns state dict."""
        outer = tk.Frame(parent, bd=2, relief='groove')

        # ── Left control strip ────────────────────────────────────────
        ctrl = tk.Frame(outer, width=215, bg='#f0f0f0')
        ctrl.pack(side='left', fill='y', padx=2, pady=2)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text=f"  Graph {idx + 1}",
                 font=('Arial', 10, 'bold'), bg='#f0f0f0',
                 anchor='w').pack(fill='x', pady=(10, 4))

        sel_vars   = []
        sel_combos = []
        ymin_vars  = []
        ymax_vars  = []

        for j in range(N_CURVES):
            color = CURVE_COLORS[j]

            # ── Selector row ──────────────────────────────────────────
            top_row = tk.Frame(ctrl, bg='#f0f0f0')
            top_row.pack(fill='x', padx=6, pady=(6, 0))

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
            'outer':      outer,
            'sel_vars':   sel_vars,
            'sel_combos': sel_combos,
            'ymin_vars':  ymin_vars,
            'ymax_vars':  ymax_vars,
            'fig':        fig,
            'canvas':     canvas,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), RESULTS_FILE)

        if not os.path.exists(path):
            messagebox.showerror(
                "File not found",
                f"'{RESULTS_FILE}' not found.\nPlease run optimise.py first.")
            return

        self.df = pd.read_csv(path, parse_dates=['DateTime'])

        dates = sorted(
            self.df['DateTime'].dt.date.unique().astype(str).tolist())
        self.date_combo['values'] = dates
        if dates:
            self.date_var.set(dates[0])

        cols = ['— none —'] + [
            c for c in self.df.columns
            if c != 'DateTime'
            and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        for p in self.panels:
            for combo in p['sel_combos']:
                combo['values'] = cols

        self._refresh_all()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _refresh_all(self):
        self._redraw(0)
        self._redraw(1)

    def _day_df(self):
        date_str = self.date_var.get()
        if not date_str or self.df is None:
            return None
        mask = self.df['DateTime'].dt.date.astype(str) == date_str
        d    = self.df[mask].copy()
        return d if not d.empty else None

    def _redraw(self, idx):
        p   = self.panels[idx]
        fig = p['fig']
        fig.clear()

        day = self._day_df()
        if day is None:
            p['canvas'].draw()
            return

        t = day['DateTime']

        # Collect active curve slots (preserves slot index for colour/override)
        active = [
            (ci, var.get())
            for ci, var in enumerate(p['sel_vars'])
            if var.get() != '— none —' and var.get() in day.columns
        ]

        if not active:
            p['canvas'].draw()
            return

        n = len(active)

        # ── Adjust figure margins to make room for right-side axes ────
        # Each extra right axis needs ~0.08 of figure width
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

        # Hide right spine on main axis when twins are present
        if n > 1:
            ax_main.spines['right'].set_visible(False)

        # ── Plot each curve on its own axis ───────────────────────────
        legend_handles = []
        legend_labels  = []
        bound_added    = set()   # track which bounds were added to legend

        for plot_idx, (ci, col) in enumerate(active):
            ax    = axes[plot_idx]
            color = CURVE_COLORS[ci]

            line, = ax.plot(t, day[col], color=color,
                            linewidth=1.7, label=col)

            # Colour the Y-axis to match the curve
            ax.set_ylabel(col, color=color, fontsize=7, labelpad=3)
            ax.tick_params(axis='y', colors=color, labelsize=7)
            ax.spines['right' if plot_idx > 0 else 'left'].set_color(color)

            # Per-curve Y-axis override
            try:
                ylo = float(p['ymin_vars'][ci].get())
                yhi = float(p['ymax_vars'][ci].get())
                if ylo < yhi:
                    ax.set_ylim(ylo, yhi)
            except ValueError:
                pass

            # Auto Min / Max lines for reservoir level columns
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

        # ── X-axis and grid (main axis only) ─────────────────────────
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax_main.grid(True, alpha=0.22, linestyle=':')
        fig.autofmt_xdate(rotation=30)

        ax_main.set_title(self.date_var.get(), fontsize=9, pad=4)

        # ── Combined legend — placed below the axes ───────────────────
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
