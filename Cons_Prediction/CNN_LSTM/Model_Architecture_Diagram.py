"""
Model_Architecture_Diagram.py
==============================
Publication-ready block diagram of the CNN-LSTM encoder-decoder
architecture used for 48h load forecasting.

Output: Output Forecast/Model_Architecture.png

Usage:  python Model_Architecture_Diagram.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'Output Forecast',
                           'Model_Architecture.png')
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_IN   = '#2166AC'   # blue        — data inputs
C_CNN  = '#4DAC26'   # green       — CNN blocks
C_LSTM = '#B2182B'   # red         — LSTM blocks
C_DEC  = '#762A83'   # purple      — decoder processing
C_EXT  = '#E08214'   # orange      — additional decoder inputs
C_OUT  = '#1A9641'   # dark green  — output
C_ARC  = '#B2182B'   # arc arrows  (h_T)

BH = 0.68   # standard box height

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def box(ax, cx, cy, w, line1, line2='', line3='', fc=C_IN):
    """Draw a rounded box centred at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - BH / 2), w, BH,
        boxstyle='round,pad=0.07',
        facecolor=fc, edgecolor='#1a1a1a', linewidth=1.3, zorder=3)
    ax.add_patch(patch)

    n = sum(bool(t) for t in [line1, line2, line3])
    offsets = {1: [0.0], 2: [0.17, -0.17], 3: [0.26, 0.0, -0.26]}
    offs = offsets[n]

    texts = [t for t in [line1, line2, line3] if t]
    sizes = [8.5, 7.0, 6.2][:n]

    for i, (txt, dy, sz) in enumerate(zip(texts, offs, sizes)):
        weight = 'bold' if i == 0 else 'normal'
        style  = 'italic' if i > 0 else 'normal'
        ax.text(cx, cy + dy, txt, ha='center', va='center',
                fontsize=sz, fontweight=weight, fontstyle=style,
                color='white', zorder=4)


def arr(ax, x1, y1, x2, y2, color='#222222', lw=1.5, rad=0.0, ls='solid'):
    cs = f'arc3,rad={rad}'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    connectionstyle=cs,
                    linestyle=ls),
                zorder=5)


def bg(ax, x0, y0, w, h, fc, ec='#aaaaaa'):
    patch = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.15',
        facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=1, alpha=0.45)
    ax.add_patch(patch)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(1.2, 10.8)
ax.axis('off')
fig.patch.set_facecolor('white')

# ---------------------------------------------------------------------------
# Section backgrounds
# ---------------------------------------------------------------------------
bg(ax, 0.15, 2.2,  4.65, 7.85, fc='#DDEEFF', ec='#2166AC')   # encoder
bg(ax, 5.5,  1.5,  9.35, 8.65, fc='#F0E6F8', ec='#762A83')   # decoder

ax.text(2.48, 10.35, 'ENCODER', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color='#154A7A')
ax.text(10.18, 10.35, 'DECODER', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color='#4A1570')

# ---------------------------------------------------------------------------
# ENCODER blocks  (centred at x = 2.48)
# ---------------------------------------------------------------------------
EX, EW = 2.48, 4.2

box(ax, EX, 9.4, EW,
    'Historical Window  —  Encoder Input',
    '12 features  ×  2 016 timesteps  (7-day lookback, 5-min)',
    fc=C_IN)

box(ax, EX, 8.1, EW,
    'Conv1D  (64 filters, kernel=5)  +  BN  +  MaxPool(÷4)',
    '2 016  →  504 timesteps  ×  64 channels',
    fc=C_CNN)

box(ax, EX, 6.8, EW,
    'Conv1D  (128 filters, kernel=5)  +  BN  +  MaxPool(÷4)',
    '504  →  126 timesteps  ×  128 channels',
    fc=C_CNN)

box(ax, EX, 5.5, EW,
    'Conv1D  (256 filters, kernel=3)  +  BN  +  MaxPool(÷2)',
    '126  →  63 timesteps  ×  256 channels',
    fc=C_CNN)

box(ax, EX, 4.2, EW,
    'LSTM  (128 units)',
    'return_sequences = True',
    fc=C_LSTM)

box(ax, EX, 2.9, EW,
    'LSTM  (64 units)   →   h_T',
    'Context vector   |   dim = 64',
    fc=C_LSTM)

# Encoder downward arrows + Dropout labels
enc_pairs = [(9.06, 8.44), (7.76, 7.14), (6.46, 5.84), (5.16, 4.54), (3.86, 3.24)]
drop_y    = [8.75, 7.45, 6.15, 4.85]

for (y1, y2) in enc_pairs:
    arr(ax, EX, y1, EX, y2)
for dy in drop_y:
    ax.text(EX + 2.25, dy, 'Dropout (p = 0.2)',
            ha='left', va='center', fontsize=5.8,
            color='#555555', style='italic', zorder=4)

# ---------------------------------------------------------------------------
# DECODER — additional inputs
# ---------------------------------------------------------------------------
box(ax, 8.2,  9.4, 2.8,
    'Statistical Profiles',
    '8 past same-weekday profiles',
    '576 steps  ×  8 weeks',
    fc=C_EXT)

box(ax, 13.2, 9.4, 3.0,
    'PV Production Estimate',
    'Weather-based MLP correction',
    '576 steps  ×  1',
    fc=C_EXT)

# ---------------------------------------------------------------------------
# DECODER — processing blocks
# ---------------------------------------------------------------------------
box(ax, 8.2,  7.3, 2.8,
    'Cross-Attention',
    'Query: h_T   |   Keys & Values: Stat Profiles',
    'Output:  weighted profile  (576 × 1)',
    fc=C_DEC)

box(ax, 12.0, 7.3, 3.5,
    'Dense (256, ReLU)  +  RepeatVector (48)',
    'Projects h_T into 48 hourly decoder slots',
    fc=C_DEC)

box(ax, 10.1, 5.2, 5.5,
    'Concatenate',
    '[ Context (Dense+Repeat)  |  Attention Output  |  PV Estimate ]',
    fc=C_DEC)

box(ax, 10.1, 3.7, 4.5,
    'TimeDistributed  Dense (1)',
    'Independent linear prediction per 5-min step',
    fc=C_DEC)

box(ax, 10.1, 2.2, 4.5,
    '48h Load Forecast',
    '576 steps  ×  5-min  =  48 hours',
    fc=C_OUT)

# ---------------------------------------------------------------------------
# Arrows — decoder inputs to processing
# ---------------------------------------------------------------------------
# Stat Profiles → Cross-Attention
arr(ax, 8.2, 9.06, 8.2, 7.64, color=C_EXT)

# PV → Concatenate  (diagonal, right to left)
arr(ax, 13.2, 9.06, 12.75, 5.54, color=C_EXT, rad=0.18)

# Cross-Attention → Concatenate (diagonal)
arr(ax, 8.2, 6.96, 7.35, 5.54)

# Dense+Repeat → Concatenate (diagonal)
arr(ax, 12.0, 6.96, 12.85, 5.54)

# Concatenate → TD Dense
arr(ax, 10.1, 4.86, 10.1, 4.04)

# TD Dense → Output
arr(ax, 10.1, 3.36, 10.1, 2.54)

# ---------------------------------------------------------------------------
# h_T arcs  (encoder → decoder)
# ---------------------------------------------------------------------------
# h_T → Dense+RepeatVector  (main context path, solid)
arr(ax, 4.58, 2.9, 10.25, 7.3,
    color=C_ARC, lw=2.0, rad=-0.22)

# h_T → Cross-Attention  (as query, dashed)
arr(ax, 4.58, 2.9, 6.8, 7.3,
    color=C_ARC, lw=1.5, rad=-0.12, ls='dashed')

# h_T label on the arc
ax.text(5.55, 5.5, 'h_T',
        ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_ARC,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=C_ARC, linewidth=1.5, alpha=0.95),
        zorder=6)

# "Query" label on dashed arc
ax.text(5.3, 4.5, '(query)', ha='center', va='center',
        fontsize=6.5, color=C_ARC, style='italic', zorder=6)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_items = [
    mpatches.Patch(fc=C_IN,   label='Data Input'),
    mpatches.Patch(fc=C_CNN,  label='CNN Block'),
    mpatches.Patch(fc=C_LSTM, label='LSTM Block'),
    mpatches.Patch(fc=C_EXT,  label='Decoder Input'),
    mpatches.Patch(fc=C_DEC,  label='Decoder Processing'),
    mpatches.Patch(fc=C_OUT,  label='Output'),
]
ax.legend(handles=legend_items, loc='lower left',
          fontsize=7.5, framealpha=0.92,
          edgecolor='#aaaaaa', ncol=3,
          bbox_to_anchor=(0.01, 0.0))

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
fig.suptitle(
    'CNN-LSTM Encoder–Decoder Architecture for 48h Load Forecasting',
    fontsize=12, fontweight='bold', y=0.99)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout(pad=0.5)
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved to: {OUTPUT_PATH}')
plt.show()