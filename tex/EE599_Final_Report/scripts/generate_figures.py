#!/usr/bin/env python3
"""Generate publication-quality vector and raster figures for the EE599 Final Report."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 8.2,
    'axes.titlesize': 9.0,
    'axes.labelsize': 8.4,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 7.4,
    'ytick.labelsize': 7.4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
    'savefig.pad_inches': 0.02,
})

NAVY = '#1B4768'
TEAL = '#2E7D6F'
TERRA = '#B8502A'
SLATE = '#6B7280'
STEEL = '#6B94B3'
AMBER = '#C77D4E'
LIGHT = '#C4C9CE'


# 1. Main figure: matched-cost gain over frozen fixed frontier + pointwise 95% CIs (with x-dodge).
with (DATA / 'main_gain.csv').open(newline='', encoding='utf-8') as f:
    records = list(csv.DictReader(f))

fig, ax = plt.subplots(figsize=(7.05, 2.64))
styles = [
    ('Independent', SLATE, (0, (1.2, 1.8)), 'D', 1.55, -0.24),
    ('Waves', NAVY, '-', 'o', 2.0, -0.08),
    ('Campaign', TERRA, (0, (4.5, 2.0)), 's', 1.6, 0.08),
    ('Campaign + waves', TEAL, (0, (5.5, 1.8, 1.2, 1.8)), '^', 1.6, 0.24),
]

for name, color, ls, marker, lw, dx in styles:
    rs = [r for r in records if r['scenario'] == name]
    x = np.array([float(r['cost']) for r in rs]) + dx
    y = np.array([float(r['gain_pp']) for r in rs])
    seeds = np.array([int(r['train_seeds_covered']) for r in rs])
    has_ci = np.array([bool(r['ci_lower_pp'] and r['ci_upper_pp']) and int(r['train_seeds_covered']) == 3 for r in rs])

    # Connect points as a visual guide
    ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=4 if name == 'Waves' else 2)

    # Full-coverage points with 95% bootstrap error bars
    if has_ci.any():
        lo = np.array([float(r['ci_lower_pp']) for r, ok in zip(rs, has_ci) if ok])
        hi = np.array([float(r['ci_upper_pp']) for r, ok in zip(rs, has_ci) if ok])
        ax.errorbar(
            x[has_ci], y[has_ci],
            yerr=np.vstack([y[has_ci] - lo, hi - y[has_ci]]),
            fmt=marker, color=color, ms=4.2, capsize=2.3,
            elinewidth=0.95, capthick=0.85,
            zorder=5 if name == 'Waves' else 3,
            label=name,
        )

    # Partial-coverage points (e.g., 2/3 seeds): hollow marker, no CI whisker
    partial = ~has_ci
    if partial.any():
        ax.plot(
            x[partial], y[partial],
            linestyle='none', marker=marker, ms=4.9,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.35,
            zorder=6,
        )

ax.axhline(0, color='#686b70', lw=0.75, zorder=1)
ax.set(
    xlim=(3.35, 15.85),
    ylim=(-4.5, 29.0),
    xticks=[4, 8, 12, 15],
    ylabel='Gain over frozen fixed frontier (pp)',
    xlabel='Mean relative communication cost (multiples of two-payload reference)',
)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.grid(axis='y', color='#d6dade', linewidth=0.48, alpha=0.75)
ax.legend(
    loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=4, frameon=False,
    handlelength=2.2, columnspacing=1.25, handletextpad=0.45,
)
ax.annotate(
    '+23.2 pp', xy=(8 - 0.08, 23.2), xytext=(9.15, 26.8),
    fontsize=7.4, color=NAVY, weight='bold',
    arrowprops=dict(arrowstyle='-', lw=0.65, color=NAVY),
)
ax.annotate(
    '2/3 seeds\n(CI omitted)', xy=(15 + 0.24, 12.7), xytext=(13.25, 6.2),
    fontsize=6.9, color=TEAL, ha='center',
    arrowprops=dict(arrowstyle='->', lw=0.65, color=TEAL, shrinkA=1, shrinkB=3.5),
)
ax.annotate(
    '\u22121.2 pp', xy=(15 - 0.24, -1.2), xytext=(13.55, -3.55),
    fontsize=7.0, color=SLATE,
    arrowprops=dict(arrowstyle='-', lw=0.6, color=SLATE),
)
fig.subplots_adjust(left=0.088, right=0.992, bottom=0.22, top=0.83)
fig.savefig(OUT / 'main_gain.pdf', bbox_inches='tight')
fig.savefig(OUT / 'main_gain.png', bbox_inches='tight', dpi=220)
plt.close(fig)


# 2. Ablation figure: grouped bar chart across unordered scenarios (seed 2026, cost 8).
with (DATA / 'ablation_cost8.csv').open(newline='', encoding='utf-8') as f:
    ab = list(csv.DictReader(f))

labels = ['Independent', 'Waves', 'Campaign', 'Camp. + waves']
x = np.arange(len(labels))
width = 0.19

variants = [
    ('full', 'Full QR-DQN', NAVY, '#123047'),
    ('direct_k', 'Direct $k$ only', STEEL, '#4d7391'),
    ('without_estimator', 'No estimator', AMBER, '#9c5c33'),
    ('without_receipt', 'No receipt', LIGHT, '#6d747b'),
]

fig, ax = plt.subplots(figsize=(3.42, 2.48))
for idx, (field, label, fc, ec) in enumerate(variants):
    vals = [float(r[field]) for r in ab]
    offset = (idx - 1.5) * width
    ax.bar(
        x + offset, vals, width=width * 0.92,
        color=fc, edgecolor=ec, linewidth=0.55,
        label=label, zorder=3,
    )

ax.axhline(0, color='#686b70', lw=0.7, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=14, ha='right')
ax.set_ylabel('Gain over fixed at cost 8 (pp)')
ax.set_ylim(-1.0, 26.5)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.grid(axis='y', color='#d6dade', lw=0.45, zorder=1)
ax.legend(
    loc='upper center', bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False,
    fontsize=6.9, handlelength=1.4, columnspacing=0.95, handletextpad=0.4,
)
fig.subplots_adjust(left=0.16, right=0.985, top=0.79, bottom=0.23)
fig.savefig(OUT / 'ablation.pdf', bbox_inches='tight')
fig.savefig(OUT / 'ablation.png', bbox_inches='tight', dpi=220)
plt.close(fig)


# 3. Conceptual system diagram: engine + both feedback paths (immediate dissent & delayed receipt).
fig, ax = plt.subplots(figsize=(7.05, 1.84))
ax.set_xlim(0, 14.2)
ax.set_ylim(0, 3.55)
ax.axis('off')


def box(x0, y0, w, h, title, details, fc='#f6f8f9', ec='#a0aab0'):
    patch = FancyBboxPatch(
        (x0, y0), w, h, boxstyle='round,pad=0.10,rounding_size=0.15',
        linewidth=0.78, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x0 + w / 2, y0 + h * 0.68, title, fontsize=8.1, weight='bold',
            ha='center', va='center', color='#1e303d')
    ax.text(x0 + w / 2, y0 + h * 0.27, details, fontsize=7.2,
            ha='center', va='center', color='#46545e', linespacing=1.2)


def arrow(a, b, col='#59656c', style='-|>', lw=0.9):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=col, connectionstyle='arc3',
    ))


y_box = 1.72
box(0.20, y_box, 2.85, 1.18, 'Agent message', 'Length $L_t$; importance $w_t$')
box(3.75, y_box, 2.90, 1.18, 'QR-DQN controller', 'Estimator + feedback; price $\\lambda$')
box(7.30, y_box, 3.00, 1.18, 'DefenseEngine', 'Odd $k_t$; 1\u20132 payloads + hashes',
    fc='#eef4f8', ec=NAVY)
box(10.95, y_box, 3.00, 1.18, 'Hash-majority vote', 'Outcome $S_t$; dissent $d_t$')

for x1, x2 in [(3.15, 3.65), (6.75, 7.20), (10.40, 10.85)]:
    arrow((x1, y_box + 0.59), (x2, y_box + 0.59))

# Attack annotation above engine -> vote
ax.text(
    8.90, 3.30,
    'Simulated AiTM attacks poison selected routes before majority voting',
    ha='center', fontsize=7.35, color='#6a6d6f', style='italic',
)

# Feedback lane 1 (upper): immediate post-round minority vote dissent (when k >= 3)
ax.plot([12.20, 12.20, 5.45, 5.45], [1.61, 1.08, 1.08, 1.61],
        color=NAVY, lw=0.88, ls='-')
arrow((5.45, 1.24), (5.45, 1.61), col=NAVY, lw=0.88)
ax.text(
    8.825, 1.18,
    'Immediate vote dissent $d_{t-1}$ (when $k_{t-1}\\geq 3$)',
    fontsize=7.1, color=NAVY, ha='center', va='bottom',
)

# Feedback lane 2 (lower): one-round-delayed signed integrity receipt
ax.plot([12.85, 12.85, 4.80, 4.80], [1.61, 0.46, 0.46, 1.61],
        color='#6b7f8a', lw=0.85, ls=(0, (4, 2)))
arrow((4.80, 0.60), (4.80, 1.61), col='#6b7f8a', lw=0.85)
ax.text(
    8.825, 0.14,
    'Delayed signed integrity receipt $S_{t-2}$ (one extra round delay; assumed cost-free)',
    fontsize=7.1, color='#55626c', ha='center', va='bottom',
)

fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.99)
fig.savefig(OUT / 'architecture.pdf', bbox_inches='tight')
fig.savefig(OUT / 'architecture.png', bbox_inches='tight', dpi=220)
plt.close(fig)

print('Generated:', *(sorted(p.name for p in OUT.glob('*.pdf'))))
