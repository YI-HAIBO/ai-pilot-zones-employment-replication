# -*- coding: utf-8 -*-
"""Generate English-label figures for the English manuscript:
Figure 1: event-study dynamic effects (gdp_growth, Table 3 data)
Figure 2: treated vs. control mean per-capita disposable income trend (panel)
Output: manuscript_en/figures/*.png (300 dpi, journal size)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(R, 'manuscript_en', 'figures')
os.makedirs(FIG, exist_ok=True)

# No embedded titles in figures: journal captions are in the manuscript text (Figure 1/Figure 2).
# ---- Figure 1: event-study dynamic effects (gdp_growth, Table 3) ----
e = np.array([-4, -3, -2, 0, 1, 2, 3, 4, 5, 6])
b = np.array([0.1565, 0.4482, -0.0218, -0.3531, -0.2374, -0.1731, -0.4960, -0.3742, 0.0315, 0.3516])
lo = np.array([-1.233, -0.096, -0.369, -1.501, -1.141, -0.939, -1.200, -1.044, -0.606, -0.394])
hi = np.array([1.546, 0.992, 0.326, 0.795, 0.666, 0.593, 0.208, 0.296, 0.669, 1.098])
err = np.vstack([b - lo, hi - b])

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.errorbar([-1], [0.0], yerr=[[0.0], [0.0]], fmt='o', color='0.55',
            markersize=7, capsize=4, zorder=3, label='Reference period (e=-1)')
ax.errorbar(e[e < 0], b[e < 0], yerr=err[:, e < 0], fmt='o', color='#1f77b4',
            markersize=7, capsize=4, lw=1.6, zorder=3, label='Pre-policy (e<0)')
ax.errorbar(e[e >= 0], b[e >= 0], yerr=err[:, e >= 0], fmt='o', color='#d62728',
            markersize=7, capsize=4, lw=1.6, zorder=3, label='Post-policy (e\u22650)')
ax.axhline(0, color='0.4', ls='--', lw=1.0, zorder=1)
ax.axvline(-0.5, color='0.3', ls=':', lw=1.2, zorder=1)
ax.text(-0.5, ax.get_ylim()[1] * 0.92, 'Policy', ha='center', va='top',
        fontsize=9, color='0.25')
ax.set_xlabel('Relative year to policy (e)', fontsize=11)
ax.set_ylabel('Coefficient (percentage points)', fontsize=11)
ax.set_xticks(list(range(-4, 7)))
ax.set_xticklabels([str(x) for x in range(-4, 7)], fontsize=10)
ax.set_xlim(-4.6, 6.6)
ax.axvspan(-4.5, -1.5, color='#f0f0f0', alpha=0.4, zorder=0)
ax.legend(fontsize=8.5, frameon=False, loc='lower left')
ax.grid(axis='y', ls=':', alpha=0.4)
ax.set_ylim(-1.8, 1.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig1_event_study.png'), bbox_inches='tight')
plt.close(fig)
print('Figure 1 saved')

# ---- Figure 2: treated vs. control mean income trend (income_yuan) ----
p = pd.read_csv(os.path.join(R, 'data', 'did_city_panel_2017_2025_v8.csv'), encoding='utf-8-sig')
p['treat_lab'] = np.where(p['treat'] == 1, 'Treated (18 cities)', 'Control (277 cities)')
mean = p.dropna(subset=['income_yuan']).groupby(['year', 'treat_lab'])['income_yuan'].mean().unstack()

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(mean.index, mean['Treated (18 cities)'], 'o-', color='#d62728', lw=2.0, ms=5,
        label='Treated (18 cities)')
ax.plot(mean.index, mean['Control (277 cities)'], 's--', color='#1f77b4', lw=1.8, ms=4.5,
        label='Control (277 cities)')
ax.axvline(2019, color='0.3', ls=':', lw=1.2)
ax.text(2019.05, ax.get_ylim()[1] * 0.95, '2019 first batch', fontsize=9, color='0.25')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Per capita disposable income (CNY)', fontsize=11)
ax.set_xticks(list(range(2017, 2026)))
ax.set_xticklabels([str(x) for x in range(2017, 2026)], fontsize=9)
ax.legend(fontsize=9, frameon=False, loc='lower right')
ax.grid(axis='y', ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig2_income_trend.png'), bbox_inches='tight')
plt.close(fig)
print('Figure 2 saved')

print()
print('Figure 2 data (treated/control mean income, CNY):')
print(mean.round(0).to_string())
