#!/usr/bin/env python3
"""Rebuild publication-quality vector plots from the supplied final(3).md aggregate tables.

IMPORTANT: These CSVs are transcription of the user's reported aggregate results;
this is not an experiment runner and cannot regenerate raw trials or CI estimates.
"""
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
    'font.family':'serif', 'font.serif':['DejaVu Serif','Times New Roman'],
    'font.size':8.2, 'axes.titlesize':9, 'axes.labelsize':8.4,
    'legend.fontsize':7.6, 'xtick.labelsize':7.4, 'ytick.labelsize':7.4,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.linewidth':0.7, 'savefig.pad_inches':0.02,
})
NAVY = '#1B4768'
MID = '#536673'
GRAY = '#6E7075'
PALE = '#9B9A97'

# 1. Main figure: original reported gain + pointwise 95% CIs. No interpolation beyond points.
with (DATA/'main_gain.csv').open(newline='') as f:
    records = list(csv.DictReader(f))
fig, ax = plt.subplots(figsize=(7.05, 2.66))
styles = [
    ('Independent', '#62666a', (0,(1,2.1)), 'D', 1.55),
    ('Waves', NAVY, '-', 'o', 2.0),
    ('Campaign', '#22292f', (0,(5,2.3)), 's', 1.55),
    ('Campaign + waves', '#92989c', (0,(6,2,1,2)), '^', 1.55),
]
for name,color,ls,marker,lw in styles:
    rs=[r for r in records if r['scenario']==name]
    x = np.array([float(r['cost']) for r in rs]); y = np.array([float(r['gain_pp']) for r in rs])
    lo=np.array([float(r['ci_lower_pp']) for r in rs]); hi=np.array([float(r['ci_upper_pp']) for r in rs])
    ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]), label=name, c=color,
                ls=ls, marker=marker, ms=4.0, lw=lw, capsize=2.35,
                elinewidth=0.9, capthick=0.8, zorder=5 if name=='Waves' else 3)
ax.axhline(0, color='#686b70', lw=0.7)
ax.set(xlim=(3.45,15.60),ylim=(-4.3,29),xticks=[4,8,12,15],
       ylabel='Gain over frozen fixed frontier (pp)',
       xlabel='Mean relative communication cost (multiples of two-payload reference)')
ax.set_yticks([0,5,10,15,20,25]); ax.grid(axis='y',color='#d6dade',linewidth=0.48,alpha=0.75)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.205),ncol=4,frameon=False,
          handlelength=2.45,columnspacing=1.3,handletextpad=0.45)
ax.annotate('+23.2 pp', xy=(8,23.2), xytext=(9.0,27.0),fontsize=7.4,color=NAVY,
            arrowprops=dict(arrowstyle='-',lw=0.65,color=NAVY))
# incomplete coverage of campaign+waves at cost15 clearly marked in caption in LaTeX
fig.subplots_adjust(left=.088,right=.992,bottom=.24,top=.82)
fig.savefig(OUT/'main_gain.pdf',bbox_inches='tight')
fig.savefig(OUT/'main_gain.png',bbox_inches='tight',dpi=200)
plt.close(fig)

# 2. Ablation: difference between main complete baseline and three variants, all seed 2026.
with (DATA/'ablation_cost8.csv').open(newline='') as f:
    ab=list(csv.DictReader(f))
labels=['Independent','Waves','Campaign','Camp. + waves']
x=np.arange(4)
fig,ax=plt.subplots(figsize=(3.35,2.42))
for field,label,color,ls,mark,lw in [
    ('full','Full QR-DQN',NAVY,'-','o',1.85),
    ('without_receipt','No receipt','#737b81',(0,(5,2)),'s',1.2),
    ('without_estimator','No estimator','#4c5358',(0,(1,2)),'^',1.2),
    ('direct_k','Direct $k$ only','#9d9fa2',(0,(5,2,1,2)),'D',1.2),
]:
    y=[float(r[field]) for r in ab]
    ax.plot(x,y,ls=ls,color=color,marker=mark,ms=3.5,lw=lw,label=label)
ax.set_xticks(x,labels,rotation=18,ha='right')
ax.set_ylabel('Gain over fixed at cost 8 (pp)')
ax.set_ylim(-1.2,27); ax.set_yticks([0,5,10,15,20,25])
ax.grid(axis='y',color='#d6dade',lw=.45)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.24),ncol=2,frameon=False,
          fontsize=6.7,handlelength=2.25,columnspacing=.9)
fig.subplots_adjust(left=.16,right=.98,top=.78,bottom=.27)
fig.savefig(OUT/'ablation.pdf',bbox_inches='tight')
fig.savefig(OUT/'ablation.png',bbox_inches='tight',dpi=200)
plt.close(fig)

# 3. Conceptual (not measured) system diagram, vector; intentionally separate experiments.
fig,ax=plt.subplots(figsize=(7.05,1.65))
ax.set_xlim(0,14.2); ax.set_ylim(0,3.15); ax.axis('off')

def box(x,y,w,h,title,details,fc='#f6f8f9',ec='#a0aab0'):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.15',
                         linewidth=.75,facecolor=fc,edgecolor=ec)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h*.68,title,fontsize=8.2,weight='bold',ha='center',va='center',color='#1e303d')
    ax.text(x+w/2,y+h*.27,details,fontsize=7.3,ha='center',va='center',color='#46545e',linespacing=1.2)

def arrow(a,b,col='#59656c',style='-|>',lw=.9):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=9,
                                 linewidth=lw,color=col,connectionstyle='arc3'))
box(.2,1.24,2.9,1.2,'Agent message','Length $L$; importance $w$')
box(3.8,1.24,2.8,1.2,'QR-DQN controller','Observed feedback; price $\\lambda$')
box(7.25,1.24,3.0,1.2,'DefenseEngine','Odd $k$; 1--2 payloads + hashes',fc='#eef4f8',ec=NAVY)
box(10.95,1.24,3.0,1.2,'Hash-majority vote','Integrity outcome $S_t$')
for x1,x2 in [(3.19,3.7),(6.68,7.15),(10.34,10.84)]: arrow((x1,1.85),(x2,1.85))
# Two annotation lanes: only receipt is returned; simulation attacker influences hashes.
ax.plot([12.6,12.6,5.22,5.22],[1.13,.55,.55,1.13],color='#6b7f8a',lw=.82,ls=(0,(4,2)))
arrow((5.22,.66),(5.22,1.13),col='#6b7f8a',lw=.8)
ax.text(8.6,.29,'One-round-delayed signed integrity receipt (assumed cost-free)',
        fontsize=7.25,color='#5d6973',ha='center')
ax.text(8.74,2.8,'Synthetic AiTM attacks corrupt selected transmissions before voting',
        ha='center',fontsize=7.4,color='#6a6d6f',style='italic')
fig.subplots_adjust(left=0,right=1,bottom=.07,top=.99)
fig.savefig(OUT/'architecture.pdf',bbox_inches='tight')
fig.savefig(OUT/'architecture.png',bbox_inches='tight',dpi=200)
plt.close(fig)
print('Generated:',*(p.name for p in OUT.glob('*.pdf')))
