#!/usr/bin/env python3
"""Publication-size plot of exact normalized electric-cluster trace distances."""
from publication_io import save_figure_atomic
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
record = json.loads((ROOT/'reproducibility/results/flux_kato/FLUX_KATO_OBSTRUCTION_CERTIFICATE.json').read_text())
rows = record['nested_seed_riesz_states']['successive_trace_distances']
radii = [r['radii'][0] for r in rows]
values = [r['trace_distance'] for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                     'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(178/25.4,73/25.4),facecolor='white')
fig.subplots_adjust(left=.13,right=.95,bottom=.24,top=.80)
ax.plot(range(4),values,'o-',color='#1F6F5F',lw=1.6,ms=4.5)
ax.axhline(.5,color='#5E6462',linestyle='--',lw=.8)
ax.set_xticks(range(4),[str(x) for x in radii])
ax.set_xlim(-.2,3.25)
ax.set_ylim(0,.58)
ax.set_yticks([0,.25,.5])
ax.set_xlabel('Lower flux cutoff R (comparison with 2R)')
ax.set_ylabel('Trace distance')
ax.spines[['top','right']].set_visible(False)
for key in ('bottom','left'):ax.spines[key].set_linewidth(.6)
ax.tick_params(width=.6,length=3)
for i,value in enumerate(values):
    ax.annotate(f'{value:.4f}',(i,value),xytext=(0,-17),textcoords='offset points',
                ha='center',fontsize=8.5,color='#171717')
ax.text(3.17,.513,'Limit 1/2',ha='right',fontsize=8.5,color='#5E6462')
fig.text(.13,.925,'Normalized Floquet clusters fail the Cauchy test',fontsize=10,weight='semibold')
fig.text(.13,.853,'Exact electric seed on one authentic oriented L24 plaquette',fontsize=8.4,color='#5E6462')
path=ROOT/'paper/figures/flux_kato_cauchy_audit'
save_figure_atomic(fig, path.with_suffix('.pdf'),facecolor='white')
save_figure_atomic(fig, path.with_suffix('.png'),dpi=220,facecolor='white')
plt.close(fig)
print(path.with_suffix('.pdf'))
