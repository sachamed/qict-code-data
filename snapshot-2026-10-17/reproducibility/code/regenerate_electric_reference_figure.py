#!/usr/bin/env python3
"""Regenerate the oriented electric reference curve from its declared moments.

The finite electric substep is a reference calculation, separate from physical
charged C10 spectroscopy. This implementation follows electric_moment_stable
and mod21_distribution in the retained compute_charged_sector_controls.py.
"""
from publication_io import save_figure_atomic
from pathlib import Path
import ast,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext,NullLocator

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT.parent
SOURCE=ROOT/'task_evidence/TASK3_REFERENCES/QICT_D21_COMPLETE_FIRST_PRINCIPLES_SUBMISSION/code/finite_representatives/compute_charged_sector_controls.py'
OUT=ROOT/'reproducibility/results/electric_reference_figure'
OUT.mkdir(parents=True,exist_ok=True)

def moment(r,n):
    if n%21==0:return 1+0j
    theta=4*np.pi*n/21
    t=np.exp(-1j*theta)
    c=np.cos(theta/2)*np.exp(-1j*theta/2)
    a=2.0**(-r) if r<1024 else 0.
    b=4.0**(-r) if r<512 else 0.
    return np.exp(-1j*22*np.pi*r*n/21)*((6-12*a)*c**r+4*b*(1+t**r))/(6-12*a+8*b)

tree=ast.parse(SOURCE.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['electric_moment_stable','mod21_distribution']]
reference={'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),reference)
radii=sorted(set(range(1,65))|{80,100,128,160,200,256,320,384,500,640,768,1000,1250,1500,1750,2000})
rows=[];errors=[]
for r in radii:
    hats=np.array([moment(r,n) for n in range(21)])
    old=np.array([reference['electric_moment_stable'](r,n) for n in range(21)])
    errors.append(float(np.max(np.abs(hats-old))))
    p=np.fft.ifft(hats).real
    assert p.min()>-2e-14 and abs(p.sum()-1)<2e-14
    rows.append({'R':r,'total_variation_to_uniform_21roots':float(.5*np.sum(np.abs(p-1/21))),
                 'max_nontrivial_fourier_moment_abs':float(np.max(np.abs(hats[1:]))),
                 'min_weight':float(p.min()),'max_weight':float(p.max())})
assert max(errors)<2e-14
with (OUT/'electric_reference_convergence.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
 'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,
 'axes.linewidth':.65,'axes.edgecolor':'#5E6462','text.color':'#171717',
 'axes.labelcolor':'#171717','xtick.color':'#5E6462','ytick.color':'#5E6462'})
fig,ax=plt.subplots(figsize=(125.56/25.4,79/25.4))
fig.subplots_adjust(left=.17,right=.94,bottom=.18,top=.86)
x=[r['R'] for r in rows]
ax.plot(x,[r['total_variation_to_uniform_21roots'] for r in rows],color='#1F6F5F',lw=1.6,label='Total-variation distance')
ax.plot(x,[r['max_nontrivial_fourier_moment_abs'] for r in rows],color='#5E6462',lw=1.1,ls='--',label='Largest nontrivial moment')
ax.set_yscale('log');ax.set_ylim(5e-11,1.3);ax.set_xlim(0,2000)
ax.set_xticks([0,500,1000,1500,2000])
ax.set_yticks([1,1e-3,1e-6,1e-9]);ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel('BCC dressing radius R')
ax.set_ylabel('Convergence measure (log scale)',labelpad=7)
ax.set_title('Oriented electric reference approaches uniformity',loc='left',fontsize=9,pad=10)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(width=.65,length=3)
ax.legend(loc='upper right',frameon=False,fontsize=8.2,handlelength=2.6)
target=PACKAGE/'02_Supplementary_Information/figures/charged_electric_largeR_convergence.pdf'
save_figure_atomic(fig, target)
save_figure_atomic(fig, OUT/'electric_reference_convergence.png',dpi=300)
plt.close(fig)
record={'classification':'DETERMINISTIC_ELECTRIC_REFERENCE_FIGURE',
 'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'all_80_radii_match_source_moments':max(errors)<2e-14,'maximum_moment_difference':max(errors),
 'plotted_quantity_order':['total variation to uniform 21-root measure','largest nontrivial Fourier moment'],
 'domain':'oriented electric-substep reference; not a physical C10 or mass spectrum',
 'uncertainty':'deterministic binary64 evaluation; no statistical interval',
 'target_dimensions_mm':[125.56,79],'style':'white field; teal focal line; neutral dashed comparison; embedded fonts',
 'figure_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(OUT/'FIGURE_REGENERATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
