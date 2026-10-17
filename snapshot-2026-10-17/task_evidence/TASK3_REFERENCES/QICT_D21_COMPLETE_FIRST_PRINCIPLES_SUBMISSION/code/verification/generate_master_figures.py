#!/usr/bin/env python3
from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)

# Figure 1: claim/dependency architecture
nodes=[
 ('C1-C7','finite / free\noperator core'),('C8','exact unitary\ngauge dynamics'),
 ('C8J','Julia matter-receiver\ncolligation'),('C9','canonical ergodic\nstationary state'),
 ('C10','3D Gauss\ndressing control'),('C11','intrinsic relative-moment\nobservables'),
 ('3D edge','C11 directed-regulator\nspectral support'),('Physical scale','independent receiver\nclock observable')]
fig,ax=plt.subplots(figsize=(12.8,5.0))
ax.axis('off')
coords=[(0.02,0.62),(0.26,0.62),(0.50,0.62),(0.74,0.62),(0.74,0.16),(0.50,0.16),(0.26,0.16),(0.02,0.16)]
for (lab,sub),(x,y) in zip(nodes,coords):
    box=FancyBboxPatch((x,y),0.20,0.20,boxstyle='round,pad=0.012,rounding_size=0.018',fill=False,linewidth=1.4)
    ax.add_patch(box); ax.text(x+0.10,y+0.13,lab,ha='center',va='center',fontsize=11,fontweight='bold'); ax.text(x+0.10,y+0.065,sub,ha='center',va='center',fontsize=8.5)
for i in range(3):
    x1,y1=coords[i];x2,y2=coords[i+1]
    ax.annotate('',xy=(x2,y2+0.10),xytext=(x1+0.20,y1+0.10),arrowprops=dict(arrowstyle='->',lw=1.2))
# vertical down and reverse row arrows
x1,y1=coords[3];x2,y2=coords[4]
ax.annotate('',xy=(x2+0.10,y2+0.20),xytext=(x1+0.10,y1),arrowprops=dict(arrowstyle='->',lw=1.2))
for i in range(4,7):
    x1,y1=coords[i];x2,y2=coords[i+1]
    ax.annotate('',xy=(x2+0.20,y2+0.10),xytext=(x1,y1+0.10),arrowprops=dict(arrowstyle='->',lw=1.2))
ax.text(0.50,0.02,'Exact microscopic dynamics, C11 spectral observables, and the independent receiver clock are separated by the physical-identification map; no finite surrogate crosses this boundary.',ha='center',fontsize=9)
fig.tight_layout()
for ext,dpi in [('pdf',None),('png',600)]:
    fig.savefig(FIG/f'qict_logical_architecture.{ext}',dpi=dpi,bbox_inches='tight')
plt.close(fig)

# Figure 2: 96x96 eigenvalue pairing as finite witness only
csvp=ROOT/'data'/'df96_spectrum_pairing.csv'
vals=[]
if csvp.exists():
    with csvp.open() as f:
        rows=list(csv.DictReader(f))
    # tolerate alternative column names
    for r in rows:
        for k in ('eigenvalue_GeV','eigenvalue','lambda'):
            if k in r and r[k] != '':
                vals.append(float(r[k])); break
if not vals:
    R=np.loadtxt(ROOT/'code'/'finite_representatives'/'DF96_real.csv',delimiter=',')
    I=np.loadtxt(ROOT/'code'/'finite_representatives'/'DF96_imag.csv',delimiter=',')
    vals=np.linalg.eigvalsh(R+1j*I).real.tolist()
vals=np.sort(np.asarray(vals))
fig,ax=plt.subplots(figsize=(8.8,5.1))
ax.plot(np.arange(1,len(vals)+1),vals,'o',ms=3.2)
ax.axhline(0,lw=0.8)
ax.set_xlabel('Ordered eigenvalue index')
ax.set_ylabel('Auxiliary finite-witness eigenvalue (reference normalization)')
ax.set_title(r'$96\times96$ finite Dirac witness: exact $\pm\lambda$ pairing')
ax.text(0.5,-0.18,'Auxiliary finite-spectrum consistency witness; this normalization is distinct from the C11 physical mass observable.',transform=ax.transAxes,ha='center',fontsize=9)
fig.tight_layout()
for ext,dpi in [('pdf',None),('png',600)]: fig.savefig(FIG/f'df96_spectrum_pairing.{ext}',dpi=dpi,bbox_inches='tight')
plt.close(fig)

print('PASS: generated active master figures (vector PDF + 600 dpi PNG)')
