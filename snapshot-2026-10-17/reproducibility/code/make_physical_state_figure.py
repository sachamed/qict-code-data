#!/usr/bin/env python3
"""Publication-size comparison of physical fields and finite C9 diagnostics."""
from publication_io import save_figure_atomic
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'paper/figures'


def main():
    p=ROOT/'reproducibility/results/physical_two_time/PHYSICAL_TWO_TIME_LEPTON_KERNELS.npz'
    c=ROOT/'reproducibility/results/finite_C9_common_state/FINITE_C9_COMMON_STATE_CERTIFICATE.json'
    rows=json.loads(c.read_text())['representation_cutoffs']
    size_mm=[178,86]
    with np.load(p,allow_pickle=False) as z:
        K=abs(z['physical_kernel'][:,0,0,0]);C=abs(z['fixed_HS_comparison_moments'][:,0,0])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
        'axes.labelsize':9,'axes.linewidth':.7,'xtick.labelsize':8.5,
        'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none',
        'text.color':'#111111','axes.labelcolor':'#111111','axes.edgecolor':'#404040'})
    fig,ax=plt.subplots(1,2,figsize=(178/25.4,86/25.4))
    fig.subplots_adjust(left=.09,right=.97,bottom=.2,top=.83,wspace=.42)
    n=np.arange(7)
    ax[0].plot(n,K,'o-',color='#1F6F5F',lw=1.6,ms=4,label='Physical K')
    ax[0].plot(n,C,'s--',color='#6F6F6F',lw=1.1,ms=4,label='Fixed GNS C')
    ax[0].set(xlabel='Microscopic time n',ylabel='Left echo modulus',
        xticks=[0,2,4,6],ylim=(0,1.04),yticks=[0,.25,.5,.75,1])
    ax[0].legend(frameon=False,loc='lower left',fontsize=8.5)
    rr=np.array([x['radius'] for x in rows[1:]])
    change=np.array([x['successive_regulator_moment_change_diagnostic'] for x in rows[1:]])
    ax[1].plot(rr,change,'o-',color='#1F6F5F',lw=1.6,ms=4)
    ax[1].set_yscale('log')
    ax[1].set(xlabel='Representation radius R',ylabel='C9 moment change (log scale)',
        xticks=[2,3,4],ylim=(.0006,.25),yticks=[.001,.01,.1])
    for x,y in zip(rr,change):
        ax[1].annotate(f'{y:.3g}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    for a,title in zip(ax,['(a) Nonstationary physical field','(b) Finite stationary C9 states']):
        a.set_title(title,loc='left',fontsize=9.5,pad=12)
        a.spines[['top','right']].set_visible(False)
        a.tick_params(width=.7,length=3)
        a.grid(axis='y',which='major',color='#E8E8E5',lw=.5)
        a.minorticks_off()
    FIG.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','svg','png'):
        save_figure_atomic(fig, FIG/f'physical_two_time_state_selection.{extension}',dpi=240)
    plt.close(fig)
    print(json.dumps({'classification':'FIGURE_GENERATED_FROM_RECORDED_DATA','size_mm':size_mm}))


if __name__=='__main__':main()
