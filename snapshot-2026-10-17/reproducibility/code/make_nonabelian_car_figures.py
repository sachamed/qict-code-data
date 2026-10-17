#!/usr/bin/env python3
"""Regenerate exact-data compact-group CAR and boundary-coherence figures."""
from publication_io import save_figure_atomic
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'reproducibility/results/nonabelian_common_car'
FIGURES=ROOT/'paper/figures'


def main():
    c=json.loads((RESULTS/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json').read_text())
    with np.load(RESULTS/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz',allow_pickle=False) as z:
        moments=z['L3_gauge_Cn6']
    bounds=np.array([r['analytic_state_and_flux_error_upper'] for r in c['cutoff_results']])
    changes=np.array([np.nan if r['moment_successive_cutoff_operator_norm_change'] is None
                      else r['moment_successive_cutoff_operator_norm_change'] for r in c['cutoff_results']])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'axes.titlesize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                         'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,
                         'text.color':'#111111','axes.labelcolor':'#111111'})
    fig,axes=plt.subplots(1,2,figsize=(178/25.4,83/25.4),gridspec_kw={'wspace':.37})
    x=np.arange(13)
    axes[0].plot(x,abs(moments[:,0,0]),color='#1F6F5F',lw=1.6,marker='o',ms=3.5,label='Left module')
    axes[0].plot(x,abs(moments[:,3,3]),color='#404040',lw=1.15,ls='--',marker='s',ms=3.2,label='Right module')
    axes[0].set(xlim=(-.3,12.3),ylim=(0,1.07),xlabel='Moment depth n',ylabel='Moment modulus (dimensionless)')
    axes[0].set_xticks([0,3,6,9,12]);axes[0].set_yticks([0,.25,.5,.75,1])
    axes[0].set_title('(a) Six CAR sources, two modules',loc='left',pad=10)
    axes[0].legend(frameon=False,loc='lower left',fontsize=8.5)
    levels=np.arange(1,5)
    axes[1].semilogy(levels,bounds,color='#1F6F5F',lw=1.6,marker='o',ms=4,label='Analytic truncation bound')
    axes[1].semilogy(levels,changes,color='#404040',lw=1.1,ls='--',marker='s',ms=3.7,label='Observed change')
    axes[1].set(xlim=(.8,4.2),ylim=(2e-16,1e-6),xlabel='Ordered cutoff level',ylabel='Bound / change (dimensionless, log)')
    axes[1].set_xticks(levels);axes[1].yaxis.set_major_locator(LogLocator(base=10,numticks=5))
    axes[1].set_title('(b) State and flux convergence',loc='left',pad=10)
    axes[1].legend(frameon=False,loc='upper right',fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out',width=.7)
    fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.84)
    for extension in ['pdf','png']:
        save_figure_atomic(fig, FIGURES/f'nonabelian_common_car_moments.{extension}',dpi=240)
    plt.close(fig)
    boundary=ROOT/'reproducibility/results/coherent_boundary'
    b=json.loads((boundary/'COHERENT_BOUNDARY_CERTIFICATE.json').read_text())
    assert b['coherent_round_trip_return_probability_exact']=='1'
    assert b['iota_dephased_round_trip_return_probability_exact']=='5/8'
    fig,ax=plt.subplots(figsize=(85/25.4,52/25.4))
    ax.barh([1,0],[1,5/8],height=.46,color=['#1F6F5F','#8D8D88'])
    ax.set_yticks([1,0],['Coherent','Iota dephased'])
    ax.set(xlim=(0,1.16),ylim=(-.5,1.55),xlabel='Round-trip return probability')
    ax.set_xticks([0,.5,1])
    ax.text(1.025,1,'1',va='center',fontsize=9)
    ax.text(5/8+.025,0,'5/8',va='center',fontsize=9)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.set_title('Archived SU(2) intertwiner',loc='left',fontsize=10,pad=8)
    fig.subplots_adjust(left=.30,right=.97,bottom=.25,top=.80)
    for extension in ['pdf','png']:
        save_figure_atomic(fig, FIGURES/f'coherent_intertwiner_return.{extension}',dpi=240)
    plt.close(fig)
    print('Generated compact-group moment and coherent intertwiner figures.')


if __name__=='__main__':main()
