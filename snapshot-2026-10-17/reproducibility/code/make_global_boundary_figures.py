#!/usr/bin/env python3
"""Regenerate publication figures from executed boundary/state certificates."""
from publication_io import save_figure_atomic
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'paper/figures'
RES=ROOT/'reproducibility/results'


def finish(fig,name):
    for extension in ['pdf','png']:
        save_figure_atomic(fig, FIG/f'{name}.{extension}',dpi=240)
    plt.close(fig)


def main():
    coupled=json.loads((RES/'coupled_boundary/COUPLED_BOUNDARY_GNS_CERTIFICATE.json').read_text())
    spatial=json.loads((RES/'spatial_state/SPATIAL_STATE_SELECTION_CERTIFICATE.json').read_text())
    chiral=json.loads((RES/'chiral_cone_domain/CHIRAL_CONE_DOMAIN_CERTIFICATE.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.labelsize':9,'axes.titlesize':9.5,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,
        'text.color':'#111111','axes.labelcolor':'#111111'})
    green='#1F6F5F';gray='#404040'
    fig,axes=plt.subplots(1,2,figsize=(178/25.4,87/25.4),gridspec_kw={'wspace':.38})
    radius=np.array([r['ground']['radius'] for r in coupled['cutoffs']])
    bounds=np.array([r['analytic_state_and_padding_error_upper'] for r in coupled['cutoffs']])
    changes=np.array([np.nan if r['moment_successive_max_change'] is None else r['moment_successive_max_change'] for r in coupled['cutoffs']])
    ax=axes[0]
    ax.semilogy(radius,bounds,'o-',color=green,lw=1.5,ms=4,label='Analytic truncation bound')
    ax.semilogy(radius,changes,'s--',color=gray,lw=1.1,ms=3.6,label='Observed moment change')
    ax.set(xlim=(3.5,10.5),ylim=(3e-16,2e-4),xlabel='Electric flux cutoff R',ylabel='Bound / change (dimensionless)')
    ax.set_xticks(radius);ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
    ax.set_title('(a) Two plaquettes, shared link',loc='left',pad=10)
    ax.legend(frameon=False,loc='upper right',fontsize=7.7)
    with np.load(RES/'spatial_state/SPATIAL_SPECTATOR_AND_C9_DATA.npz',allow_pickle=False) as z:
        counts=z['plaquette_counts'];values=abs(z['primitive_A7_spectator_Cn6'][:,1,0,0])
    q=spatial['exact_U1_primitive_return_certificate']['return_modulus_squared_upper']
    envelope=q**(counts-1)
    assert np.all(values>0) and np.all(values<=envelope+1e-12)
    ax=axes[1]
    ax.semilogy(counts,values,'o-',color=green,lw=1.5,ms=3.5,label='Computed three-group moment')
    ax.semilogy(counts,envelope,'s--',color=gray,lw=1.1,ms=3.4,label='Exact U(1) upper envelope')
    ax.set_xscale('log',base=10)
    ax.set(xlim=(.85,3800),ylim=(1e-180,3),xlabel='Isolated plaquette count N',ylabel='First-moment modulus (dimensionless)')
    ax.set_xticks([1,10,100,1000]);ax.set_xticklabels(['1','10','100','1000'])
    ax.set_yticks([1,1e-50,1e-100,1e-150])
    ax.set_title('(b) Pure A7 spectator dependence',loc='left',pad=10)
    ax.legend(frameon=False,loc='lower left',fontsize=7.7)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',width=.7)
    fig.subplots_adjust(left=.095,right=.985,bottom=.20,top=.86)
    finish(fig,'coupled_boundary_state_limits')
    fig,ax=plt.subplots(figsize=(110/25.4,75/25.4))
    records=chiral['radius_records'];r=np.array([a['coordinate_range'] for a in records])
    low=np.array([a['unavoidable_uniform_operator_error_lower'] for a in records])
    upper=np.array([a['line_fourier_truncation_upper'] for a in records])
    ax.semilogy(r,upper,'s--',color=gray,lw=1.15,ms=3.5,label='Axis Fourier truncation: upper bound')
    ax.semilogy(r,low,'o-',color=green,lw=1.5,ms=4,label='Any range-R projector: error lower bound')
    ax.set(xlim=(2,50),ylim=(4e-17,.5),xlabel='Coordinate range R',ylabel='Operator error bound (dimensionless)')
    ax.set_xticks([4,12,24,36,48]);ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
    ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',width=.7)
    ax.legend(frameon=False,loc='upper right',fontsize=7.2)
    fig.subplots_adjust(left=.20,right=.985,bottom=.20,top=.95)
    finish(fig,'free_overlap_range_bounds')
    print('Generated coupled_boundary_state_limits and free_overlap_range_bounds, PDF and PNG.')


if __name__=='__main__':main()
