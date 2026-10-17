#!/usr/bin/env python3
"""Final-size figure of computed coupled EW moments and resolvent bounds."""
from publication_io import save_figure_atomic
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'reproducibility/results/common_coupled_electroweak'
FIG=ROOT/'paper/figures'


def main():
    certpath=RESULT/'COMMON_COUPLED_ELECTROWEAK_GNS_CERTIFICATE.json'
    data=RESULT/'COMMON_COUPLED_ELECTROWEAK_GNS_RESOLVENTS.npz'
    cert=json.loads(certpath.read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest()==cert['data_sha256']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none',
        'text.color':'#111111','axes.labelcolor':'#111111'})
    fig,ax=plt.subplots(1,2,figsize=(178/25.4,86/25.4))
    fig.subplots_adjust(left=.095,right=.96,bottom=.21,top=.85,wspace=.45)
    with np.load(data,allow_pickle=False) as z:c=z['moments']
    for index,color,ls,mark,label in [(0,'#1F6F5F','-','o','Left doublet'),
                  (3,'#404040','--','s','Right singlet')]:
        y=np.abs(c[:,index,index])
        ax[0].plot(np.arange(13),y,color=color,ls=ls,marker=mark,ms=3.5,lw=1.4,label=label)
    ax[0].set(xlabel='Moment order n',ylabel='Relative moment modulus',
        xlim=(-.2,12.3),ylim=(0,1.035),xticks=[0,3,6,9,12],yticks=[0,.25,.5,.75,1])
    ax[0].set_title('(a) Coupled electroweak moments',loc='left',pad=12)
    ax[0].legend(frameon=False,fontsize=8.5,loc='lower left',handlelength=2.5)
    points=cert['resolvent_and_self_energy_bounds'];z=np.array([r['real_z'] for r in points])
    for key,color,ls,marker,label in [
            ('self_energy_operator_error_bound','#1F6F5F','-','o','Self-energy'),
            ('resolvent_operator_error_bound','#404040','--','s','Resolvent')]:
        y=np.array([r[key] for r in points])
        ax[1].semilogy(z,y,color=color,ls=ls,marker=marker,ms=4,lw=1.4)
        ax[1].text(4.12,y[-1],label,color=color,fontsize=8.5,va='center')
    ax[1].set(xlabel='Real spectral coordinate z',ylabel='Operator error bound (log scale)',
        xlim=(1.85,5.0),xticks=[2,3,4],ylim=(1e-9,3e-3))
    ax[1].set_title('(b) Controlled resolvents for z > 1',loc='left',pad=12)
    ax[1].minorticks_off()
    for suffix in ['pdf','svg','png']:
        save_figure_atomic(fig, FIG/('coupled_electroweak_spectral_moments.'+suffix),dpi=240)
    plt.close(fig)
    from PIL import Image
    p=FIG/'coupled_electroweak_spectral_moments.png'
    Image.open(p).convert('L').save('/tmp/qict-ew-figure-grayscale.png')
    print(json.dumps({'figure':str(p),'C0':[float(abs(c[0,0,0])),float(abs(c[0,3,3]))]}))


if __name__=='__main__':main()
