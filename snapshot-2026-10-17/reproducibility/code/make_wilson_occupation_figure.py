#!/usr/bin/env python3
"""Regenerate the source-complement and enclosed physical-kernel figure."""
from publication_io import save_figure_atomic
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'reproducibility/results/static_wilson_car'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source=RESULTS/'STATIC_PHYSICAL_KERNEL_ARITHMETIC_CERTIFICATE.json'
    cert=json.loads(source.read_text())
    data=RESULTS/'STATIC_PHYSICAL_KERNEL_INTERVAL_CENTRES.npz'
    assert sha(data)==cert['interval_centre_data_sha256']
    s=np.arange(9);complement=s*(s+2)/(s+1)**2*np.sin(2*(s+1)*np.pi/63)**2
    curves=[]
    with np.load(data,allow_pickle=False) as z:
        denominator=int(str(z['common_denominator']))
        for nl in range(4):
            values=z['gaussian_integer_centres'][4*nl,:,0]
            curves.append(np.array([complex(float(F(int(a),denominator)),float(F(int(b),denominator))) for a,b in values]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
       'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.7,
       'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
       'text.color':'#171717','axes.labelcolor':'#171717'})
    fig,axes=plt.subplots(1,2,figsize=(172/25.4,87/25.4))
    fig.subplots_adjust(left=.095,right=.975,bottom=.20,top=.85,wspace=.34)
    a,b=axes
    a.plot(s/2,complement,color='#1F6F5F',marker='o',lw=1.6,ms=4)
    a.set(xlabel=r'Neutral character spin $j$',ylabel='Source-complement probability',
          xlim=(-.05,4.12),ylim=(0,.72),xticks=[0,1,2,3,4],yticks=[0,.2,.4,.6])
    a.set_title('(a) Electric transfer to the complement',loc='left',pad=14)
    styles=[('#8D8D88',':',''),('#404040','--','s'),('#6F6F6F','-.','^'),('#1F6F5F','-','o')]
    names=[r'$n_L=0$',r'$n_L=1$',r'$n_L=2$',r'$n_L=3$']
    for n,(curve,(colour,line,marker)) in enumerate(zip(curves,styles)):
        b.plot(range(7),abs(curve),color=colour,ls=line,marker=marker,lw=1.6 if n==3 else 1.1,ms=4,label=names[n])
    b.set(xlabel=r'Microscopic time $n$',ylabel=r'Physical return modulus $|K(n,0)|$',
          xlim=(-.05,6.1),ylim=(0,1.04),xticks=[0,2,4,6],yticks=[0,.25,.5,.75,1])
    b.set_title('(b) Occupied static sources, '+r'$n_R=0$',loc='left',pad=14)
    b.legend(frameon=False,ncol=2,loc='lower left',fontsize=8,handlelength=2,columnspacing=1)
    for ax in axes:ax.tick_params(length=3,width=.7);ax.grid(axis='y',color='#E4E7E5',lw=.5);ax.set_axisbelow(True)
    figure=ROOT/'paper/figures/wilson_occupation_kernels.pdf'
    figure.parent.mkdir(exist_ok=True);save_figure_atomic(fig, figure,metadata={'Title':'Wilson occupation kernels','Author':'Mohamed Sacha'})
    plt.close(fig)
    print('Regenerated '+str(figure.relative_to(ROOT)))


if __name__=='__main__':main()
