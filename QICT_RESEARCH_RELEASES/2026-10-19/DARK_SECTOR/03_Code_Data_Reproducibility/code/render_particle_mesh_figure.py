"""Render the stored PM density fields with one common, readable colour scale."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style

def render(root):
    root=Path(root)
    with np.load(root/'data/particle_mesh_snapshots.npz') as data:
        keys=list(data.files)
        selected=[keys[0],min(keys,key=lambda k:abs(float(k[1:])-.5)),'a1.00']
        projections=[np.log1p(np.maximum(data[k].mean(axis=2),-.99)) for k in selected]
    vmin=min(float(a.min()) for a in projections);vmax=max(float(a.max()) for a in projections)
    fig,axs=plt.subplots(1,3,figsize=(6.4,2.75),constrained_layout=True,sharey=True)
    for ax,key,proj in zip(axs,selected,projections):
        im=ax.imshow(proj,origin='lower',extent=[0,100,0,100],vmin=vmin,vmax=vmax,
                     cmap='cividis',interpolation='nearest')
        ax.set_title(f'a = {key[1:]}')
        ax.set_xlabel('x [Mpc/h]');ax.set_xticks([0,50,100]);ax.set_yticks([0,50,100])
    axs[0].set_ylabel('y [Mpc/h]')
    fig.colorbar(im,ax=axs,shrink=.73,label='ln(projected density / mean)')
    target=(Path(__file__).resolve().parents[2]/'01_Main_Manuscript/figures/particle_mesh_3d.pdf' if root==Path(__file__).resolve().parents[1] else root/'figures/particle_mesh_3d.pdf')
    plot_style.save_figure(fig,target)
    plt.close(fig)

if __name__=='__main__':render(Path(__file__).resolve().parents[1])
