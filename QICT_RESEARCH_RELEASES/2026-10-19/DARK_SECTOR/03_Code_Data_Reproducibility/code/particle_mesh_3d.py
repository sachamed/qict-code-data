"""Actual 3-D periodic particle-mesh N-body toy: periodic, dimensionless comoving coordinates.
Conditional Planck-background benchmark; no QICT particle parameters are inferred.
Methods: CIC assignment and interpolation, Fourier Poisson, midpoint in eta=ln(a).
Initial conditions: three coherent Zel'dovich plane waves, NOT a Gaussian realization.
No halo identification / virial-resolution claim. Outputs density 3D and projections.
"""
from pathlib import Path
import os
import numpy as np, json
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style
OUT=Path(os.environ.get('QICT_PM_OUTPUT',str(Path(__file__).resolve().parents[1])))

(OUT/'data').mkdir(exist_ok=True)
Om=0.315; Ol=1-Om
L=100.0 # comoving Mpc/h, arbitrary box
N=32; ngrid=32; nsteps=int(os.environ.get("QICT_PM_STEPS","300"))
ai=0.05; af=1.0
assert nsteps>0

def background(a):
    Hsq=Om*a**(-3)+Ol
    Oma=Om*a**(-3)/Hsq
    friction=2-1.5*Oma
    return Oma,friction

def growrhs(eta,y):
    a=np.exp(eta);oma,fr=background(a)
    return [y[1],1.5*oma*y[0]-fr*y[1]]
growth=solve_ivp(growrhs,(np.log(1e-3),0),[1e-3,1e-3],dense_output=True,rtol=1e-11,atol=1e-12)
y0=growth.sol(0)[0]
dini=growth.sol(np.log(ai))[0]/y0
fini=growth.sol(np.log(ai))[1]/growth.sol(np.log(ai))[0]
coords=(np.arange(N)+0.5)*L/N
q=np.stack(np.meshgrid(coords,coords,coords,indexing='ij'),axis=-1).reshape(-1,3)
kf=2*np.pi/L
amplitude=1.1
shift=-dini*amplitude*np.sin(kf*q)/kf
x=(q+shift)%L
v=fini*shift.copy()
ng3=ngrid**3
ks=2*np.pi*np.fft.fftfreq(ngrid,d=L/ngrid)
kx,ky,kz=np.meshgrid(ks,ks,ks,indexing='ij')
k2=kx*kx+ky*ky+kz*kz
k2[0,0,0]=np.inf

def cic_weights(xx):
    xx=(xx%L)*ngrid/L
    base=np.floor(xx).astype(np.int64)
    d=xx-base
    for dx in (0,1):
      for dy in (0,1):
       for dz in (0,1):
        delta=np.array([dx,dy,dz])
        ids=(base+delta)%ngrid
        w=np.prod(np.where(delta[None,:]==1,d,1-d),axis=1)
        yield ids,w

def deposit(x):
    mass=np.zeros((ngrid,ngrid,ngrid),dtype=float)
    for ids,w in cic_weights(x):
        np.add.at(mass,(ids[:,0],ids[:,1],ids[:,2]),w)
    return mass

def force(x,a):
    dm=deposit(x)
    delta=dm/(len(x)/ng3)-1
    dh=np.fft.fftn(delta)
    oma,_=background(a)
    # -3*Omega_m(a)/2 grad inverse-Laplacian delta
    dh0=(1.5*oma)*dh/k2
    forcegrid=[np.fft.ifftn(1j*ki*dh0).real for ki in [kx,ky,kz]]
    accel=np.zeros_like(x)
    for ids,w in cic_weights(x):
        accel+=np.stack([fj[ids[:,0],ids[:,1],ids[:,2]] for fj in forcegrid],axis=1)*w[:,None]
    return accel, delta

def rhs(eta,xx,vv):
    a=np.exp(eta);acc,_=force(xx,a)
    _,fr=background(a)
    return vv,-fr*vv+acc

snapshots={}
statistics={}
def store(a,xx):
    delta=deposit(xx)/(len(xx)/ng3)-1
    s={'scale_factor':float(a),'mean_delta':float(delta.mean()),
       'rms_delta':float(np.std(delta)), 'p99_delta':float(np.percentile(delta,99)),
       'peak_delta':float(delta.max()),'min_delta':float(delta.min()),
       'total_particle_mass_units':float(delta.sum()+delta.size),
       'nparticles':int(len(xx))}
    statistics[f'a{a:.2f}']=s
    snapshots[f'a{a:.2f}']=delta
store(ai,x)
eta=np.linspace(np.log(ai),np.log(af),nsteps+1)
for i in range(nsteps):
    hh=eta[i+1]-eta[i]
    a=eta[i]
    dx1,dv1=rhs(a,x,v)
    xmid=(x+0.5*hh*dx1)%L
    vmid=v+0.5*hh*dv1
    dx2,dv2=rhs(a+0.5*hh,xmid,vmid)
    x=(x+hh*dx2)%L
    v=v+hh*dv2
    acur=np.exp(eta[i+1])
    if any(abs(acur-s)<0.012 for s in [.25,.5,1]) or i==nsteps-1:
        store(acur,x)
store(1.0,x)
np.savez_compressed(OUT/'data'/'particle_mesh_snapshots.npz',**snapshots)
from render_particle_mesh_figure import render
render(OUT)
with open(OUT/'data'/'particle_mesh_results.json','w') as f:
  json.dump({'assumptions':{'Planck_background':Om,'N':N,'ngrid':ngrid,
    'L_Mpc_over_h':L,'initial_three_wave_amplitude_at_a1':amplitude,
    'nsteps':nsteps,'initial_scale_factor':ai,'not_gaussian':True},
    'snapshots':statistics},f,indent=2)
for kk,ss in statistics.items():
  print(kk,ss['rms_delta'],ss['peak_delta'],ss['mean_delta'])
assert np.all(np.isfinite(v))
assert all(abs(s['mean_delta'])<1e-12 for s in statistics.values())
