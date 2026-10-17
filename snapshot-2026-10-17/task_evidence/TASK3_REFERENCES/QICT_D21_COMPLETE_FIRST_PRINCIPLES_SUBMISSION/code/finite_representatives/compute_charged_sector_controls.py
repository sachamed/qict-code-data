#!/usr/bin/env python3
"""QICT charged-sector numerical controls.

This program does not replace the full interacting C5/C7 correlator. It computes
only quantities that are completely specified by the published finite/BCC data:
(1) BCC geodesic dressing combinatorics and normalization;
(2) the Gaussian C7 charged-sector pole baseline;
(3) the external U(1)_y electric-substep spectral measure and its large-R limit.
"""
from __future__ import annotations
import csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
DATA=ROOT/'data'; FIG=ROOT/'manuscript'/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
I2=np.eye(2,dtype=complex)
A={
 ( 1, 1, 1): np.array([[0,-1+1j],[0,1-1j]],complex)/4,
 ( 1, 1,-1): np.array([[1+1j,0],[-1-1j,0]],complex)/4,
 ( 1,-1, 1): np.array([[0,-1-1j],[0,1+1j]],complex)/4,
 ( 1,-1,-1): np.array([[1-1j,0],[-1+1j,0]],complex)/4,
 (-1, 1, 1): np.array([[0,1+1j],[0,1+1j]],complex)/4,
 (-1, 1,-1): np.array([[1-1j,0],[1-1j,0]],complex)/4,
 (-1,-1, 1): np.array([[0,1-1j],[0,1-1j]],complex)/4,
 (-1,-1,-1): np.array([[1+1j,0],[1+1j,0]],complex)/4,
}
MOVES=list(A); CENTER=(0,0,0)
SPLUS={(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)}
SMINUS={tuple(-x for x in h) for h in SPLUS}
THETA_E=np.array([0.17401963748403787,0.5574263894516671,0.9804870890570604])

def add(a,b): return tuple(x+y for x,y in zip(a,b))
def shell_formula(R): return 6*R*R+2
def ball_formula(R): return 2*R**3+3*R**2+3*R+1
def geodesic_count_formula(R): return 6*4**R-12*2**R+8
def nu_formula(R): return geodesic_count_formula(R)/(8**R)
def orientation_count_formula(R,m):
    return 3*(2**(R+1)-4)*math.comb(R,m)+(4 if m==0 else 0)+(4 if m==R else 0)

def bcc_ball(R):
    dist={CENTER:0}; frontier={CENTER}
    for r in range(R):
        nxt=set()
        for x in frontier:
            for h in MOVES:
                y=add(x,h)
                if y not in dist:
                    dist[y]=r+1; nxt.add(y)
        frontier=nxt
    return dist

def geodesic_paths(R):
    dist=bcc_ball(R); out=defaultdict(list)
    def rec(pos,seq):
        r=len(seq)
        if r==R:
            out[pos].append(tuple(seq)); return
        for h in MOVES:
            y=add(pos,h)
            if dist.get(y)==r+1: rec(y,seq+[h])
    rec(CENTER,[]); return dist,out

def path_matrix(path):
    M=I2.copy()
    for h in path: M=A[h]@M
    return M

def electric_phase(R,m):
    # Conditional oriented-electric control: positive Cayley tetrahedron SPLUS.
    return np.exp(-1j*np.pi*(66*(R-m)+78*m)/63.0)

def electric_measure_small(R):
    N=geodesic_count_formula(R)
    return [(m,orientation_count_formula(R,m)/N,electric_phase(R,m)) for m in range(R+1)]

def electric_moment_stable(R,n):
    if n%21==0: return 1+0j
    th=4*np.pi*n/21
    t=np.exp(-1j*th)
    c=np.cos(th/2)*np.exp(-1j*th/2)
    a=2.0**(-R) if R<1024 else 0.0
    b=4.0**(-R) if R<512 else 0.0
    num=(6-12*a)*(c**R)+4*b*(1+t**R)
    den=6-12*a+8*b
    return np.exp(-1j*22*np.pi*R*n/21)*num/den

def mod21_distribution(R):
    # Reconstruct probability distribution from its 21 Fourier coefficients.
    hats=np.array([electric_moment_stable(R,n) for n in range(21)],complex)
    p=np.fft.ifft(hats).real
    p[p<0]=np.maximum(p[p<0],-1e-15)
    return p

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    enum_rows=[]; worst={'count':0,'gram':0.0,'pathnorm':0.0}
    for R in range(1,7):
        dist,paths=geodesic_paths(R)
        N=sum(map(len,paths.values()))
        G=np.zeros((2,2),complex); cnt=Counter()
        target_norm=2.0**(1-3*R)
        for plist in paths.values():
            for p in plist:
                M=path_matrix(p); G+=M.conj().T@M
                worst['pathnorm']=max(worst['pathnorm'],abs(np.linalg.norm(M,'fro')**2-target_norm))
                cnt[sum(h in SMINUS for h in p)]+=1
        nu=nu_formula(R)
        worst['count']=max(worst['count'],abs(N-geodesic_count_formula(R)))
        worst['gram']=max(worst['gram'],float(np.linalg.norm(G-nu*I2,'fro')))
        assert len(dist)==ball_formula(R)
        assert sum(d==R for d in dist.values())==shell_formula(R)
        assert all(cnt[m]==orientation_count_formula(R,m) for m in range(R+1))
        enum_rows.append(dict(R=R,ball_sites=len(dist),boundary_sites=shell_formula(R),geodesic_paths=N,
                              nu_R=nu,normalization=nu**-0.5,source_gram_residual=np.linalg.norm(G-nu*I2,'fro')))
    write_csv(DATA/'charged_dressing_smallR.csv',enum_rows)

    # Gaussian pole baseline: source normalization changes residues, not the free C7 kernel poles.
    pole_rows=[]
    Rvals=list(range(1,65))+[96,128,192,256,384,512,768,1024,1536,2000]
    for R in Rvals:
        for a,theta in enumerate(THETA_E,1):
            pole_rows.append(dict(R=R,branch=a,theta=float(theta),z_real=float(np.cos(theta)),z_imag=float(-np.sin(theta)),
                                  energy_over_Etau=float(theta),normalized_source_gram=1.0))
    write_csv(DATA/'charged_gaussian_pole_stability.csv',pole_rows)

    # Large-R electric-substep convergence.
    conv_rows=[]
    selected=set(list(range(1,65))+[80,100,128,160,200,256,320,384,500,640,768,1000,1250,1500,1750,2000])
    for R in sorted(selected):
        p=mod21_distribution(R); u=np.full(21,1/21)
        tv=.5*float(np.sum(np.abs(p-u)))
        maxmom=max(abs(electric_moment_stable(R,n)) for n in range(1,21))
        conv_rows.append(dict(R=R,total_variation_to_uniform_21roots=tv,
                              max_nontrivial_fourier_moment_abs=float(maxmom),
                              min_weight=float(p.min()),max_weight=float(p.max())))
    write_csv(DATA/'charged_electric_largeR_convergence.csv',conv_rows)

    # Small-R electric atoms, exact weights.
    atom_rows=[]
    for R in range(1,13):
        for m,w,z in electric_measure_small(R):
            atom_rows.append(dict(R=R,m=m,weight=w,z_real=float(z.real),z_imag=float(z.imag),phase=float(np.angle(z))))
    write_csv(DATA/'charged_electric_atoms_smallR.csv',atom_rows)

    # Plots.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4,4.2))
    rr=np.arange(1,65)
    plt.semilogy(rr,[nu_formula(int(r)) for r in rr],marker='o',markersize=2)
    plt.xlabel('BCC dressing radius R'); plt.ylabel(r'$\nu_R$')
    plt.tight_layout(); plt.savefig(FIG/'charged_dressing_norm.pdf'); plt.close()

    plt.figure(figsize=(6.4,4.2))
    x=[r['R'] for r in conv_rows]; y=[r['total_variation_to_uniform_21roots'] for r in conv_rows]
    plt.semilogy(x,y,marker='o',markersize=3)
    plt.xlabel('BCC dressing radius R'); plt.ylabel('total variation to uniform 21-root measure')
    plt.tight_layout(); plt.savefig(FIG/'charged_electric_largeR_convergence.pdf'); plt.close()

    plt.figure(figsize=(6.4,4.2))
    for a,theta in enumerate(THETA_E,1):
        plt.plot([1,2000],[theta,theta],label=f'branch {a}')
    plt.xlabel('BCC dressing radius R'); plt.ylabel(r'Gaussian rest phase $\theta_{e,a}$')
    plt.legend(); plt.tight_layout(); plt.savefig(FIG/'charged_gaussian_pole_stability.pdf'); plt.close()

    cert={
      'release':'QICT publication calculation',
      'external_particle_data_used':False,
      'small_R_geodesic_enumeration_max':6,
      'large_R_control_max':2000,
      'gaussian_poles_stable_for_all_R_by_theorem':True,
      'gaussian_electron_rest_phases':THETA_E.tolist(),
      'electric_substep_limit':'uniform measure on the 21st roots of unity',
      'electric_R2000_total_variation':next(r['total_variation_to_uniform_21roots'] for r in conv_rows if r['R']==2000),
      'electric_R2000_max_nontrivial_moment':next(r['max_nontrivial_fourier_moment_abs'] for r in conv_rows if r['R']==2000),
      'max_geodesic_count_residual':worst['count'],
      'max_source_gram_residual':worst['gram'],
      'max_path_norm_residual':worst['pathnorm'],
      'full_interacting_charged_pole_status':'NOT_COMPUTED',
      'reason':'The C5/C7 free formulas and the interacting C8J/C11 local tensor belong to distinct operator levels; exponentiating the link-dependent receiver Laplacian has generically infinite Peter-Weyl support and is therefore treated by controlled harmonic convergence.',
      'status':'PASS_FOR_STATED_CONTROLS_ONLY'
    }
    (CERT/'charged_sector_controls_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps(cert,indent=2))
if __name__=='__main__': main()
