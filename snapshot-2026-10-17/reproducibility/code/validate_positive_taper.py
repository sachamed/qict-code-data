#!/usr/bin/env python3
"""Finite-dimensional checks of exact exterior sine-taper bounds (not physical G2 data)."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261010)
checks = 0
examples = []

def ok(condition, label):
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1

def nrm(a):
    return float(np.linalg.norm(a, 2))

def unitary(d):
    a = rng.standard_normal((d,d)) + 1j*rng.standard_normal((d,d))
    q,r = np.linalg.qr(a)
    return q @ np.diag(np.diag(r)/np.abs(np.diag(r)))

def taper(N):
    u = np.sqrt(2/(N+2)) * np.sin(np.pi*np.arange(1,N+2)/(N+2))
    b = np.array([np.dot(u[n:],u[:N+1-n]) for n in range(N+1)])
    return u,b

def geo_error(N,b,r):
    return float(sum((1-b[n])/r**(n+1) for n in range(1,N+1)) + 1/(r**(N+1)*(r-1)))

for N in (1,2,4,16,64,128):
    u,b = taper(N)
    delta = 1-np.cos(np.pi/(N+2))
    ok(abs(np.vdot(u,u)-1)<2e-14, f'normalization_{N}')
    ok(abs(b[1]-(1-delta))<2e-14, f'extremal_first_moment_{N}')
    H = np.diag(np.ones(N)/2,1)+np.diag(np.ones(N)/2,-1)
    ok(abs(np.linalg.eigvalsh(H)[-1]-(1-delta))<2e-14, f'path_radius_{N}')
    for n in (1, N, N+1, 2*N+1):
        bn=b[n] if n<=N else 0.
        ok(1-bn<=n*n*delta+4e-14, f'shift_bound_{N}_{n}')
    for r in (1.1,1.5,2.,3.):
        E=geo_error(N,b,r)
        ok(E <=delta*(r+1)/(r-1)**3 + 3e-13, f'quadratic_upper_{N}_{r}')
        exact=1/(r-1)
        approx=sum(b[n]/r**(n+1) for n in range(N+1))
        ok(abs((exact-approx)-E)<4e-13, f'atomic_witness_{N}_{r}')
        ok(abs(1/approx-1/exact)>=((r-1)**2)*delta/(r*r)-4e-13, f'optimal_selfenergy_{N}_{r}')

for d in (1,3,6):
    D=2*d+3
    U=unitary(D)
    V=unitary(D)[:,:d]
    W=unitary(D)
    C=[np.eye(d,dtype=complex)]
    C2=[np.eye(d,dtype=complex)]
    for n in range(1,65):
        C.append(V.conj().T@np.linalg.matrix_power(U,n)@V)
        C2.append(V.conj().T@np.linalg.matrix_power(W,n)@V)
    for N in (1,4,16,64):
        u,b=taper(N)
        if N<=4:
            T=np.block([[C[j-k] if j>=k else C[k-j].conj().T for k in range(N+1)] for j in range(N+1)])
            for theta in (.2,1.3):
                A=np.vstack([u[j]*np.exp(1j*j*theta)*np.eye(d) for j in range(N+1)])
                gram=A.conj().T@T@A
                density=np.eye(d,dtype=complex)
                for n in range(1,N+1):
                    density += b[n]*(np.exp(-1j*n*theta)*C[n]+np.exp(1j*n*theta)*C[n].conj().T)
                ok(nrm(gram-density)<3e-11,f'toeplitz_gram_{d}_{N}_{theta}')
                ok(np.linalg.eigvalsh(gram).min()>-3e-11,f'positive_density_{d}_{N}_{theta}')
        for radius in (1.1,2.,5.):
            for phase in (0.,.7):
                z=radius*np.exp(1j*phase)
                I=np.eye(d)
                G=V.conj().T@np.linalg.solve(z*np.eye(D)-U,V)
                GS=sum(b[n]*C[n]/z**(n+1) for n in range(N+1))
                sig=z*I-C[1]-np.linalg.inv(G)
                sigS=z*I-C[1]-np.linalg.inv(GS)
                E=geo_error(N,b,radius)
                stem=f'd={d},N={N},r={radius},phase={phase}'
                ok(nrm(np.linalg.inv(G)) <=radius+1+3e-11,'inverse_G_'+stem)
                ok(nrm(np.linalg.inv(GS))<=radius+1+3e-11,'inverse_GS_'+stem)
                ok(nrm(G-GS)<=E+3e-11,'resolvent_'+stem)
                ok(nrm(sig-sigS)<=(radius+1)**2*E+3e-11,'selfenergy_'+stem)
                # Convex mixture with a second unitary is a positive moment sequence.
                w=.02
                Chat=[(1-w)*C[n]+w*C2[n] for n in range(N+1)]
                GH=sum(b[n]*Chat[n]/z**(n+1) for n in range(N+1))
                sh=z*I-Chat[1]-np.linalg.inv(GH)
                Etot=E+sum(b[n]*(2*w)/radius**(n+1) for n in range(1,N+1))
                ok(nrm(np.linalg.inv(GH))<=radius+1+3e-11,'positive_inexact_inverse_'+stem)
                ok(nrm(sig-sh)<=2*w+(radius+1)**2*Etot+3e-11,'positive_inexact_error_'+stem)
                if d==6 and radius==2 and phase==.7:
                    examples.append({'depth':N,'exact_difference':nrm(sig-sigS),'bound':(radius+1)**2*E})

payload={'classification':'FINITE_UNITARY_TESTS_NOT_PHYSICAL_MASS_DATA',
         'checks_passed':checks,'checks_failed':0,
         'mathematical_scope':'Positive matrix moments, exterior resolvents, arbitrary finite dimensions; sine taper fixed radius; no physical G2 pole calculations',
         'examples':examples}
p=ROOT/'results/POSITIVE_TAPER_VALIDATION.json'
p.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
print('PASS',checks,'checks; output',p)
