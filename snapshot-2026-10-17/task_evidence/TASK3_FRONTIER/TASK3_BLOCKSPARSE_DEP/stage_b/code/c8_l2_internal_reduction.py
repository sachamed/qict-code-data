#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
import c8_ordered_matter_stream as c8
I2=np.eye(2,dtype=complex)

def internal_stream_block(g,s,mu):
    Pp=(I2+c8.PAULI[mu])/2; Pm=(I2-c8.PAULI[mu])/2
    Um=c8.transporter(g,s,mu,-1); Up=c8.transporter(g,s,mu,+1)
    return np.kron(Pp,Um)+np.kron(Pm,Up)

def macro_block(g,s):
    Bz=internal_stream_block(g,s,2); s1=c8.shift(s,2,+1)
    By=internal_stream_block(g,s1,1); s2=c8.shift(s1,1,+1)
    Bx=internal_stream_block(g,s2,0); s3=c8.shift(s2,0,+1)
    return Bx@By@Bz,s3

def block(M,t,s):
    ti=c8.SID[t]; si=c8.SID[s]
    return M[ti*4:(ti+1)*4,si*4:(si+1)*4]

def active_link_ids_for_macro(s):
    out=[]; cur=s
    for mu in (2,1,0):
        # + branch uses (mu,cur); - branch uses link beginning at cur-e_mu.
        out.append((mu,cur))
        out.append((mu,c8.shift(cur,mu,-1)))
        cur=c8.shift(cur,mu,+1)
    return out,cur

def self_test():
    maxerr=0.0; maxleak=0.0; rows=[]; active=set()
    s0=(0,0,0); s=s0
    for _ in range(2):
        ll,s=active_link_ids_for_macro(s);active.update(ll)
    assert len(active)==12
    for seed in range(4):
        for d in (8,9,10):
            gs=[c8.random_links(300000+100*seed+t) for t in range(d)]
            F=np.eye(32,dtype=complex); R=np.eye(4,dtype=complex); s=s0
            for g in gs:
                A,_=c8.macro(g); F=A@F
                B,sn=macro_block(g,s); R=B@R; s=sn
            E=block(F,s,s0); err=float(np.linalg.norm(E-R,2)); maxerr=max(maxerr,err)
            leak=0.0
            for t in c8.SITES:
                if t!=s: leak=max(leak,float(np.linalg.norm(block(F,t,s0),2)))
            maxleak=max(maxleak,leak)
            rows.append({'seed':seed,'d':d,'endpoint':list(s),'block_residual_2norm':err,'off_route_position_leak_2norm':leak,
                         'reduced_unitarity_defect_2norm':float(np.linalg.norm(R.conj().T@R-np.eye(4),2))})
    assert maxerr<3e-14 and maxleak<3e-14
    return {'status':'PASS_EXACT_L2_C8_POSITION_ELIMINATION','full_dimension':32,'reduced_internal_dimension':4,
            'time_dependent_random_gauge_tests':len(rows),'tested_d':[8,9,10],
            'max_full_vs_reduced_block_residual_2norm':maxerr,'max_off_route_position_leak_2norm':maxleak,
            'active_physical_links_over_two_macro_parities':len(active),
            'active_links':[[mu,*s] for mu,s in sorted(active)],
            'theorem':'On L=2, +e_mu and -e_mu have the same endpoint. Each C8 substep is therefore a deterministic site permutation times a 4x4 spin x weak block. The site after every substep is fixed by parity, so arbitrary time-dependent gauge evolution along a route is represented exactly by 4x4 internal blocks; no 8-site position superposition is required.',
            'rows':rows,
            'scope':'Exact matter-stream reduction. Gauge-field PEPO contraction remains required.'}
if __name__=='__main__': print(json.dumps(self_test(),indent=2))
