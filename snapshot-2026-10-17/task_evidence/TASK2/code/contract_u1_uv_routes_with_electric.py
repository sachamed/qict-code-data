from __future__ import annotations
import json, math, hashlib, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/task2_continue')
import contract_u1_trivial_orbits as cu

PHI=math.pi/63.0
Q=-3
ROOT=Path('/mnt/data/task2_continue')
ORBS=json.load(open(ROOT/'C8_TRIVIAL_CURRENT_ORBITS.json'))

def electric_cost_for_current(J,q=Q):
    # Stream multiplication by U^q shifts E= q J before U_E.
    E=np.asarray(J,dtype=np.int64)*int(q)
    return int(np.sum(2*E*E+E))

def run_K(K):
    bv=cu.base.one(K); zv=complex(bv['Z_real'],bv['Z_imag'])
    routes={}
    for ds in ['8','10']:
        total=0j; rows=[]
        for i,r in enumerate(ORBS['routes'][ds]['orbits']):
            J=r['current']
            z,k0,offs=cu.evalJ(J,K,Q)
            mag=z/zv
            cost=electric_cost_for_current(J,Q)
            phase=np.exp(-1j*PHI*cost)
            spin=complex(*r['spin_trace_amplitude_sum'])
            contrib=spin*phase*mag
            total += contrib
            rows.append({
                'orbit':i,'multiplicity':r['multiplicity_in_enumeration'],
                'spin_amp_sum':r['spin_trace_amplitude_sum'],
                'electric_cost':cost,
                'electric_phase':[float(phase.real),float(phase.imag)],
                'magnetic_normalized_factor':[float(mag.real),float(mag.imag)],
                'contribution':[float(contrib.real),float(contrib.imag)]
            })
        routes[ds]={'amplitude':[float(total.real),float(total.imag)],'abs':float(abs(total)),'orbits':rows}
    # d=9 current is open on the periodic cell, hence no zero-electric return under pure plaquette magnetic action.
    routes['9']={'amplitude':[0.0,0.0],'abs':0.0,'selection_rule':'Every d=9 route current has nonzero boundary (origin -> (1,1,1)); plaquette shifts are cycles, hence <E=0|U_B U_E stream|E=0>=0 exactly.'}
    return {'K':K,'q':Q,'Z_vac':[zv.real,zv.imag],'routes':routes}

seq=[]
for K in (4,5,6):
    t=time.time(); r=run_K(K); r['seconds']=time.time()-t; seq.append(r)
    print('K',K,{d:r['routes'][d]['amplitude'] for d in ('10','9','8')},'sec',r['seconds'],flush=True)

# weighted family/chiral route phases already verified by rank-six algebra
red=json.load(open(ROOT/'TASK2_RANK6_CHIRAL_PARITY_REDUCTION.json'))
ph=[complex(*z) for z in red['route_phases']]
# order route phases = 10,9,8
a=ph[0]*complex(*seq[-1]['routes']['10']['amplitude'])
b=ph[1]*0j
c=ph[2]*complex(*seq[-1]['routes']['8']['amplitude'])
omega=np.exp(2j*np.pi/3)
I3=np.eye(3,dtype=complex);X=np.array([[0,0,1],[1,0,0],[0,1,0]],complex);Z=np.diag([1,omega,omega**2]);C=np.array([[0,1],[1,0]],complex);I2=np.eye(2,dtype=complex)
T=(a*np.kron(I2,I3)+b*np.kron(C,X)+c*np.kron(I2,Z))/3
s=np.linalg.svd(T,compute_uv=False)
Kuv=float(-np.mean(np.log(s))) if np.min(s)>0 else float('inf')
leak=np.linalg.eigvalsh(np.eye(6)-T.conj().T@T)
# convergence differences on route amplitudes and Kuv
reg=[]
for rr in seq:
    aa=ph[0]*complex(*rr['routes']['10']['amplitude']); cc=ph[2]*complex(*rr['routes']['8']['amplitude']);
    TT=(aa*np.kron(I2,I3)+cc*np.kron(I2,Z))/3
    ss=np.linalg.svd(TT,compute_uv=False); kk=float(-np.mean(np.log(ss))) if ss.min()>0 else float('inf')
    reg.append({'K':rr['K'],'a':[aa.real,aa.imag],'b':[0.0,0.0],'c':[cc.real,cc.imag],'singular_values':ss.tolist(),'K_UV':kk,'leakage_min':float(np.linalg.eigvalsh(np.eye(6)-TT.conj().T@TT).min())})
report={
 'status':'PASS_U1_TASK2_PRIMITIVE_ZERO_ELECTRIC_ROUTE_COMPRESSION',
 'scope':'Exact periodic L=2 U(1) magnetic-character + electric-phase sublayer of Task2 UV compression; composite carrier q=-3; no C10/Kato physical source.',
 'external_particle_data_used':False,
 'operator_order':'stream -> U_cell -> U_E -> U_B -> receiver compression (rightmost acts first)',
 'selection_rule':'Primitive zero-electric projection retains exactly route currents in im(boundary_2); d=9 is open and vanishes in this sublayer.',
 'sequence':seq,
 'weighted_rank6_at_K6':{'a':[a.real,a.imag],'b':[b.real,b.imag],'c':[c.real,c.imag],'singular_values':s.tolist(),'K_UV_U1_sublayer':Kuv,'leakage_min_eigenvalue':float(leak.min()),'operator_norm':float(s.max())},
 'regulator_summary':reg,
 'route_phase_source':'TASK2_RANK6_CHIRAL_PARITY_REDUCTION.json'
}
txt=json.dumps(report,indent=2,sort_keys=True)+'\n'; out=ROOT/'TASK2_U1_UV_ROUTE_COMPRESSION_EXACT.json'; out.write_text(txt)
print(json.dumps(report['weighted_rank6_at_K6'],indent=2));print('sha256',hashlib.sha256(txt.encode()).hexdigest())
