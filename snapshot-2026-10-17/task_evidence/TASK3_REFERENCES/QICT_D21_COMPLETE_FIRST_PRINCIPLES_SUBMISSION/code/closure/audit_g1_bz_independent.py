#!/usr/bin/env python3
"""Independent mesh audit of the C7 BZ nodes on two distinct meshes."""
from pathlib import Path
import json, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; DATA=ROOT/'data'
CERT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

def dn(k):
    x,y,z=k; cx,cy,cz=np.cos([x,y,z]); sx,sy,sz=np.sin([x,y,z])
    d=cx*cy*cz-sx*sy*sz
    n=np.array([sx*cy*cz+cx*sy*sz,cx*sy*cz-sx*cy*sz,cx*cy*sz+sx*sy*cz])
    return float(d),n

def Jdet(k,h):
    J=np.zeros((3,3)); k=np.asarray(k,float)
    for j in range(3):
        e=np.zeros(3); e[j]=h
        J[:,j]=(dn(k+e)[1]-dn(k-e)[1])/(2*h)
    return float(np.linalg.det(J))

def audit(N):
    # endpoint=False avoids duplicate -pi/pi boundary. N multiple of 4 includes +-pi/2 and 0.
    grid=np.linspace(-math.pi,math.pi,N,endpoint=False)
    # Use N=32/64 so 0 and +/-pi/2 are exact grid sites.
    cand=[]
    for x in grid:
      for y in grid:
       for z in grid:
        d,n=dn((x,y,z))
        if np.linalg.norm(n)<2e-12:
            det=Jdet((x,y,z),2e-5)
            cand.append({'kappa':[float(x),float(y),float(z)],'d':d,'qe':'0' if d>0 else 'pi','index':int(np.sign(det))})
    z0=[c for c in cand if c['qe']=='0']
    signs=sorted(set(c['index'] for c in z0))
    pi=[c for c in cand if c['qe']=='pi']
    return {'N':N,'nodes_found':len(cand),'zero_qe_nodes_found':len(z0),'pi_qe_nodes_found':len(pi),
            'zero_qe_index_signs':signs,'zero_qe_index_sum':sum(c['index'] for c in z0),
            'pi_qe_index_sum':sum(c['index'] for c in pi),
            'has_opposite_zero_qe_indices':signs==[-1,1],'nodes':cand}
res=[audit(32),audit(64)]
ok=all(r['has_opposite_zero_qe_indices'] and r['nodes_found']==16 and r['zero_qe_nodes_found']==8 and r['pi_qe_nodes_found']==8 and r['zero_qe_index_sum']==0 and r['pi_qe_index_sum']==0 for r in res)
cert={'gap':'G1','taxonomy':'CALCULATED','method':'independent uniform BZ meshes; direct Pauli-vector zeros; central-difference Jacobian',
      'meshes':res,'external_particle_data_opened':False,'status':'PASS' if ok else 'FAIL'}
(CERT/'g1_bz_independent_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
(DATA/'g1_bz_independent_nodes.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'status':cert['status'],'summary':[{k:r[k] for k in ('N','nodes_found','zero_qe_nodes_found','pi_qe_nodes_found','zero_qe_index_signs','zero_qe_index_sum','pi_qe_index_sum')} for r in res]},indent=2))
raise SystemExit(0 if ok else 1)
