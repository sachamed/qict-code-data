from __future__ import annotations
import importlib.util, numpy as np, math, json, time, argparse, hashlib
from pathlib import Path
from scipy.linalg import eigh,eig
spec=importlib.util.spec_from_file_location('m','/mnt/data/task2_continue/fullcube_q3_floquet_kato.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PHI=math.pi/63

def run(extra=8,steps=25,s0=0.02):
 t=time.time();arr,KE,KB,seed,miss=m.build(m.EMIN+extra);K=KB.toarray()
 # primitive ground branch H_prim=K_E+K_B, not a single electric seed
 H=K+np.diag(KE); ew,evec=eigh(H,check_finite=False); g=evec[:,0]; g/=np.linalg.norm(g)
 prim_gap=float(ew[1]-ew[0]) if len(ew)>1 else float('nan')
 # diagonalize KB once; exact dense exponentials at each homotopy point
 kw,Q=eigh(K,check_finite=False)
 prev=g.copy(); rows=[]
 for s in np.linspace(s0,1.0,steps):
  # canonical distributed prescription: scale BOTH electric and magnetic phases by s
  Ub=(Q*np.exp(-1j*s*PHI*kw))@Q.conj().T
  U=Ub*np.exp(-1j*s*PHI*KE)[None,:]
  vals,vecs=eig(U,check_finite=False,overwrite_a=True)
  ovs=np.abs(vecs.conj().T@prev)**2; j=int(np.argmax(ovs)); v=vecs[:,j]; v/=np.linalg.norm(v)
  a=np.vdot(prev,v)
  if abs(a): v*=np.exp(-1j*np.angle(a))
  lam=vals[j]/abs(vals[j]); d=np.abs(np.angle(vals*np.conj(lam))); d[j]=10
  res=float(np.linalg.norm(U@v-lam*v)); ov=float(ovs[j]); gap=float(d.min())
  rows.append({'s':float(s),'phase':float(np.angle(lam)),'mod':float(abs(vals[j])),'step_overlap':ov,'phase_gap':gap,'residual':res,'primitive_overlap':float(abs(np.vdot(g,v))**2)})
  prev=v
 out={'status':'PASS_Q3_R1_CANONICAL_KATO_DENSE' if min(r['phase_gap'] for r in rows)>1e-8 and max(r['residual'] for r in rows)<1e-10 else 'CHECK',
      'construction':'primitive ground of H=K_E+K_B; canonical Kato homotopy U(s)=exp(-is phi K_B) exp(-is phi K_E), s>=0.02',
      'Q_composite_stream':3,'R_aux_cube':1,'electric_window':extra,'basis_size':len(arr),'truncated_shift_fraction':miss,
      'primitive_ground_energy':float(ew[0]),'primitive_gap':prim_gap,'steps':steps,'s0':s0,
      'final_phase':rows[-1]['phase'],'min_phase_gap':min(r['phase_gap'] for r in rows),'min_step_overlap':min(r['step_overlap'] for r in rows[1:]),
      'max_residual':max(r['residual'] for r in rows),'final_overlap_primitive':rows[-1]['primitive_overlap'],'rows':rows,'seconds':time.time()-t,
      'scope':'Auxiliary compact full-cube composite q=3 regulator; not the final BCC rank-six PEPO contraction.'}
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--extra',type=int,default=8);ap.add_argument('--steps',type=int,default=25);a=ap.parse_args();o=run(a.extra,a.steps);txt=json.dumps(o,indent=2,sort_keys=True)+'\n';p=Path(f'/mnt/data/task2_continue/Q3_R1_CANONICAL_KATO_DENSE_W{a.extra}.json');p.write_text(txt);print(json.dumps({k:o[k] for k in ['status','basis_size','primitive_ground_energy','primitive_gap','final_phase','min_phase_gap','min_step_overlap','max_residual','seconds']},indent=2));print('sha256',hashlib.sha256(txt.encode()).hexdigest())
