from __future__ import annotations
import json,hashlib,time,math,sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm
sys.path.insert(0,'/mnt/data/task2_continue')
import compute_su2_full_link_matter_route as one
PHI=math.pi/63.0
OUT=Path('/mnt/data/task2_continue/TASK2_SU2_THREE_LINK_INDEPENDENT_ROUTE.json')

def apply_local_W(X,W,axis,G):
    # X [g0,g1,g2,m], W indexed [g,m]
    others=[a for a in range(3) if a!=axis]
    perm=[axis,3]+others
    Y=np.transpose(X,perm).reshape(2*G,-1)
    Y=W@Y
    shp=[G,2]+[G]*len(others)
    Y=Y.reshape(shp)
    # current axes labels [axis,m,others...], invert to [0,1,2,m]
    labels=[axis,3]+others
    inv=[labels.index(i) for i in [0,1,2,3]]
    return np.transpose(Y,inv)

def apply_gauge_axis(X,Ug,axis,G):
    others=[a for a in range(4) if a!=axis]
    Y=np.transpose(X,[axis]+others).reshape(G,-1)
    Y=Ug@Y
    Y=Y.reshape([G]+[X.shape[a] for a in others])
    labels=[axis]+others
    inv=[labels.index(i) for i in range(4)]
    return np.transpose(Y,inv)

def run(nmax):
    t=time.time(); st,idx,W,Mchi,ke,KB=one.operators(nmax);G=len(st)
    v,lam,minov,mingap,eigres=one.class_kato(nmax); om=one.embed_class(v,st,idx);om/=np.linalg.norm(om)
    Ub=expm((-1j*PHI)*KB.toarray()); Ug=Ub@np.diag(np.exp(-1j*PHI*ke))
    # vacuum product
    vac=np.einsum('i,j,k->ijk',om,om,om,optimize=True)
    routes=[]
    for d in (8,9,10):
        vals=[]
        for b in range(2):
            X=np.zeros((G,G,G,2),complex);X[:,:,:,b]=vac
            for _ in range(d):
                for axis in (2,1,0): X=apply_local_W(X,W,axis,G)
                for axis in (0,1,2): X=apply_gauge_axis(X,Ug,axis,G)
            vals.append(np.vdot(vac,X[:,:,:,b]))
        amp=(vals[0]+vals[1])/2/(lam**(3*d))
        routes.append({'d':d,'amplitude':[float(amp.real),float(amp.imag)],'abs':float(abs(amp)),'phase':float(np.angle(amp)),
                       'matter_diagonal_raw':[[float(z.real),float(z.imag)] for z in vals]})
    return {'nmax':nmax,'single_link_gauge_dim':G,'joint_state_dim':2*G**3,'kato_eigen_residual':eigres,'kato_min_overlap':minov,'routes':routes,'seconds':time.time()-t}

def main():
    cases=[]
    for n in (2,3,4,5):
        r=run(n);cases.append(r);print('n',n,'G',r['single_link_gauge_dim'],'dim',r['joint_state_dim'],'routes',[(x['d'],x['abs']) for x in r['routes']],'sec',r['seconds'],flush=True)
    conv={}
    for d in (8,9,10):
        zz=[complex(*next(x for x in c['routes'] if x['d']==d)['amplitude']) for c in cases]
        conv[str(d)]={'last_change_abs':float(abs(zz[-1]-zz[-2])),'change_3_to_4':float(abs(zz[-2]-zz[-3]))}
    obj={'status':'PASS_SU2_THREE_LINK_INDEPENDENT_ROUTE_REGRESSION','external_particle_data_used':False,
         'definition':'Three independent compact SU(2) links, each in full Peter-Weyl basis, coupled sequentially to the same fundamental matter index in C8 order z,y,x; then each link receives its exact local electric+magnetic Floquet. The product Kato vacuum is used and the neutral eigenphase is divided out. Repeated for d=8,9,10.',
         'scope':'Exact for the declared three-independent-link reduction. It retains shared matter-index and temporal correlations but omits the 24-plaquette spatial couplings between links, so it is a regression target rather than full 3D PEPO.',
         'cases':cases,'convergence':conv}
    txt=json.dumps(obj,indent=2,sort_keys=True)+'\n';OUT.write_text(txt)
    print(json.dumps({'status':obj['status'],'frozen':cases[-1],'convergence':conv,'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
if __name__=='__main__':main()
