from __future__ import annotations
import os,sys,pickle,time,math,json,argparse
from collections import defaultdict
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,'/mnt/data/task3_resume')
import _RESUME_recoupled_mps_engine as r
STATE='/mnt/data/task3_resume/kprim_q3_coarse.pkl'

def build_matrix(blocks):
    # Encode composite row/column labels bijectively into int64 for fast unique.
    N=sum(len(x[0]) for x in blocks)
    rc=np.empty(N,np.int64); cc=np.empty(N,np.int64); vals=np.empty(N,np.complex128);o=0
    for bn,phys,inrb,mode,v in blocks:
        n=len(bn);sl=slice(o,o+n)
        rc[sl]=(bn.astype(np.int64)<<24) | (int(phys[0])<<16) | (int(phys[1])<<8) | int(phys[2])
        cc[sl]=(np.int64(inrb)<<16) | np.int64(int(mode)+32768)
        vals[sl]=v;o+=n
    ur,ri=np.unique(rc,return_inverse=True); uc,ci=np.unique(cc,return_inverse=True)
    A=np.zeros((len(ur),len(uc)),np.complex128);np.add.at(A,(ri,ci),vals)
    return A,ur,uc

def spectrum(A):
    # Direct singular values avoid loss of small but physical modes in A^H A.
    from scipy.linalg import svdvals
    sv=svdvals(A,check_finite=False)
    return np.asarray(sv*sv,float)

def factors(A,k):
    # Robust truncated SVD.  The earlier Gram-eigensystem route can lose the
    # numerical range on strongly rank-deficient sector blocks.
    from scipy.linalg import svd
    if k<=0:
        return np.zeros((A.shape[0],0),complex), np.zeros((0,A.shape[1]),complex)
    U,s,Vh=svd(A,full_matrices=False,check_finite=False,lapack_driver='gesvd')
    k=min(int(k),len(s))
    Q=U[:,:k]
    R=s[:k,None]*Vh[:k,:]
    return Q,R

def run(prev,si,budget,out):
    state=pickle.load(open(STATE,'rb')); z=np.load(prev,allow_pickle=False)
    ranks=z['ranks'];secs=z['sectors'];base=z['baseoff'];roff=z['roff'];coloff=z['coloff'];rdata=z['rdata'];col_in=z['col_in'];col_mode=z['col_mode']
    adj=r.core_adj(state.cores[si]);oldLsecs=state.bond_sectors[si];oldRsecs=state.bond_sectors[si+1];e=r.ORDER[si]
    contrib=defaultdict(list);gen=0;t=time.time()
    for ii in range(len(ranks)):
      k=int(ranks[ii]);
      if k<=0:continue
      cs,ce=int(coloff[ii]),int(coloff[ii+1]);nc=ce-cs;R=rdata[int(roff[ii]):int(roff[ii+1])].reshape(k,nc);sec=tuple(map(int,secs[ii]));b0=int(base[ii])
      for jj in range(nc):
        inlb=int(col_in[cs+jj]);mode=int(col_mode[cs+jj]);rv=R[:,jj];nz=np.flatnonzero(np.abs(rv)>1e-15)
        if len(nz)==0:continue
        bn=b0+nz.astype(np.int32);vv=rv[nz];oldL=oldLsecs[inlb]
        for oldphys,inrb,az in adj.get(inlb,()):
          oldR=oldRsecs[inrb];trans=[]
          if mode in (r.HB_IDLE,r.HB_DONE):
            if sec==oldL: trans.append((oldphys,oldR,mode,1+0j))
            if mode==r.HB_IDLE:
              for pid in r._HB_START.get(si,()):
                for nphys,nrs,f in r.pruned_local_transitions(pid,e,oldL,oldR,sec,oldphys,1,12):trans.append((nphys,nrs,r.HB_DONE if si==r._HB_SPAN[pid][1] else pid,f))
          else:
            pid=mode
            for nphys,nrs,f in r.pruned_local_transitions(pid,e,oldL,oldR,sec,oldphys,1,12):trans.append((nphys,nrs,r.HB_DONE if si==r._HB_SPAN[pid][1] else pid,f))
          for nphys,nrs,nm,f in trans:
            vals=vv*(az*f);contrib[nrs].append((bn,nphys,inrb,nm,vals));gen+=len(vals)
      if ii%500==0:print(json.dumps({'assemble_sector':ii,'of':len(ranks),'out_sectors':len(contrib),'generated':gen,'elapsed':time.time()-t}),flush=True)
    print(json.dumps({'stage':'assembled','site':si,'edge':e,'out_sectors':len(contrib),'generated':gen,'blocks':sum(map(len,contrib.values())),'seconds':time.time()-t}),flush=True)
    outsecs=list(contrib);lams=[];shapes=[];t1=time.time()
    for j,s in enumerate(outsecs):
      A,_,_=build_matrix(contrib[s]);lams.append(spectrum(A));shapes.append(A.shape)
      if j%250==0:print(json.dumps({'spec_sector':j,'of':len(outsecs),'singulars':sum(map(len,lams)),'elapsed':time.time()-t1}),flush=True)
    flat=[]
    for j,lam in enumerate(lams):flat.extend((float(x),j,k) for k,x in enumerate(lam))
    flat.sort(key=lambda x:x[0]);used=0.;dc=[0]*len(lams);B2=budget*budget
    for sq,j,k in flat:
      if used+sq<=B2*(1+1e-14):used+=sq;dc[j]+=1
      else:break
    keep=[len(lams[j])-dc[j] for j in range(len(lams))]
    print(json.dumps({'stage':'rank_selected','site':si,'budget':budget,'actual_discard':math.sqrt(used),'total_singulars':len(flat),'kept_rank':sum(keep),'discarded_rank':sum(dc),'rank_max':max(keep or [0]),'max_shape':list(max(shapes,key=lambda x:x[0]*x[1]))}),flush=True)
    # Factorized output
    qdatas=[];rdatas=[];row_out=[];row_phys=[];col_in2=[];col_mode2=[];qoff=[0];roff2=[0];rowoff=[0];coloff2=[0];ranks2=[];secs2=[];base2=[0];relmax=0.;t2=time.time()
    for j,s in enumerate(outsecs):
      A,ur,uc=build_matrix(contrib[s]);k=keep[j]
      if k:
        Q,R=factors(A,k)
        if j<5 or j%250==0:relmax=max(relmax,float(np.linalg.norm(A-Q@R)/(np.linalg.norm(A)+1e-300)))
        qdatas.append(Q.ravel());rdatas.append(R.ravel());row_out.extend((ur>>24).astype(np.int64).tolist());row_phys.extend(zip(((ur>>16)&255).astype(int),((ur>>8)&255).astype(int),(ur&255).astype(int)));col_in2.extend((uc>>16).astype(np.int64).tolist());col_mode2.extend(((uc&65535)-32768).astype(np.int64).tolist())
        qoff.append(qoff[-1]+Q.size);roff2.append(roff2[-1]+R.size);rowoff.append(rowoff[-1]+len(ur));coloff2.append(coloff2[-1]+len(uc));ranks2.append(k);secs2.append(s);base2.append(base2[-1]+k)
      else:
        qoff.append(qoff[-1]);roff2.append(roff2[-1]);rowoff.append(rowoff[-1]);coloff2.append(coloff2[-1]);ranks2.append(0);secs2.append(s);base2.append(base2[-1])
      if j%250==0:print(json.dumps({'factor_sector':j,'bond':base2[-1],'elapsed':time.time()-t2}),flush=True)
    qdata=np.concatenate(qdatas) if qdatas else np.empty(0,complex);rdata2=np.concatenate(rdatas) if rdatas else np.empty(0,complex)
    np.savez(out,qdata=qdata,rdata=rdata2,qoff=np.asarray(qoff,np.int64),roff=np.asarray(roff2,np.int64),rowoff=np.asarray(rowoff,np.int64),coloff=np.asarray(coloff2,np.int64),row_out=np.asarray(row_out,np.int32),row_phys=np.asarray(row_phys,np.int16),col_in=np.asarray(col_in2,np.int32),col_mode=np.asarray(col_mode2,np.int16),ranks=np.asarray(ranks2,np.int32),sectors=np.asarray(secs2,np.int16),baseoff=np.asarray(base2,np.int64))
    meta={'status':'PASS_HB_FACTORIZED_STEP','site':si,'edge':e,'budget':budget,'discarded_l2':math.sqrt(used),'bond':int(base2[-1]),'q_elements':int(qdata.size),'r_elements':int(rdata2.size),'sample_rel':relmax,'file_bytes':os.path.getsize(out),'seconds':time.time()-t}
    open(out+'.json','w').write(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('prev');ap.add_argument('site',type=int);ap.add_argument('budget',type=float);ap.add_argument('out');a=ap.parse_args();run(a.prev,a.site,a.budget,a.out)
