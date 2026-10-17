#!/usr/bin/env python3
"""Regenerate ten formerly untraced reference figures from equations or retained code.

The output manifest records inputs, CSV arrays, dimensions and checks. The
nonuniform C21 delay profile is now the explicit profile used in the supplied
verify_exact_cycle.py, whose mean is 1. The static strip is a finite diagnostic.
The BCC resistance is a classical tangent calculation. Neither is a physical
charged mass calculation. No random data or experimental parameters are used.
"""
from publication_io import save_figure_atomic
from pathlib import Path
import csv, hashlib, importlib.util, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.special import gamma

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT.parent
OLD=ROOT/'task_evidence/TASK3_REFERENCES/QICT_D21_COMPLETE_FIRST_PRINCIPLES_SUBMISSION'
OUT=ROOT/'reproducibility/results/reference_figures'
OUT.mkdir(parents=True,exist_ok=True)
TEAL='#1F6F5F'; GRAY='#666966'; INK='#171717'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
 'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.linewidth':.65,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
 'ps.fonttype':42,'text.color':INK,'axes.labelcolor':INK,
 'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})
records=[]
SPECS={
 'c21_laplacian_spectrum':('02_Supplementary_Information',125.56,75,'Exact cycle eigenvalues','j=0,...,20; lambda_j=4 sin^2(pi*j/21)'),
 'c21_heat_trace':('02_Supplementary_Information',125.56,75,'Heat trace of the 21-cycle','K(u)=sum_j exp(-u*lambda_j), u in [0,8]'),
 'bcc_photon_directional_coefficient':('02_Supplementary_Information',154.8,90,'BCC directional transport coefficient','b=(11-8 sum n_i^4)/32'),
 'fig3_bcc_angle':('01_Main_Manuscript',172,95,'BCC ray-wavevector angular coefficient','C=sqrt(sum n_i^6-(sum n_i^4)^2)/3'),
 'q_admissibility_sieve':('01_Main_Manuscript',130.72,78,'Endpoint ranks on the admitted congruence branch','r=7+6n, n=0,...,5; entropy log(3r)'),
 'c21_reciprocal_delay_cancellation':('02_Supplementary_Information',125.56,78,'Origin-resolved delays and their exact cyclic means','tau_j=1+.1cos(2pi*j/21)+.03sin(4pi*j/21)'),
 'c8_static_strip_moments':('02_Supplementary_Information',134.16,80,'Finite static-strip moments','Supplied compute_c8_static_charge_strip.py, N=13, R=1,2,3'),
 'c8_static_strip_regulator_stability':('02_Supplementary_Information',134.16,80,'Static-strip first-moment regulator comparison','Retained c8_static_charge_strip_convergence.csv'),
 'c10_bcc_gauss_coulomb_convergence':('02_Supplementary_Information',123.84,76,'Dirichlet BCC resistance approaches its analytic limit','Supplied verify_c10_bcc_gauss_coulomb.py, R=1,...,12,14,16,18,20,24'),
 'fig1_operator_to_observable':('01_Main_Manuscript',168.56,85,'Representation, state and sources jointly determine the observable','Main microscopic specification and relative-moment definition')
}
(OUT/'VISUAL_SPEC.json').write_text(json.dumps({'figures':SPECS,'uncertainty':'Deterministic analytic or finite numerical references; no sampling intervals.','style':'White, embedded sans-serif, teal focal and neutral baselines, final-size geometry.','reference_grammar':['GPT-6 Astra system-card figures 1,2,5,6','Append-only context diagram','HealthBench reliability curves','Needle retrieval heatmap']},indent=2)+'\n')
def canvas(key):
 _,w,h,_,_=SPECS[key]
 f,a=plt.subplots(figsize=(w/25.4,h/25.4));f.subplots_adjust(left=.16,right=.96,bottom=.20,top=.94)
 return f,a

def save(key,f,data=None,source_files=(),checks=None):
 folder,w,h,claim,source=SPECS[key]
 dest=PACKAGE/folder/'figures'/f'{key}.pdf'
 save_figure_atomic(f, dest,metadata={'Title':claim,'Creator':'QICT deterministic reference figure generator','CreationDate':None,'ModDate':None})
 save_figure_atomic(f, OUT/f'{key}.png',dpi=300)
 if data is not None:
  cols,rows=data
  with (OUT/f'{key}.csv').open('w',newline='') as stream:
   writer=csv.writer(stream);writer.writerow(cols);writer.writerows(rows)
 plt.close(f)
 records.append({'document':folder,'figure':dest.name,'source':source,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},'dimensions_mm':[w,h],'checks':checks or {},'data_csv':f'{key}.csv' if data is not None else None})
 print('PASS',key,flush=True)

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def main():
 j=np.arange(21);lam=4*np.sin(np.pi*j/21)**2
 f,a=canvas('c21_laplacian_spectrum');a.scatter(j,lam,s=20,color=TEAL);a.set(xlabel='Cycle mode j',ylabel='Laplacian eigenvalue',xticks=[0,5,10,15,20],ylim=(-.1,4.2))
 save('c21_laplacian_spectrum',f,(['j','lambda_j'],zip(j,lam)),checks={'zero_mode':float(lam[0]),'trace':float(sum(lam))})
 u=np.linspace(0,8,401);trace=np.exp(-u[:,None]*lam[None,:]).sum(1)
 f,a=canvas('c21_heat_trace');a.plot(u,trace,color=TEAL,lw=1.6);a.set(xlabel='Heat parameter u (dimensionless)',ylabel='Heat trace K$_{21}$(u)',xlim=(0,8),ylim=(0,22));a.axhline(1,color=GRAY,lw=.75,ls='--')
 save('c21_heat_trace',f,(['u','heat_trace'],zip(u,trace)),checks={'K_at_zero':float(trace[0])})
 cmap=LinearSegmentedColormap.from_list('qict_teal',['#F1F5F3','#8EB5A8',TEAL])
 for key,theta,phi,metric in [
 ('bcc_photon_directional_coefficient',np.linspace(0,np.pi,181),np.linspace(-np.pi,np.pi,361),'b'),
 ('fig3_bcc_angle',np.linspace(0,np.pi/2,181),np.linspace(0,np.pi/2,181),'C')]:
  ph,th=np.meshgrid(phi,theta);ns=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
  s4=(ns**4).sum(0);s6=(ns**6).sum(0)
  values=(11-8*s4)/32 if metric=='b' else np.sqrt(np.maximum(s6-s4*s4,0))/3
  f,a=canvas(key);f.subplots_adjust(left=.12,right=.87,bottom=.18,top=.94)
  scale=np.pi if metric=='b' else np.pi/180
  mesh=a.pcolormesh(ph/scale,th/scale,values,cmap=cmap,shading='auto',rasterized=True)
  a.set(xlabel='Azimuth '+('ϕ/π' if metric=='b' else 'ϕ (degrees)'),ylabel='Polar angle '+('θ/π' if metric=='b' else 'θ (degrees)'))
  cb=f.colorbar(mesh,ax=a,fraction=.05,pad=.04);cb.set_label(metric+' (dimensionless)')
  if metric=='C':a.set(xticks=[0,30,60,90],yticks=[0,30,60,90])
  save(key,f,(['theta_radian','phi_radian',metric],zip(th.ravel(),ph.ravel(),values.ravel())),checks={'min':float(values.min()),'max_sampled':float(values.max())})
 r=np.arange(7,38,6);entropy=np.log(3*r)
 f,a=canvas('q_admissibility_sieve');a.plot(r,entropy,'o-',ms=4,lw=1,color=GRAY);a.scatter(r[0],entropy[0],s=32,color=TEAL,zorder=3);a.set(xlabel='Receiver rank r',ylabel='Entropy log(3r)',xticks=r,xlim=(4.5,39))
 save('q_admissibility_sieve',f,(['r','q','neutral_rank','entropy'],zip(r,2*r,3*r,entropy)))
 delay=1+.1*np.cos(2*np.pi*j/21)+.03*np.sin(4*np.pi*j/21);mean=delay.mean();rows=[];residual=[]
 f,a=canvas('c21_reciprocal_delay_cancellation')
 for d,col,ls in [(10,'#888C89',':'),(9,'#454A47','--'),(8,TEAL,'-')]:
  val=np.array([sum(delay)-2*sum(delay[(origin+np.arange(d))%21]) for origin in j])/mean
  a.plot(j,val,color=col,ls=ls,lw=1.4,label=f'd = {d}; mean = {21-2*d}');a.axhline(21-2*d,color=col,lw=.6,ls=ls,alpha=.6)
  rows.extend(zip(j,[d]*21,val));residual.append(abs(val.mean()-(21-2*d)))
 a.set(xlabel='Cyclic origin',ylabel='Delay difference / mean edge delay',xticks=[0,5,10,15,20]);f.subplots_adjust(top=.83);a.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,fontsize=7.2,columnspacing=1.0,handlelength=2.0)
 assert max(residual)<1e-13
 save('c21_reciprocal_delay_cancellation',f,(['origin','separation_d','normalized_delay_difference'],rows),source_files=[OLD/'code/finite_representatives/verify_exact_cycle.py'],checks={'mean_edge_delay':float(mean),'max_cyclic_mean_error':float(max(residual))})
 strip_path=OLD/'code/finite_representatives/compute_c8_static_charge_strip.py';strip=module(strip_path,'qict_reference_strip');strip.DATA=OUT
 saved=list(csv.DictReader((OLD/'data/c8_static_charge_strip_convergence.csv').open()))
 f,a=canvas('c8_static_strip_moments');rows=[];errors=[]
 for radius,col,ls in [(1,'#888C89',':'),(2,'#454A47','--'),(3,TEAL,'-')]:
  result=strip.run_case(radius,13,save_moments=True)
  values=list(csv.DictReader((OUT/f'c8_static_charge_moments_R{radius}_N13.csv').open()))
  nn=np.array([int(x['n']) for x in values]);y=np.array([float(x['abs']) for x in values])
  ref=next(x for x in saved if int(x['N'])==13 and int(x['R'])==radius)
  errors.append(max(abs(result[k]-float(ref[k])) for k in ['c1_abs','c2_abs','c10_abs']))
  a.plot(nn,y,color=col,ls=ls,lw=1.25,label=f'R = {radius}');rows.extend(zip(nn,[radius]*len(nn),y))
 a.set(xlabel='Floquet step n',ylabel='Absolute moment |C$_R$(n)|',ylim=(.3,1.03),xlim=(0,63));a.legend(loc='lower left')
 assert max(errors)<1e-9
 save('c8_static_strip_moments',f,(['n','R','absolute_moment'],rows),source_files=[strip_path,OLD/'data/c8_static_charge_strip_convergence.csv'],checks={'max_archived_C1_C2_C10_difference':float(max(errors))})
 f,a=canvas('c8_static_strip_regulator_stability');rows=[]
 for radius,col,ls in [(1,'#888C89',':'),(2,'#454A47','--'),(3,TEAL,'-')]:
  rr=sorted([x for x in saved if int(x['R'])==radius],key=lambda x:int(x['N']));nn=[int(x['N']) for x in rr];yy=[float(x['c1_abs']) for x in rr]
  a.plot(nn,yy,'o',color=col,ms=4,label=f'R = {radius}');a.plot(nn,yy,color=col,ls=ls,lw=1.2);rows.extend(zip(nn,[radius]*len(nn),yy))
 a.set(xlabel='Finite gauge regulator order N',ylabel='Absolute first moment |C$_R$(1)|',xticks=[13,15,17],xlim=(12.6,17.8));f.subplots_adjust(top=.84);a.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,fontsize=8)
 save('c8_static_strip_regulator_stability',f,(['N','R','absolute_first_moment'],rows),source_files=[OLD/'data/c8_static_charge_strip_convergence.csv'])
 gauss_path=OLD/'code/finite_representatives/verify_c10_bcc_gauss_coulomb.py';gauss=module(gauss_path,'qict_reference_gauss')
 radii=list(range(1,13))+[14,16,18,20,24];vals=[gauss.resistance(x) for x in radii];res=np.array([x[1] for x in vals]);limit=gamma(.25)**4/(32*np.pi**3)
 assert np.all(np.diff(res)>0) and np.all(res<limit)
 f,a=canvas('c10_bcc_gauss_coulomb_convergence');a.plot(radii,res,'o-',color=TEAL,lw=1.5,ms=3,label='Finite Dirichlet ball');a.axhline(limit,color=GRAY,lw=1,ls='--',label='Analytic infinite-volume value');a.set(xlabel='BCC graph radius R',ylabel='Unit-charge resistance',xticks=[0,5,10,15,20,25]);a.legend(loc='lower right',fontsize=8)
 save('c10_bcc_gauss_coulomb_convergence',f,(['R','vertices','resistance'],zip(radii,[x[0] for x in vals],res)),source_files=[gauss_path],checks={'monotone_increase':True,'analytic_limit':float(limit)})
 key='fig1_operator_to_observable';f,a=canvas(key);f.subplots_adjust(left=.015,right=.985,bottom=.03,top=.97);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
 boxes=[(.02,.64,'Receiver and route\nrepresentation'),(.37,.64,'Compact gauge\nand CAR update'),(.72,.64,'Selected neutral\nstate'),(.02,.19,'Dressed chiral\nsource maps'),(.37,.19,'Relative moments\nand resolvent'),(.72,.19,'Spectral and\ncurrent matching')]
 for i,(xx,yy,label) in enumerate(boxes):
  a.add_patch(Rectangle((xx,yy),.25,.21,edgecolor=TEAL if i==4 else GRAY,facecolor='#F4F8F6' if i==4 else 'white',linewidth=1.0));a.text(xx+.125,yy+.105,label,ha='center',va='center',fontsize=9)
 def arrow(start,end):a.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':.85,'color':GRAY})
 arrow((.27,.745),(.37,.745));arrow((.62,.745),(.72,.745));arrow((.145,.64),(.145,.40));arrow((.495,.64),(.495,.40));arrow((.845,.64),(.59,.40));arrow((.27,.295),(.37,.295));arrow((.62,.295),(.72,.295))
 a.text(.845,.10,'Independent clock\nsets physical units',ha='center',va='center',fontsize=8.5)
 save(key,f)
 (OUT/'FIGURE_REGENERATION.json').write_text(json.dumps({'status':'PASS','count':len(records),'figures':records},indent=2)+'\n')

if __name__=='__main__':main()
