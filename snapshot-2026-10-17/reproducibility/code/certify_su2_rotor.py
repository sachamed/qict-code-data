#!/usr/bin/env python3
"""Exact rational error certificates for the declared infinite SU(2) rotor.

Floating-point eigensolvers propose trial vectors only. All inequalities,
Sturm counts, residuals and displayed interval endpoints use exact rationals.
"""
from __future__ import annotations
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import json, csv, hashlib, sys
sys.set_int_max_str_digits(100000)
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'

def rat(x): return {'numerator':str(x.numerator),'denominator':str(x.denominator)}
def sqrt_up(x, digits=55):
    assert x>=0
    scale=10**digits
    k=isqrt(x.numerator*scale*scale//x.denominator)
    if F(k*k,scale*scale)<x:k+=1
    return F(k,scale)
def decimal_bound(x,digits=28,upper=False):
    scale=10**digits
    k= -((-x.numerator*scale)//x.denominator) if upper else (x.numerator*scale)//x.denominator
    sign='-' if k<0 else ''; k=abs(k)
    return f'{sign}{k//scale}.{k%scale:0{digits}d}'
def enclosure(mid,err):
    return [decimal_bound(mid-err),decimal_bound(mid+err,upper=True)]
def apply(diag,b,v):
    return [diag[i]*v[i]-b*(v[i-1] if i else 0)-b*(v[i+1] if i+1<len(v) else 0) for i in range(len(v))]
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def sturm_count(diag,b,x):
    # Sturm characteristic-polynomial sequence, skipping exact zeros.
    p0=F(1);p1=diag[0]-x; signs=[1]
    if p1:signs.append(1 if p1>0 else -1)
    for d in diag[1:]:
        p0,p1=p1,(d-x)*p1-b*b*p0
        if p1:signs.append(1 if p1>0 else -1)
    return sum(a!=b for a,b in zip(signs,signs[1:]))
def eigen_bracket(diag,b,index):
    lo=F(-1);hi=max(diag)+2*b+1
    for _ in range(140):
        mid=(lo+hi)/2
        if sturm_count(diag,b,mid)<=index:lo=mid
        else:hi=mid
    assert sturm_count(diag,b,lo)==index and sturm_count(diag,b,hi)==index+1
    return lo,hi

def solve_tridiagonal(diag,b,rhs):
    d=diag.copy();y=rhs.copy()
    for i in range(1,len(d)):
        assert d[i-1]!=0
        factor=-b/d[i-1];d[i]-=factor*(-b);y[i]-=factor*y[i-1]
    x=[F(0)]*len(d);x[-1]=y[-1]/d[-1]
    for i in range(len(d)-2,-1,-1):x[i]=(y[i]+b*x[i+1])/d[i]
    assert apply(diag,b,x)==rhs
    return x

def certificate(N,A):
    b=A/2;diag=[F(n*(n+2))+A for n in range(N+1)]
    mp.mp.dps=75
    M=mp.matrix(N+1)
    for i,d in enumerate(diag):
        M[i,i]=mp.mpf(d.numerator)/d.denominator
        if i<N:M[i,i+1]=M[i+1,i]=-mp.mpf(b.numerator)/b.denominator
    ev,Q=mp.eigsy(M)
    w=[F(mp.nstr(Q[i,0],60)) for i in range(N+1)]
    if w[0]<0:w=[-x for x in w]
    S=dot(w,w);Hw=apply(diag,b,w);mu=dot(w,Hw)/S
    residual=[x-mu*y for x,y in zip(Hw,w)]
    eps2=(dot(residual,residual)+b*b*w[-1]*w[-1])/S
    # Tail H_Q >= D_Q >= q_N. Block Young inequality, t=q_N/2.
    q=F((N+1)*(N+3));t=q/2;delta=b*b/(q-t)
    l1,u1=eigen_bracket(diag,b,1)
    lower_excited=min(t,l1-delta)
    gap=lower_excited-mu
    assert gap>0
    energy_err=eps2/gap
    sin_bound=sqrt_up(eps2)/gap
    wl=F(1,2)*dot(w,[ (w[i-1] if i else 0)+(w[i+1] if i<N else 0) for i in range(N+1)])/S
    spec={'N':N,'ground_energy_interval':enclosure(mu-energy_err/2,energy_err/2),'wilson_expectation_interval':enclosure(wl,2*sin_bound),'spectral_gap_lower':decimal_bound(gap),'state_projector_error_upper':decimal_bound(sin_bound,50,True),'residual_norm_upper':decimal_bound(sqrt_up(eps2),50,True),'proof_arithmetic':'exact rational arithmetic with outward decimal rounding','trial_vector_rational':[rat(x) for x in w], 'exact_certificate':{'rayleigh':rat(mu),'residual_squared':rat(eps2),'finite_second_eigenvalue_lower':rat(l1),'finite_second_eigenvalue_upper':rat(u1),'infinite_first_excited_lower':rat(lower_excited),'ground_energy_error_upper':rat(energy_err),'projector_error_upper':rat(sin_bound)}}
    correlations=[]
    qv=[F(1,2)*((w[i-1] if i else 0)+(w[i+1] if i<N else 0)) for i in range(N+1)]
    for z in [F(1,2),F(1),F(2)]:
        assert z>energy_err
        y=solve_tridiagonal([d-mu+z for d in diag],b,qv)
        y_norm=sqrt_up(dot(y,y)/S)
        tail=sqrt_up((-b*y[-1]-w[-1]/2)**2/S)
        value=dot(qv,y)/S
        unconnected_error=(2*sin_bound+tail+energy_err*y_norm)/z
        connected=value-wl*wl/z
        connected_error=unconnected_error+4*sin_bound/z
        correlations.append({'z':str(z),'resolvent_correlation_interval':enclosure(value,unconnected_error),'connected_resolvent_interval':enclosure(connected,connected_error),'connected_absolute_error_upper':decimal_bound(connected_error,50,True),'exact_connected_center':rat(connected),'exact_connected_error_upper':rat(connected_error),'exact_linear_solve_verified':True})
    spec['correlations']=correlations
    return spec

def main():
    parameters=ROOT/'data/MODEL_PARAMETERS.json';p=json.loads(parameters.read_text());A=F(p['compact_gauge_action_multipliers']['SU2'])
    assert A>0
    rows=[certificate(N,A) for N in [8,12,16,24]]
    # Independently formed rational trial spaces must yield intersecting enclosures.
    for key in ['ground_energy_interval','wilson_expectation_interval']:
        assert max(F(x[key][0]) for x in rows)<=min(F(x[key][1]) for x in rows)
    for j in range(3):
        iv=[x['correlations'][j]['connected_resolvent_interval'] for x in rows]
        assert max(F(x[0]) for x in iv)<=min(F(x[1]) for x in iv)
    result={'classification':'RIGOROUS_LOCAL_SU2_ROTOR_CERTIFICATE','scope':'The fixed single-rotor sector H=D+A(I-M/2), A=32, on l2(N0). No many-body continuum or particle-mass inference.','parameters_sha256':hashlib.sha256(parameters.read_bytes()).hexdigest(),'A':str(A),'ground_state_existence_and_uniqueness':'proved analytically for all A>0 in Supplementary rotor theorem','proof_methods':['bounded perturbation and compact resolvent','irreducible Jacobi ground state','exact Sturm counts','block quadratic-form bound','Temple inequality','residual resolvent identity'],'truncation_cutoffs':[x['N'] for x in rows],'certificates':rows,'all_rational_checks_pass':True,'floating_point_role':'Proposes trial vectors; none of the interval inequalities depends on an unverified floating-point eigenvalue.'}
    R.mkdir(exist_ok=True);(R/'CERTIFIED_SU2_ROTOR.json').write_text(json.dumps(result,indent=2)+'\n')
    with (R/'CERTIFIED_SU2_ROTOR_SUMMARY.csv').open('w',newline='') as f:
        out=csv.writer(f);out.writerow(['N','E0_lower','E0_upper','W_lower','W_upper','gap_lower','projector_error_upper'])
        for x in rows:out.writerow([x['N'],*x['ground_energy_interval'],*x['wilson_expectation_interval'],x['spectral_gap_lower'],x['state_projector_error_upper']])

    selected=rows[-1]
    table=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{p{0.25\linewidth}p{0.65\linewidth}}',r'\toprule Quantity & Certified enclosure at $N=24$ \\ \midrule']
    for label,key in [(r'$E_0$','ground_energy_interval'),(r'$\langle W\rangle$','wilson_expectation_interval')]:
        lo,hi=map(F,selected[key]);table.append(label+' & $['+decimal_bound(lo,18)+', '+decimal_bound(hi,18,True)+r']$ \\')
    for row in selected['correlations']:
        lo,hi=map(F,row['connected_resolvent_interval']);table.append('$G_c('+row['z']+')$ & $['+decimal_bound(lo,18)+', '+decimal_bound(hi,18,True)+r']$ \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Local interacting rotor observables. Decimal endpoints are rounded outward from the exact-rational certificates. All quantities are dimensionless; $z$ is the positive Laplace variable.}\label{tab:certified-rotor}',r'\end{table}']
    (ROOT.parent/'paper/sections/supplement/rotor_results_table.tex').write_text('\n'.join(table)+'\n')

    print(json.dumps({'pass':True,'cutoffs':result['truncation_cutoffs'],'N24':{k:v for k,v in rows[-1].items() if k not in ['trial_vector_rational','exact_certificate','correlations']},'correlations_N24':[{k:v for k,v in x.items() if not k.startswith('exact_')} for x in rows[-1]['correlations']]},indent=2))
if __name__=='__main__':main()
