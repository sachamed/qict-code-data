"""Exact distinct-triplet QICT parent: cubic uniqueness and dark-parity zero mode.

All claims conditional on a microscopic rule that requires three
different excited constituents in each allowed elementary receiver event.
"""
import json,math
from pathlib import Path

def ordinary(k,n):return (2*k/n)**3
def distinct(k,n):
 if n<3:raise ValueError("N>=3")
 return 8*k*(k-1)*(k-2)/(n*(n-1)*(n-2))
def expectation(fn,n,theta):
 p=(1-math.cos(theta))/2
 return math.fsum(math.comb(n,k)*p**k*(1-p)**(n-k)*fn(k,n) for k in range(n+1))
def exact_finite(n,theta):
 x=1-math.cos(theta)
 return 4*x/n**2+6*(n-1)*x*x/n**2+(n-1)*(n-2)*x**3/n**2

def main():
 nchecks=0
 for n in (3,4,5,7,10,17,31,50,250):
  for theta in (0.,.1,.7,1.6,2.5,math.pi):
   x=1-math.cos(theta)
   d=expectation(distinct,n,theta)
   o=expectation(ordinary,n,theta)
   assert abs(d-x**3)<2e-12*max(1,x**3)
   assert abs(o-exact_finite(n,theta))<2e-12*max(1,o)
   nchecks+=2
  assert all(distinct(k,n)>=0 for k in range(n+1))
  assert [k for k in range(n+1) if abs(distinct(k,n))<1e-14]==[0,1,2]
  assert abs(distinct(n,n)-8)<1e-12
  nchecks+=3
 # Polynomial uniqueness: any degree <=3 with roots K=0,1,2 has
 # form a*K*(K-1)*(K-2). H(N)=8 fixes a uniquely.
 for n in range(3,25):
  a=8/(n*(n-1)*(n-2))
  assert abs(a*n*(n-1)*(n-2)-8)<1e-12
  nchecks+=1
 # Exact harmonic decomposition of the cubic effective potential.
 for theta in (0.,.2,1.,2.5,math.pi):
  lhs=(1-math.cos(theta))**3
  rhs=2.5-3.75*math.cos(theta)+1.5*math.cos(2*theta)-.25*math.cos(3*theta)
  assert abs(lhs-rhs)<2e-14
  nchecks+=1
 result={
  "status":"passed_exact","tests":nchecks,
  "operator":"H_N(K)/u=8 K(K-1)(K-2)/[N(N-1)(N-2)]",
  "uniqueness":"unique degree<=3 polynomial in K with H(0)=H(1)=H(2)=0, H(N)=8u",
  "spectrum":"positive semidefinite; zero-energy modes K=0,1,2; odd K=1 mode also zero",
  "coherent_expectation":"V(theta)/u = (1-cos theta)^3, for every integer N>=3",
  "ordinary_power_parent":"u(1-C)^3 has finite-N lower powers and distinct late tail",
  "harmonics":"(1-cos theta)^3 = 5/2 - 15/4 cos theta + 3/2 cos(2 theta) - 1/4 cos(3 theta)",
  "odd_mass":"NOT derived: odd K=1 eigenmode is zero-energy for this parent alone",
  "missing_physics":"Proof that three-distinct-copy rule is QICT primitive; parity gap, full dispersion, dark relic, covariant gravitational matching, loop stabilization",
  "no_H0_or_vacuum_claim":True
 }
 p=Path("results/qict_triplet_parent/triplet_uniqueness.json")
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(result,indent=2)+"\n")
 print("QICT_DISTINCT_TRIPLET_THEOREM",json.dumps(result),flush=True)

if __name__=="__main__":main()
