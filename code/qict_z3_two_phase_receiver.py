"""Conditional Z3 three-channel receiver: two physical relative phases.

The identification of endpoint classes with phase-bearing channels is an
additional microscopic hypothesis and is not a theorem of QICT.
"""
import json, math
from pathlib import Path
import numpy as np

N=250
M=np.array([[-1,1],[-1,0]],dtype=int)
R=np.array([[0,1],[1,0]],dtype=int)  # swap receiver channels 1 and 2
LINK=np.array([[1,0],[-1,1],[0,-1]],dtype=int)
A=np.array([[2.,-1.],[-1.,2.]])
G=A/3

def coeffs(n):
    if int(n)!=n or n<3:raise ValueError("N integer >= 3")
    return 4/n**2,6*(n-1)/n**2,(n-1)*(n-2)/n**2

def V(q,exact=False):
    a,b,c=coeffs(N)
    th=LINK@np.asarray(q,float)
    x=1-np.cos(th)
    return float(np.sum(x**3 if exact else a*x+b*x*x+c*x**3))

def grad(q):
    a,b,c=coeffs(N)
    th=LINK@np.asarray(q,float);x=1-np.cos(th)
    return LINK.T@((a+2*b*x+3*c*x*x)*np.sin(th))

def hessian(q):
    a,b,c=coeffs(N)
    th=LINK@np.asarray(q,float);x=1-np.cos(th)
    p1=a+2*b*x+3*c*x*x;p2=2*b+6*c*x
    return LINK.T@np.diag(p2*np.sin(th)**2+p1*np.cos(th))@LINK

def sign_operator(q):
    """Cyclic invariant, reflection-odd cubic composite of three link phases."""
    a,b=np.asarray(q,float)
    return float(np.sin(a)+np.sin(b-a)-np.sin(b))

def main():
    assert np.array_equal(M@M@M,np.eye(2,dtype=int))
    assert np.array_equal(R@R,np.eye(2,dtype=int))
    assert np.array_equal(R@M@R,M@M)
    assert np.array_equal(R.T@A@R,A)
    assert np.array_equal(M.T@A@M,A)
    assert np.linalg.matrix_rank(G)==2
    assert np.all(np.linalg.eigvalsh(G)>0)
    checks=7
    for q in ([0,0],[.02,.3],[.4,1.1],[1.3,-.9],[2.7,.8]):
      q=np.array(q,float)
      for exact in (False,True):
        assert np.isclose(V(q,exact),V(M@q,exact),atol=1e-12);checks+=1
        assert np.isclose(V(q,exact),V(R@q,exact),atol=1e-12);checks+=1
      assert abs(sign_operator(M@q)-sign_operator(q))<1e-12
      assert abs(sign_operator(R@q)+sign_operator(q))<1e-12
      checks+=2
      assert np.allclose(grad(M@q),np.linalg.solve(M.T,grad(q)),atol=1e-12)
      assert np.allclose(hessian(M@q),np.linalg.solve(M.T,hessian(q))@np.linalg.inv(M),atol=1e-12)
      assert V(q)>=-1e-12;checks+=3
    H=hessian([0,0])
    masses=np.linalg.eigvals(np.linalg.solve(G,H))
    expected=12/(N*N)
    assert np.allclose(masses,[expected,expected],rtol=1e-12)
    assert np.allclose(H,coeffs(N)[0]*A)
    assert np.allclose(R.T@H@R,H);checks+=3
    for th in (.2,1.,2.5,3.):
      assert abs((1-math.cos(th))**3-(2.5-3.75*math.cos(th)+1.5*math.cos(2*th)-.25*math.cos(3*th)))<5e-14
      checks+=1
    assert abs(hessian([.2,.8])[0,1])>1e-6;checks+=1
    result={
      "status":"passed_exact_Z3_two_phase_gate","tests":checks,"N":N,
      "microscopic_hypothesis":"three independent phase-bearing channels plus one redundant diagonal U(1)",
      "hypothesis_not_derived_from_QICT":True,
      "relative_phase_space":"U(1)^3/U(1)_diag = T²",
      "Z3_rotation":M.tolist(),"channel_reflection":R.tolist(),
      "conditional_full_group":"D3 is isomorphic to S3 if channel reflection is an exact microscopic involution",
      "reflection_odd_C3_invariant_composite":"sin(alpha)+sin(beta-alpha)-sin(beta)=4 sin(alpha/2)sin((beta-alpha)/2)sin(beta/2)",
      "composite_particle_warning":"a reflection-odd composite operator is not proof of an elementary stable dark matter particle",
      "kinetic_metric":G.tolist(),
      "link_angles":"alpha,beta-alpha,-beta",
      "invariant_potential":"sum over links of P_N(1-cos(angle)); P_N finite-charge coherent polynomial",
      "finite_N_coefficients":coeffs(N),
      "Hessian_at_symmetric_vacuum":H.tolist(),
      "canonical_mass2_eigenvalues":sorted(np.real(masses).tolist()),
      "Z3_enforced_degeneracy":True,
      "true_two_field_vs_harmonics":"Three Fourier harmonics of a single phase are not three independent fields",
      "cubic_triplet_limit":"P_N(x) replaced by x^3 and both vacuum curvatures vanish",
      "two_staggered_activation_episodes_require":"different initial conditions, nonlinear dynamics, or independently derived controlled Z3 breaking",
      "does_not_prove":["three physical channels in original QICT","two-field EDE fit","H0 tension","dark matter mass","relic density","late vacuum"],
      "existing_literature":"M. Bella et al., arXiv:2604.13535 (2026), independently investigated multifield EDE"
    }
    p=Path("results/qict_z3_two_phase/z3_exact.json");p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+"\n")
    print("QICT_Z3_TWO_PHASE_THEOREM",json.dumps(result),flush=True)

if __name__=="__main__":main()
