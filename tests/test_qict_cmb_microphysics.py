"""Continuous-identity tests for the QICT CMB closure; no observed likelihood."""
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from qict_cmb_microphysics import (positive_branch,receiver_fraction,evolve,
    derivative,beta,beta_gradient,cold_linear_sources)

class TestQICTMicrophysical(unittest.TestCase):
    def test_friedmann_unique(self):
        a=np.geomspace(1e-7,1,300)
        E,A,floor=positive_branch(a)
        self.assertLess(np.max(abs(E*E*(1-receiver_fraction(E))-A)/E**2),1e-11)
        self.assertTrue(np.all(np.diff(E)<0))
    def test_zero_response(self):
        a=np.geomspace(.001,1,25)
        E,A,_=positive_branch(a,lam=0)
        self.assertLess(max(abs(E*E-A)/A),1e-12)
    def test_particle_number_invariant(self):
        g=.8
        sol=evolve(g=g);x,v,rc=sol.y
        n=rc*np.exp(3*sol.t)/np.sqrt(1+g*g*x*x)
        self.assertLess(max(abs(n/n[0]-1)),1e-8)
    def test_uncoupled_dilution(self):
        sol=evolve(g=0)
        self.assertLess(max(abs(sol.y[2]*np.exp(3*sol.t)/.25-1)),1e-8)
    def test_beta_finite_difference(self):
        x=.3;g=.8;eps=1e-6
        self.assertAlmostEqual(beta(x,g),(np.log(1+g*g*(x+eps)**2)-np.log(1+g*g*(x-eps)**2))/(4*eps),places=9)
        self.assertAlmostEqual(beta_gradient(x,g),(beta(x+eps,g)-beta(x-eps,g))/(2*eps),places=9)
    def test_coupled_scalar_source(self):
        src=cold_linear_sources(.5,.1,.02,.003,.004,1e-5,3e-6,2e-6,
                                .6,.4,.2,.01,.003,.1,.05,.02)
        self.assertAlmostEqual(src[-1],.404)
        self.assertTrue(all(np.isfinite(v) for v in src))

if __name__=="__main__":
    unittest.main()
