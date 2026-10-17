#!/usr/bin/env python3
from __future__ import annotations
import math
from fractions import Fraction
import numpy as np

omega = np.exp(2j*np.pi/3)
I3 = np.eye(3,dtype=complex)
X = np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=complex)
Z = np.diag([1,omega,omega**2])
W = [I3,X,Z]

def electron_route_coefficients():
    dvals=[10,9,8]
    eps={10:Fraction(1,210),9:Fraction(1,63),8:Fraction(5,168)}
    phi=Fraction(1,63); cL,cH=18,24
    c=[]
    for d in dvals:
        ph=phi*(d*cL+(21-d)*cH)+eps[d]
        c.append(np.exp(-1j*np.pi*float(ph)))
    return np.array(c,dtype=complex)

def receiver_return(coeff=None, holonomy=None):
    if coeff is None: coeff=electron_route_coefficients()
    if holonomy is None: holonomy=np.ones(3,dtype=complex)
    return sum(coeff[a]*holonomy[a]*W[a] for a in range(3))/3

def singular_angles(T):
    s=np.linalg.svd(T,compute_uv=False)
    s=np.sort(s)
    return s,np.arcsin(np.clip(s,0,1))

def route_unitary(coeff=None):
    if coeff is None: coeff=electron_route_coefficients()
    U=np.zeros((9,9),dtype=complex)
    for a in range(3): U[3*a:3*a+3,3*a:3*a+3]=coeff[a]*W[a]
    return U

def analytic_zero_holonomy(coeff=None):
    # Choose a,b,c after holonomy so that b^3=omega a^3 and c^3=omega^2 a^3.
    # Then a^3+b^3+c^3=0 exactly in algebra.
    if coeff is None: coeff=electron_route_coefficients()
    a=coeff[0]
    b_target=a*np.exp(2j*np.pi/9)
    c_target=a*np.exp(4j*np.pi/9)
    h=np.array([1.0+0j,b_target/coeff[1],c_target/coeff[2]],dtype=complex)
    return h

def postcompression_phase(theta,alpha):
    y=np.cos(theta)*np.cos(alpha)
    return np.arccos(np.clip(y,-1.0,1.0))
