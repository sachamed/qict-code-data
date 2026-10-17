#!/usr/bin/env python3
from fractions import Fraction as F
from pathlib import Path
import cmath,json,math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'certificates/REPRESENTATION_KERNEL_Z6.json'
# LH Weyl multiplets: triality mod 3, SU2 parity, integer hypercharge y=6Y
reps={
 'Q':(1,1,1), 'uc':(-1,0,-4), 'dc':(-1,0,2),
 'L':(0,1,-3), 'ec':(0,0,6), 'nuc':(0,0,0)
}
# enumerate center elements with theta=n*pi/3, a mod3, b mod2 and find kernel
kernel=[]
residuals={}
for a in range(3):
    for b in range(2):
        for n in range(6):
            theta=n*math.pi/3
            errs={}
            for name,(tri,par,y) in reps.items():
                z=cmath.exp(2j*math.pi*a*tri/3)*((-1)**(b*par))*cmath.exp(1j*theta*y)
                errs[name]=abs(z-1)
            if max(errs.values())<1e-12:
                kernel.append((a,b,n)); residuals[f'{a},{b},{n}']=errs
expected=[(n%3,n%2,n) for n in range(6)]
assert sorted(kernel)==sorted(expected)
out={
 'status':'PASS_EXACT_REPRESENTATION_KERNEL_Z6',
 'integer_hypercharge_convention':'y=6Y',
 'multiplets':{k:list(v) for k,v in reps.items()},
 'kernel_elements_a_b_n':kernel,
 'generator':'xi=(omega_3,-1,exp(i*pi/3))',
 'kernel_order':len(kernel),
 'theorem':'The central kernel of the declared matter representation is exactly <xi> ~= Z6; therefore the faithful global gauge group is the quotient by this kernel.',
 'max_numerical_residual':max(max(v.values()) for v in residuals.values())
}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
