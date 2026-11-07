#!/usr/bin/env python3
"""Patch an independently obtained kabeleh/iDM CLASS fork for a QICT test branch.

Not a stand-alone CLASS implementation. Never run against an unreviewed fork:
the target is the public CLASS 3.3.4 interacting-quintessence fork at
https://github.com/kabeleh/iDM (GPL-3.0); source remains in that project.
The script refuses changes if expected anchors do not match.

QICT mode is enabled ONLY by scf_potential=QICT_finite_N while
model_cdm=i (the existing variable-mass CDM path in that fork).

Parameters: scf_parameters = [V0, F, N, reserved,
                             0,0,0,0,0,0,phi_ini,phi'_ini]
cdm_c is here reinterpreted as the nonnegative mass-modulation g.
All 2-parameter x1,x2 derivatives below are physical derivatives in the
fork's scalar-field normalization. Primary CMB spectra are not valid until
the CLASS gauge and Einstein-constraint tests pass.
"""
import sys,re,pathlib

root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve()
def file(n):return root/n

def replace_once(s,old,new,tag):
    c=s.count(old)
    if c!=1:raise RuntimeError(f"{tag}: expected one anchor, got {c}")
    return s.replace(old,new,1)
def edit(n,fn):
    p=file(n)
    old=p.read_text()
    new=fn(old)
    if old==new:raise RuntimeError(f"No changes in {n}")
    p.write_text(new)
    print("PATCHED",n,flush=True)

# Public functions shared by background and first-order Einstein perturbations.
helper=r"""
/* QICT conditional microscopic sector.
 * phi and F use the scalar field units already implemented by this fork.
 * For the model-specific case g=0 all derivatives vanish exactly.
 */
double qict_mass_factor(struct background *pba,double phi) {
  double g=pba->cdm_c,F=pba->scf_parameters[1];
  double s=sin(phi/(2.0*F));
  return sqrt(1.0+g*s*s);
}
double qict_beta(struct background *pba,double phi) {
  double g=pba->cdm_c,F=pba->scf_parameters[1];
  double theta=phi/F,s=sin(theta);
  double D=1.0+g*sin(theta/2.0)*sin(theta/2.0);
  return g*s/(4.0*F*D);
}
double qict_msecond_over_m(struct background *pba,double phi) {
  double g=pba->cdm_c,F=pba->scf_parameters[1];
  double th=phi/F,s=sin(th),c=cos(th);
  double D=1.0+g*sin(th/2.0)*sin(th/2.0);
  double b=g*s/(4.0*F*D);
  double bp=g/(4.0*F*F)*(c/D-g*s*s/(2.0*D*D));
  return bp+b*b;
}
"""
def header(s):
    marker="    /** DM–DE Interactions (KBL) **/"
    add=marker+"\n    double qict_mass_factor(struct background *pba,double phi);\n    double qict_beta(struct background *pba,double phi);\n    double qict_msecond_over_m(struct background *pba,double phi);"
    return replace_once(s,marker,add,"header")
edit("include/background.h",header)

def background(s):
    s=replace_once(s,"double rho_cdm_prime(\n",helper+"\ndouble rho_cdm_prime(\n","helper")
    original="    return -pba->cdm_c * (1. + tanh(pba->cdm_c * phi)) * pvecback[pba->index_bg_rho_cdm];"
    s=replace_once(s,original,
       "    if (pba->scf_potential == 10) return qict_beta(pba,phi)*pvecback[pba->index_bg_rho_cdm];\n"+original,
       "rho prime")
    target="pba->Omega0_cdm * H0_sq / a3 * pba->cdm_f_phi0_inv / (1.0 + exp(2.0 * pba->cdm_c * pvecback_B[pba->index_bi_phi_scf]));"
    s=replace_once(s,target,
       "pba->Omega0_cdm * H0_sq / a3 * pba->cdm_f_phi0_inv *\n          ((pba->scf_potential == 10) ?\n            qict_mass_factor(pba,pvecback_B[pba->index_bi_phi_scf]) :\n            (1.0 / (1.0 + exp(2.0*pba->cdm_c*pvecback_B[pba->index_bi_phi_scf]))));",
       "rho cdm")
    # Correct present-day number normalization for g=0 QICT and for g!=0.
    s,n=re.subn(r"pba->cdm_f_phi0 = 0\.5;\s+pba->cdm_f_phi0_inv = 2\.0;",
      "pba->cdm_f_phi0 = (pba->scf_potential == 10) ? 1.0 : 0.5;\\n       pba->cdm_f_phi0_inv = (pba->scf_potential == 10) ? 1.0 : 2.0;",s)
    if n!=1:raise RuntimeError(f"uncoupled normalization: expected one anchor got {n}")
    s=replace_once(s,"double f_new = 1.0 / (1.0 + exp_2c_phi0); /* = G(f_cur) */",
      "double f_new = (pba->scf_potential == 10) ?\n           qict_mass_factor(pba,phi_0) : (1.0/(1.0+exp_2c_phi0));",
      "mass renormalization")
    s=replace_once(s,"pba->cdm_f_phi0_inv = 1.0 + exp_2c_phi0; /* = 1/f(phi_0), for the hot path */",
      "pba->cdm_f_phi0_inv = (pba->scf_potential == 10) ? (1.0/f_new) : (1.0+exp_2c_phi0);",
      "mass renormalization convergence")
    s=replace_once(s,"if (f_next <= 0.0 || f_next > 1.0 || !isfinite(f_next))",
      "if (f_next <= 0.0 || f_next > ((pba->scf_potential == 10) ? sqrt(1.0+pba->cdm_c) : 1.0) || !isfinite(f_next))",
      "mass renormalization safeguard")
    # New positive finite-N spin-coherent potential; c1=V0, c2=F,c3=N.
    marker="  default:\n    *V = 0.;"
    add=r"""  case 10: /* QICT: V = V0 (c1*x+c2*x^2+c3*x^3), x=1-cos(phi/F). */
  {
    double V0=c1,F=c2,N=c3,th=phi/F;
    double co=cos(th),si=sin(th),x=1.0-co;
    double c_linear=4.0/(N*N),c_square=6.0*(N-1.0)/(N*N);
    double c_cubic=(N-1.0)*(N-2.0)/(N*N);
    double Px=c_linear+2.0*c_square*x+3.0*c_cubic*x*x;
    double Pxx=2.0*c_square+6.0*c_cubic*x;
    double Pxxx=6.0*c_cubic;
    double xp=si/F,xpp=co/(F*F),xppp=-si/(F*F*F);
    double xpppp=-co/(F*F*F*F);
    *V=V0*(c_linear*x+c_square*x*x+c_cubic*x*x*x);
    *dV=V0*Px*xp;
    *ddV=V0*(Pxx*xp*xp+Px*xpp);
    if (d3V) *d3V=V0*(Pxxx*xp*xp*xp+3.0*Pxx*xp*xpp+Px*xppp);
    if (d4V) *d4V=V0*(6.0*Pxxx*xp*xp*xpp+
          3.0*Pxx*xpp*xpp+4.0*Pxx*xp*xppp+Px*xpppp);
    break;
  }
  default:
    *V = 0.;"""
    s=replace_once(s,marker,add,"potential")
    return s
edit("source/background.c",background)

def input_f(s):
    needle="       else if (strstr(string1, \"DoubleExp\") != NULL)\n       {\n         pba->scf_potential = 9;\n       }"
    if needle not in s: needle=needle.replace("       ","      ")\n    addition=needle+"\n       else if (strstr(string1, \"QICT_finite_N\") != NULL)\n       {\n         pba->scf_potential = 10;\n       }"
    s=replace_once(s,needle,addition,"input potential enum")
    marker="    /** 8.b.2) SCF initial conditions from attractor solution */"
    guard=r"""    /* Strict QICT test branch: fixed charge N>=3 and nonnegative mass coupling.
     * scf_parameters uses V0,F,N,c4, 4 q's, 2 exponents, phi0,phi'0.
     */
    if (pba->scf_potential == 10) {
      class_test(pba->scf_parameters_size != 12,errmsg,
          "QICT requires exactly 12 scf_parameters");
      class_test(pba->scf_parameters[0] < 0.0 ||
                 pba->scf_parameters[1] <= 0.0 ||
                 pba->scf_parameters[2] < 3.0 ||
                 pba->cdm_c < 0.0,errmsg,
          "QICT requires V0>=0, F>0, N>=3 and g=cdm_c>=0");
      class_test(pba->scf_parameters[4]!=0.0 ||
                 pba->scf_parameters[5]!=0.0 ||
                 pba->scf_parameters[6]!=0.0 ||
                 pba->scf_parameters[7]!=0.0,errmsg,
          "QICT forbids extra phenomenological scalar-DM coupling q1..q4");
    }
"""
    return replace_once(s,marker,guard+"\n"+marker,"input potential validation")
edit("source/input.c",input_f)

def perturbations(s):
    # Use QICT beta wherever the fluid perturbation's variable mass is evaluated.
    old="(-pba->cdm_c * (1 + tanh(pba->cdm_c * ppw->pvecback[pba->index_bg_phi_scf])))"
    new="((pba->scf_potential==10) ? qict_beta(pba,ppw->pvecback[pba->index_bg_phi_scf]) : (-pba->cdm_c*(1+tanh(pba->cdm_c*ppw->pvecback[pba->index_bg_phi_scf]))))"
    n=s.count(old)
    if n<2:raise RuntimeError(f"delta stress anchors {n}<2")
    s=s.replace(old,new)
    old="dy[pv->index_pt_theta_cdm] += (pba->cdm_c * (1 + tanh(pba->cdm_c * ppw->pvecback[pba->index_bg_phi_scf]))) * (-k2 * y[pv->index_pt_phi_scf] + y[pv->index_pt_theta_cdm] * pvecback[pba->index_bg_phi_prime_scf]);"
    new=r"""if (pba->scf_potential == 10) {
             double b = qict_beta(pba,ppw->pvecback[pba->index_bg_phi_scf]);
             dy[pv->index_pt_theta_cdm] += b * (k2*y[pv->index_pt_phi_scf]
                     - y[pv->index_pt_theta_cdm]*pvecback[pba->index_bg_phi_prime_scf]);
           } else {
             dy[pv->index_pt_theta_cdm] += (pba->cdm_c * (1 + tanh(pba->cdm_c * ppw->pvecback[pba->index_bg_phi_scf]))) * (-k2 * y[pv->index_pt_phi_scf] + y[pv->index_pt_theta_cdm] * pvecback[pba->index_bg_phi_prime_scf]);
           }"""
    s=replace_once(s,old,new,"QICT Euler")
    old=r"""        cosh_c_phi = cosh(pba->cdm_c * pvecback[pba->index_bg_phi_scf]);
        tanh_c_phi = tanh(pba->cdm_c * pvecback[pba->index_bg_phi_scf]);"""
    new=r"""        if (pba->scf_potential == 10) {
          double phi=pvecback[pba->index_bg_phi_scf];
          double b=qict_beta(pba,phi);
          double mb2=qict_msecond_over_m(pba,phi);
          dy[pv->index_pt_mom_scf] += -a2*pvecback[pba->index_bg_rho_cdm] *
             (mb2*y[pv->index_pt_phi_scf] +
              b*y[pv->index_pt_delta_cdm] +
              2.0*pvecmetric[ppw->index_mt_psi]*b);
        } else {
        cosh_c_phi = cosh(pba->cdm_c * pvecback[pba->index_bg_phi_scf]);
        tanh_c_phi = tanh(pba->cdm_c * pvecback[pba->index_bg_phi_scf]);"""
    s=replace_once(s,old,new,"QICT perturbated KG")
    # The original hyperbolic branch ends with two consecutive braces:
    # close our new ELSE directly after the specific +a² source expression.
    old="+ 2 * pvecmetric[ppw->index_mt_psi] * pvecback[pba->index_bg_rho_cdm] * (-pba->cdm_c * (1 + tanh_c_phi)));                    //-2a^2 psi *  m' n, Psi in CLASS = phi in xPand"
    new=old+"\n        } /* end legacy hyperbolic mass branch */"
    s=replace_once(s,old,new,"QICT KG closing")
    return s
edit("source/perturbations.c",perturbations)
print("QICT_PATCH_COMPLETED",flush=True)
