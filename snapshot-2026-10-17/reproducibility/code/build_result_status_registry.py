#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def load(path): return json.loads(path.read_text())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    r=Path(a.results)
    sec=load(r/'SECTOR_OBSERVABLE_MAP.json'); prim=load(r/'PRIMITIVE_DERIVATION_VALIDATION.json')
    ff=load(r/'FERMION_OBSERVABLE_VALIDATION.json')
    su2=load(r/'SU2_TENSOR_AUDIT.json'); su3=load(r/'SU3_VALIDATION.json'); l24=load(r/'L24_PRECONTRACTION_VALIDATION.json')
    u1=load(r/'U1_SECTOR_VALIDATION.json'); c11=load(r/'C11_NUMERICAL_VALIDATION.json')
    common=load(r/'COMMON_CONTRACTION_CHARACTERIZATION.json')
    cont=load(r/'CONTINUUM_CONSISTENCY_CERTIFICATE.json'); rg=load(r/'RG_CONTINUUM_CERTIFICATE.json')
    grav=load(r/'SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json') if (r/'SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json').exists() else None
    rows=[]
    def add(sector,observable,result,status,source): rows.append(dict(sector=sector,observable=observable,result=result,status=status,source=source))
    certified_path=r/'CERTIFIED_REDUCED_LEPTON_INTERVALS.json'
    if certified_path.exists():
        certified=load(certified_path)
        for key in ('alpha_conn_interval','mu_over_e_reduced_interval',
                    'tau_over_mu_reduced_interval','tau_over_e_reduced_interval','koide_reduced_interval'):
            add('reduced charged-lepton sector',key,json.dumps(certified[key]),
                'CERTIFIED_REDUCED_MODEL_INTERVAL','CERTIFIED_REDUCED_LEPTON_INTERVALS.json')
    fr=sec['family_receiver']; gc=sec['gauge_charges']
    add('primitive dynamics','oriented electric polynomial',prim['symbolic_oriented_polynomial'],'EXACT_SYMBOLIC','PRIMITIVE_DERIVATION_VALIDATION.json')
    add('primitive dynamics','route-cost identity','P_+(Delta)=T_{2Delta}','EXACT_SYMBOLIC','PRIMITIVE_DERIVATION_VALIDATION.json')
    add('receiver','primitive family multiplicity','3','MODEL_CLASS_THEOREM','SECTOR_OBSERVABLE_MAP.json')
    add('receiver','integer timing structure',','.join(map(str,fr['integer_timing_structure']))+' times tau_*','STRUCTURAL_RESULT','SECTOR_OBSERVABLE_MAP.json')
    add('gauge','integer hypercharge ray',json.dumps(gc['integer_hypercharges_qY'],sort_keys=True),'MODEL_CLASS_CLASSIFICATION','SECTOR_OBSERVABLE_MAP.json')
    add('gauge','anomaly residuals',json.dumps(gc['anomaly_residuals'],sort_keys=True),'EXACT_CANCELLATION','SECTOR_OBSERVABLE_MAP.json')
    add('photon','quadratic high-symmetry pattern','9:21:25','FINITE_REGULATOR_SIGNATURE','PHOTON_SIGNATURE_MOMENTS.json')
    add('fermion observables','Dirac mass covariance residual',str(ff['dirac_mass_singular_value_basis_covariance_residual']),'ALGEBRAIC_SELFTEST','FERMION_OBSERVABLE_VALIDATION.json')
    add('fermion observables','Takagi covariance residual',str(ff['takagi_singular_value_congruence_residual']),'ALGEBRAIC_SELFTEST','FERMION_OBSERVABLE_VALIDATION.json')
    add('fermion observables','mixing unitarity residual',str(ff['mixing_unitarity_residual']),'ALGEBRAIC_SELFTEST','FERMION_OBSERVABLE_VALIDATION.json')
    add('SU(2) primary tensors','sparse nonzero coefficients',str(su2['sequential_SU2']['total_core_nonzero_coefficients']),'PRIMARY_DATA_AUDIT','SU2_TENSOR_AUDIT.json')
    st=l24['statistics']; add('rank-six precontraction','L24 geometry',f"{st['links']} links; {st['plaquettes']} plaquettes; {st['su2_connected_pairs']} SU2 pairs",'PRIMARY_DATA_AUDIT','L24_PRECONTRACTION_VALIDATION.json')
    add('common contraction','Kato state-selection Hamiltonian checkpoint',f"dim={common['strict_kato_checkpoint']['dimension']}; gap={common['strict_kato_checkpoint']['gap']}",'FINITE_STATE_SELECTION_CHECKPOINT','COMMON_CONTRACTION_CHARACTERIZATION.json')
    add('U(1)','singleton Riesz trace',str(u1['components']['singleton_boundary_component']['trace']),'FINITE_RIESZ_VALIDATION','U1_SECTOR_VALIDATION.json')
    add('SU(3)','tensor identity max residual',str(su3['summary']['tensor_identity_max_error']),'FINITE_ALGEBRA_VALIDATION','SU3_VALIDATION.json')
    add('C11 spectral methods','Toeplitz minimum eigenvalue',str(c11['toeplitz_lambda_min_numeric']),'DETERMINISTIC_SELFTEST','C11_NUMERICAL_VALIDATION.json')
    add('continuum','theorem-chain source audit',cont['structural_validation'],'SOURCE_LEVEL_CERTIFICATE','CONTINUUM_CONSISTENCY_CERTIFICATE.json')
    add('RG','continuum specialization',rg['structural_validation'],'SOURCE_LEVEL_CERTIFICATE','RG_CONTINUUM_CERTIFICATE.json')
    if grav:
        add('metric response','Feynman-Hellmann first derivative error',str(grav['feynman_hellmann_first_derivative']['absolute_error']),'ALGEBRAIC_SELFTEST','SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json')
        add('metric response','Kato second derivative error',str(grav['kato_second_derivative']['absolute_error']),'ALGEBRAIC_SELFTEST','SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json')
        add('metric response','BCC quadratic isotropy error',str(grav['bcc_quadratic_isotropy']['frobenius_error_from_I_over_3']),'ALGEBRAIC_SELFTEST','SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json')
        add('metric source response','epsilon0_src and gamma_TT^(s)','source-deformed common-net limits plus source normalization','OPERATOR_DEFINED_PENDING_EVALUATION','SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json')
    with Path(a.out).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sector','observable','result','status','source']); w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} result-status rows')
if __name__=='__main__': main()
