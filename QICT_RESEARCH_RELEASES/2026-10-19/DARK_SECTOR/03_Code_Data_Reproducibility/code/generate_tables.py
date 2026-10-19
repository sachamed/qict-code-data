"""Regenerate numerical table rows and the prediction register from result data."""
from pathlib import Path
from decimal import Decimal, ROUND_FLOOR
import csv,json,re

P=Path(__file__).resolve().parents[1]
def load(p):return json.loads((P/p).read_text())

def main():
    moments=load('code/operator_closure_results.json')['regulators']
    exact=load('data/EXACT_MOMENT_POLYNOMIAL_CERTIFICATE.json')['blocks']
    source=P/'../01_Main_Manuscript/sections/01_spectral.tex';text=source.read_text()
    tables=list(re.finditer(r'\\begin\{table\}.*?\\end\{table\}',text,re.S))
    assert len(tables)==3
    rows=[]
    rows.append('\n'.join(f"{r['regulator']} & {r['neutral_carrier_dimension']} & {r['min_eigenvalue']:.9f} & {r['max_eigenvalue']:.9f}\\\\" for r in moments))
    rows.append('\n'.join(f"{r['regulator']} & {r['inverse_toeplitz_atom_min_eigenvalue']:.10f} & {r['inverse_toeplitz_atom_max_eigenvalue']:.10f} & {r['fejer_to_optimized_max_ratio']:.2f}\\\\" for r in moments))
    rows.append('\n'.join(f"R{r} & "+next(b['atom_weight_upper_outward_12dp'] for b in exact if b['regulator']==f'R{r}' and b['source_group']=='left')+' & '+next(b['atom_weight_upper_outward_12dp'] for b in exact if b['regulator']==f'R{r}' and b['source_group']=='right')+r'\\' for r in range(1,5)))
    for match,body in reversed(list(zip(tables,rows))):
        table=match.group()
        a=table.index(r'\midrule')+len(r'\midrule');b=table.index(r'\bottomrule',a)
        table=table[:a]+'\n'+body+'\n'+table[b:]
        text=text[:match.start()]+table+text[match.end():]
    source.write_text(text)

    def lower_sci(x):
        value=Decimal.from_float(x)
        exponent=value.adjusted()
        mantissa=value.scaleb(-exponent).quantize(Decimal('0.00001'),rounding=ROUND_FLOOR)
        return f'{mantissa}'+r'\times10^{'+str(exponent)+'}'
    certified=load('data/OPERATOR_SPECTRAL_CERTIFICATE.json')['regulators']
    source=P/'../01_Main_Manuscript/sections/01b_operator_spectra.tex';text=source.read_text()
    rows='\n'.join(row['regulator']+' & '+str(row['left']['candidate_count'])+' & $'+
                   lower_sci(row['left']['unitary_chord_gap_lower'])+'$ & '+
                   str(row['right']['candidate_count'])+' & $'+
                   lower_sci(row['right']['unitary_chord_gap_lower'])+r'$\\'
                   for row in certified)
    match=re.search(r'\\begin\{table\}.*?\\end\{table\}',text,re.S)
    assert match is not None
    table=match.group();a=table.index(r'\midrule')+len(r'\midrule');b=table.index(r'\bottomrule',a)
    table=table[:a]+'\n'+rows+'\n'+table[b:]
    source.write_text(text[:match.start()]+table+text[match.end():])

    base=load('code/results.json');thermal=load('code/closure_results.json')
    portal=load('data/PORTAL_THERMAL_RESULTS.json');growth=load('data/metric_growth_galaxy.json')
    extended=load('data/ANALYTIC_EXTENSION_RESULTS.json')
    updates={
        'SP13':r'R4 chord gaps $g_L\geq'+lower_sci(certified[-1]['left']['unitary_chord_gap_lower'])+
                r'$, $g_R\geq'+lower_sci(certified[-1]['right']['unitary_chord_gap_lower'])+'$',
        'LSS4':f"Spherical collapse threshold $a={base['spherical_top_hat']['collapse_scale_factor']:.7f}$",
        'GAL2':r'$\xi_g(10\Mp)='+f"{base['galaxy_correlation']['xi_g_at_10Mpc']:.6f}"+r'$ for $b_g=1.5$',
        'DM9':r'Constant-rate freeze-out $\Omega_\chi h^2='+f"{thermal['freezeout']['omega_h2']:.6f}"+'$',
        'DM11':r'$\Omega_Sh^2='+f"{portal['Omega_h2']:.7f}"+r'$ at $m_S=300$ GeV, $\lambda_{HS}=0.20$',
        'LEN6':r'Angular convergence ratio at $\ell=200$: $'+f"{growth['growth']['angular_convergence_B_over_GR_equal_initial_density']['200']:.7f}"+'$',
        'GR4':r'$a_1='+f"{extended['recorded_metric_response']['taylor_coefficients_a0_to_a3'][1]:.8f}"+r'$, $a_2='+f"{extended['recorded_metric_response']['taylor_coefficients_a0_to_a3'][2]:.8f}"+r'$, analytic disk $|s|<'+f"{extended['recorded_metric_response']['guaranteed_analytic_disk_radius_s']:.4f}"+'$',
        'DM14':r'$\partial\ln Y/\partial\ln T_{\rm RH}='+f"{extended['thermal']['finite_reheating_sensitivity'][-1]['dlogY_dlogT_reheat']:.5f}"+r'$ at $x_{\rm RH}=10$',
        'GAL6':r'$M_w(3\,\mathrm{Gyr})='+f"{extended['cosmological_inversions']['gas_mass_conservation']['cumulative_wind_Msun']/1e9:.5f}"+r'\times10^9M_\odot$',
    }
    # The CSV is the canonical register. The standalone PDF resolves its document
    # references during the build, so the article need not duplicate this table.
    with (P/'data/PREDICTION_LEDGER.csv').open(newline='') as stream:
        records=[{key:row[key] for key in
                  ('id','quantity_latex','domain_and_provenance_latex')}
                 for row in csv.DictReader(stream) if not row['id'].startswith('SI')]
    by_id={row['id']:row for row in records}
    for key,value in updates.items():
        by_id[key]['quantity_latex']=value
    assert len({r['id'] for r in records})==len(records)
    with (P/'data/PREDICTION_LEDGER.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    print(f'Regenerated four numerical tables; prediction register: {len(records)} entries.')

if __name__=='__main__':main()
