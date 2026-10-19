"""Build all three documents in temporary directories, publishing final PDFs only.

The standalone register resolves cross-references from the current main/SI
compilations. Its source and CSV remain usable independently of the build tree.
"""
from pathlib import Path
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / '03_Code_Data_Reproducibility/data'


def compile_document(folder, stem, target):
    shutil.copytree(folder, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.aux', '*.log', '*.out',
                                                  '*.staged', stem + '.pdf'))
    for _ in range(3):
        run = subprocess.run(['pdflatex', '-halt-on-error',
                              '-interaction=nonstopmode', stem + '.tex'],
                             cwd=target, capture_output=True, text=True,
                             errors='replace')
        if run.returncode:
            raise RuntimeError(str(folder) + '\n' + run.stdout[-14000:] + run.stderr)
    log = (target / (stem + '.log')).read_text(errors='replace')
    issues = [line for line in log.splitlines()
              if re.search(r'undefined|multiply defined|Overfull|^!|Fatal error',
                           line, re.I)]
    if issues:
        raise RuntimeError(str(folder) + '\n' + '\n'.join(issues))
    labels = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',
                             (target / (stem + '.aux')).read_text()))
    return (target / (stem + '.pdf')).read_bytes(), labels


def support_for(identifier):
    prefix = re.match(r'[A-Z]+', identifier).group()
    number = int(re.search(r'\d+', identifier).group())
    if prefix == 'SP':
        if number <= 3:
            return 'Finite moments; exact recorded-data certificate'
        if number <= 6:
            return 'Analytic proof; numerical matrix controls'
        if 12 <= number <= 15:
            return 'Analytic proof; outward-rounded operator certificate'
        if number == 16:
            return 'Analytic proof; independent moment reconstruction'
        return 'Analytic proof under the stated hypotheses'
    if prefix == 'MAT':
        return 'Analytic proof or explicit construction'
    if prefix == 'DM':
        if number == 1:
            return 'Declared effective-model input'
        if number in (2, 7):
            return 'Rate calculation; published LZ comparison'
        if number in (10, 12, 13, 14):
            return 'Analytic derivation; thermal numerical controls'
        if number == 15:
            return 'Analytic stability proof'
        return 'Specified thermal or scattering model calculation'
    if prefix == 'DE':
        return ('Conditional physical identity' if number in (1, 2, 5)
                else 'Reconstruction from stated observational inputs')
    if prefix == 'CMB':
        return ('Transfer-functional identity or analytic proof' if number >= 5
                else 'Reference-background numerical calculation')
    if prefix == 'GR':
        return ('Analytic proof; finite-response checks' if number in (3, 5, 6)
                else 'Nine-state source evaluation')
    if prefix == 'LEN' and number == 8:
        return 'Analytic inversion; integrated-growth check'
    if prefix == 'PM':
        return '200/300-step comparison at the stated mesh'
    if prefix == 'GAL' and number in (5, 6):
        return 'Exact reservoir solution and balance check'
    return 'Reference-model numerical calculation'


def generate_register(main_labels, si_labels):
    with (DATA / 'PREDICTION_LEDGER.csv').open(newline='') as handle:
        original = [row for row in csv.DictReader(handle)
                    if not row['id'].startswith('SI')]
    rows = []
    for row in original:
        entry = {key: row[key] for key in
                 ('id', 'quantity_latex', 'domain_and_provenance_latex')}
        entry.update(document='Main article', support=support_for(row['id']))
        rows.append(entry)
    extra = [
        ('SI1', r'$\|P_{\mathcal R}-P\|\leq r\eta_{\mathcal R}$; isolated rank-one eigenvalue persists',
         r'Proposition~\ref{si:prop:persistence}; common physical resolvents; source pole requires positive residue',
         'Contour proof; matrix-sequence controls'),
        ('SI2', r'$|E_{\chi,\mathcal R}-E_\chi|\leq2r^2\eta_{\mathcal R}$; '
                r'$\|Z_{\mathcal R}-Z\|\leq r\eta_{\mathcal R}+2\varepsilon_{\mathcal R}$',
         r'Equations~\eqref{si:eq:energy-bound}--\eqref{si:eq:residue-bound}; positive limiting source weight',
         'Analytic bounds; independent contour integration'),
        ('SI3', r'Connected cumulants determine $\Gamma^{(3)}$ and $\Gamma^{(4)}$',
         r'Proposition~\ref{si:prop:vertices}; locally invertible source-to-field map',
         'Exact cumulants; independent inverse-map derivatives'),
        ('SI4', 'Fixed dynamics admit distinct initial abundances',
         r'Proposition~\ref{si:prop:state}; specified admissible initial densities',
         'Convex density-operator construction'),
        ('SI5', r'$h^2+k\ell_*^2/a^2=(\widehat u+\epsilon_0)/(3\gamma_R)$',
         r'Equation~\eqref{si:eq:copy-friedmann}; matched Einstein branch and cosmological state',
         'Algebraic substitution of gravitational matching'),
        ('SI6', r'$v_c^2=r\Psi^{\prime}$; deflection from $\Phi+\Psi$',
         r'Equation~\eqref{si:eq:common-galaxy-metric}; common weak-field metric and stated symmetries',
         'Timelike and null-geodesic relations'),
        ('SI7', 'On-shell amplitude from the connected four-point function and pole residues',
         r'Equation~\eqref{si:eq:lsz}; stable scalar branch, Lorentzian continuation and asymptotic states',
         'LSZ reduction with the specified field normalization'),
        ('SI8', r'$\mathrm d\sigma=\overline{|\mathcal M|^2}\mathrm d\Phi_f/(4F S_f)$; '
                r'$F=\sqrt{(p_1\cdot p_2)^2-m_1^2m_2^2}$',
         r'Equation~\eqref{si:eq:cross-section}; specified incoming particles and final-state symmetry factor',
         'Relativistic flux and phase-space identity'),
        ('SI9', 'Primordial curvature covariance determines the power-spectrum input to CMB transfers',
         r'Equation~\eqref{si:eq:primordial-state}; homogeneous isotropic initial state and specified transfer evolution',
         'Spectral covariance definition and transfer integrals'),
        ('SI10', r'$u^\dagger Zu\geq u^\dagger Z_{\mathcal R}u-r\eta_{\mathcal R}-2\varepsilon_{\mathcal R}$',
         r'Equation~\eqref{si:eq:visibility-certificate}; unit source vector and verified common-limit error bounds',
         'Analytic enclosure; visible and invisible source controls'),
        ('SI11', r'$\begin{gathered}[\max\{0,\sqrt{z_{\mathcal R}}-d_{\mathcal R}\}]^2\leq z\\'
                 r'z\leq\min\{1,(\sqrt{z_{\mathcal R}}+d_{\mathcal R})^2\}\end{gathered}$',
         r'Equation~\eqref{si:eq:amplitude-residue}; normalized source; '
         r'$z=u^\dagger Zu$, $d_{\mathcal R}=r\eta_{\mathcal R}+\varepsilon_{\mathcal R}$',
         'Reverse-triangle proof; weak-overlap matrix control'),
    ]
    for identifier, quantity, domain, support in extra:
        rows.append(dict(id=identifier, quantity_latex=quantity,
                         domain_and_provenance_latex=domain,
                         document='Supplement', support=support))

    groups = {
        'SP': 'Source spectra and finite operator certificates',
        'MAT': 'Identifiability and independent operator information',
        'GR': 'Finite gravitational response',
        'DE': 'Gravitational matching and expansion',
        'DM': 'Particle interactions and thermal production',
        'LSS': 'Matter growth and structure',
        'GAL': 'Galaxy populations and gas dynamics',
        'PM': 'Particle--mesh evolution',
        'LEN': 'Lensing and scalar metric response',
        'CMB': 'CMB geometry and transfer relations',
        'SI': 'Supplementary physical identities',
    }
    order = {prefix: index for index, prefix in enumerate(groups)}
    def row_key(row):
        match = re.fullmatch(r'([A-Z]+)(\d+)', row['id'])
        return order[match[1]], int(match[2])
    rows.sort(key=row_key)
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate prediction identifiers')

    def resolve(value, labels):
        def replacement(match):
            kind, key = match.groups()
            if key not in labels:
                raise KeyError('Unresolved register reference: ' + key)
            return '(' + labels[key] + ')' if kind == 'eqref' else labels[key]
        return re.sub(r'\\(eqref|ref)\{([^}]+)\}', replacement, value)

    lines = ['% Generated from the main register and verified supplementary identities.',
             r'\newcommand{\predictionrows}{%']
    previous_group = None
    for row in rows:
        prefix = re.match(r'[A-Z]+', row['id']).group()
        if prefix != previous_group:
            lines.append(r'\multicolumn{4}{@{}l@{}}{\textbf{'+groups[prefix]+r'}}\\*')
            previous_group = prefix
        labels = si_labels if row['document'] == 'Supplement' else main_labels
        quantity = resolve(row['quantity_latex'], labels)
        domain = resolve(row['domain_and_provenance_latex'], labels)
        lines.append(f"{row['id']} & {quantity} & {row['document']}: {domain} & {row['support']}\\\\")
    lines.append('}')
    (ROOT / '04_Theoretical_Predictions/rows.tex').write_text('\n'.join(lines) + '\n')
    with (DATA / 'PREDICTION_LEDGER.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'quantity_latex',
                                'domain_and_provenance_latex', 'document', 'support'])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    with tempfile.TemporaryDirectory(prefix='qict-scientific-package-') as td:
        work = Path(td)
        main_pdf, main_labels = compile_document(ROOT / '01_Main_Manuscript',
                                                'manuscript', work / 'main')
        si_pdf, si_labels = compile_document(ROOT / '02_Supplementary_Information',
                                            'supplement', work / 'si')
        count = generate_register(main_labels, si_labels)
        predictions_pdf, _ = compile_document(ROOT / '04_Theoretical_Predictions',
                                              'predictions', work / 'predictions')
        for folder, stem, content in (
                ('01_Main_Manuscript', 'manuscript', main_pdf),
                ('02_Supplementary_Information', 'supplement', si_pdf),
                ('04_Theoretical_Predictions', 'predictions', predictions_pdf)):
            dest = ROOT / folder / (stem + '.pdf')
            staged = dest.with_name(dest.name + '.staged')
            staged.write_bytes(content)
            os.replace(staged, dest)
    print(json.dumps({'documents': 3, 'register_entries': count,
                      'latex_reference_layout_checks': 'PASS',
                      'auxiliary_files_distributed': 0}))


if __name__ == '__main__':
    main()
