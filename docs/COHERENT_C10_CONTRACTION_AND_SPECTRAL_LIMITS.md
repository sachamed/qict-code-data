# Coherent C10 contraction and spectral limits

The C10 dressing already specified in the QICT source is a coherent weighted sum of ordered path holonomies. This note gives its recursive evaluation, an error bound determined by the conserved path flow, and the conditions connecting its source operators to physical spectral observables. The microscopic interaction, state selection, and metric coupling retain their original definitions.

## Coherent evaluation of the prescribed dressing

Let \(\mathcal G=(V,E)\) be a finite directed acyclic graph with a unique sink \(b\). At every other vertex, let \(p_{vw}\geq0\) and \(\sum_w p_{vw}=1\). Let \(U_{vw}\) be bounded contractions on a common Hilbert space; exact link transports are unitary. The source-to-boundary dressing is
\[
D_v=\sum_{\gamma:v\to b}P(\gamma)U_\gamma,\qquad
P(\gamma)=\prod_{e\in\gamma}p_e,
\]
where \(U_\gamma\) is the ordered product of link operators.

**Proposition 1.** The dressing is the unique solution of
\[
D_b=I,\qquad D_v=\sum_{w:v\to w}p_{vw}U_{vw}D_w.
\tag{1}
\]
Moreover, \(\|D_v\|\leq1\). If \(U_{vw}\mapsto g_vU_{vw}g_w^\dagger\), then \(D_v\mapsto g_vD_vg_b^\dagger\).

*Proof.* Partition paths according to their first edge. Each remaining path begins at the edge's target, giving (1). Reverse topological order determines every operator uniquely. Contractivity follows inductively from the convexity of the norm. The gauge factors at adjacent endpoints cancel in each ordered product. \(\square\)

For a vector or a block of six source vectors, set \(y_b=x\) and evaluate \(y_v=\sum_w p_{vw}U_{vw}y_w\) in reverse topological order. This uses one operator application per edge. It avoids path enumeration and does not require a dense matrix for \(D_v\). The Hilbert-space or tensor-bond dimensions still determine the cost of each application.

The adjoint has a forward algorithm. Initialize \(m_v=x\) at the chosen source and zero at other vertices. In topological order, accumulate
\[
m_w\leftarrow m_w+p_{vw}U_{vw}^\dagger m_v.
\]
The final message is \(D_v^\dagger x\).

The coefficients in (1) are the path weights of the prescribed dressing. Replacing the construction by a channel with Kraus operators \(\sqrt{P(\gamma)}U_\gamma\) would remove the cross terms between different paths. For example,
\[
D\rho D^\dagger
=\sum_{\gamma,\gamma'}P(\gamma)P(\gamma')
 U_\gamma\rho U_{\gamma'}^\dagger .
\]
These off-diagonal contributions are part of the coherent source definition.

## Flow-weighted error control

Suppose approximate link operators satisfy
\[
\|\widetilde U_e-U_e\|\leq\epsilon_e,\qquad
\|\widetilde U_e\|\leq1 .
\]
Let \(F_v(e)\) be the probability that a path starting at \(v\) traverses edge \(e\).

**Proposition 2.** With identical path weights and sink data,
\[
\boxed{\ \|\widetilde D_v-D_v\|
\leq\sum_eF_v(e)\epsilon_e\ }. \tag{2}
\]

*Proof.* Write
\[
U_{vw}D_w-\widetilde U_{vw}\widetilde D_w
=(U_{vw}-\widetilde U_{vw})D_w
+\widetilde U_{vw}(D_w-\widetilde D_w).
\]
Contractivity and (1) give \(a_v\leq\sum_w p_{vw}(\epsilon_{vw}+a_w)\), with \(a_b=0\). Expanding this scalar recursion gives the expected accumulated error along a path, which equals the right-hand side of (2). \(\square\)

In the C10 positive-flow decomposition, expected edge occupation equals the conserved electric flow. The same archived flow therefore weights the local approximation errors. If every edge error is at most \(\epsilon_{\rm link}\), then
\[
\|\widetilde D_v-D_v\|
\leq \epsilon_{\rm link}\,\mathbb E_v[|\gamma|].
\]
The mean path length follows from another scalar recursion,
\(\ell_b=0,\ \ell_v=1+\sum_w p_{vw}\ell_w\).
No path count appears in the bound.

For approximations with \(\|\widetilde U_e\|\leq s_e\), use
\(a_v=\sum_w p_{vw}(\epsilon_{vw}+s_{vw}a_w)\).
For weights whose exact row sums have not been established, also propagate the bound \(b_b=1,\ b_v=\sum_w p_{vw}b_w\) and replace the first term by \(\epsilon_{vw}b_w\). Probability errors and floating-point arithmetic require their own bounds. The implementation reports the floating evaluation of these conditional inequalities; it does not label that evaluation a physical certificate.

## Source normalization and moments

For a specified state \(\rho\) and the six dressed chiral source operators \(O_a\), define the Hilbert-Schmidt source map
\[
Qv=\sum_{a=1}^{6}v_aO_a\rho^{1/2},\qquad
G=Q^\dagger Q.
\]
When \(G\succeq\gamma I_6\) with \(\gamma>0\), the normalized embedding is
\[
V=QG^{-1/2},\qquad V^\dagger V=I_6.
\]
Thus the embedding and its normalization follow from the source operators and state. They are derived quantities, rather than independently adjustable physical parameters. A singular Gram matrix requires investigation of the source space; assigning six arbitrary orthonormal columns would change the observable.

For the relative unitary
\[
\mathcal W(X)=U_fXU_0^\dagger ,
\]
the normalized moments \(C_n=V^\dagger\mathcal W^nV\) satisfy \(C_0=I_6\), \(C_{-n}=C_n^\dagger\), and block-Toeplitz positivity. Their identification with the specified two-time physical correlations requires the state condition established separately in the manuscript.

If two constructions use isometries differing by at most \(\delta_V\) and unitaries differing by at most \(\delta_{\mathcal W}\), then
\[
\|\widetilde C_n-C_n\|
\leq2\delta_V+|n|\delta_{\mathcal W}. \tag{3}
\]
This follows by adding and subtracting the intermediate source compressions and telescoping the difference of the unitary powers. Equation (3) connects operator and source error budgets to the spectral reconstruction.

## Stationarity and the thermodynamic limit

For a unitary \(U_0\) and a density operator \(\rho_0\),
\[
\rho_T=T^{-1}\sum_{t=0}^{T-1}U_0^t\rho_0U_0^{-t}
\]
satisfies
\[
\|U_0^n\rho_TU_0^{-n}-\rho_T\|_1
\leq\min(2,2|n|/T).
\tag{4}
\]
Cancellation of overlapping terms proves the estimate. In finite dimension, temporal averaging converges to the spectral pinching of \(\rho_0\) in the eigenspaces of \(U_0\).

For bounded sources, define
\[
\begin{aligned}
K_{T,ab}(n)&=\operatorname{Tr}
 [\rho_TU_0^{-n}O_a^\dagger U_f^nO_b],\\
C_{T,ab}(n)&=\operatorname{Tr}
 [\rho_T^{1/2}O_a^\dagger U_f^nO_b\rho_T^{1/2}U_0^{-n}].
\end{aligned}
\]
If \(G_T\succeq\gamma I\), their Gram-normalized versions obey
\[
\|\widehat K_T(n)-\widehat C_T(n)\|
\leq\frac{c_O^2}{\gamma}\sqrt{\min(2,2|n|/T)},\qquad
c_O=\sup_{\|v\|=1}\left\|\sum_a v_aO_a\right\|.
\tag{5}
\]
Cyclicity of the trace expresses the difference through
\([\rho_T^{1/2},U_0^{-n}]\); the Schatten Cauchy-Schwarz and Powers–Størmer inequalities, followed by (4), prove (5) [1]. Temporal averaging provides stationary ensembles. A physical vacuum selection requires the stated microscopic selection principle.

A gap used for a finite-volume contour, a gap of an auxiliary state-selection operator, and a gap of the physical Hamiltonian are distinct conditions. A uniform positive gap above the physical vacuum would exclude excitations whose energies approach zero. It cannot serve as a universal assumption for a theory intended to contain physical massless modes. Gapped spectral-flow theorems apply under their stated gap hypotheses [2].

An alternative sufficient condition for local state convergence is
\[
\|\omega_{R+1}|_{\mathfrak A_\Lambda}
       -\omega_R|_{\mathfrak A_\Lambda}\|
\leq b_R(\Lambda),\qquad
\sum_R b_R(\Lambda)<\infty
\]
for every fixed local algebra \(\mathfrak A_\Lambda\). Telescoping gives a unique limit along the selected sequence and the error bound
\[
\|\omega_\infty|_{\mathfrak A_\Lambda}
       -\omega_R|_{\mathfrak A_\Lambda}\|
\leq\sum_{s\geq R}b_s(\Lambda).
\]
Compatibility of the finite states supplies a state on the quasilocal algebra. Invariance passes to the limit when the local dynamics converges as well. This criterion requires actual summable bounds; finitely many volume comparisons alone do not establish the tail or independence of the chosen state-selection sequence.

## Spectral interpretation

A persistent source overlap and a vanishing quasimode residual can certify a spectral atom. Specifically, if
\[
\|(W_R-e^{i\theta_R})\psi_R\|\leq r_R,\quad
\|\psi_R\|=1,\quad
\liminf_R|\langle\psi_R,V_Rv\rangle|\geq\alpha>0,
\]
then the spectral projection \(P_R\) onto a chordal window of radius \(\delta_R\) satisfies
\[
\|P_RV_Rv\|^2
\geq\bigl(|\langle\psi_R,V_Rv\rangle|-r_R/\delta_R\bigr)_+^2 .
\]
If \(r_R/\delta_R\to0\), the window shrinks around a common physical energy \(E_*\), and the physical source measures converge weakly, their limit has weight at least \(\alpha^2\) at \(E_*\). The spectral theorem bounds the discarded quasimode component by \(r_R/\delta_R\); the closed-set convergence inequality gives the limiting atom.

This criterion applies to channels whose physical spectral interpretation includes an atom. In charged sectors with a long-range electromagnetic field satisfying the Gauss-law hypotheses, the infraparticle problem requires a threshold formulation [3]. The QICT archive already distinguishes Gauss-dressed charged thresholds from quark short-distance masses. Particle interpretation also requires momentum dependence, a common time calibration, and the appropriate treatment of unstable states.

The nine C10 audit entries group operator, state, source, and validation outputs. Completing their construction requires the coherent microscopic interaction on the declared carrier and the selected state. The path recursion supplies the dressing contraction once the link actions are specified. Metric observables additionally require the microscopic response prescription and its dynamical tensor calculation.

## Implementation and validation

- \`code/coherent_path_dag.py\` implements forward and adjoint applications and the conditional error recursion.
- \`tests/test_coherent_path_dag.py\` checks an independent path enumeration, noncommuting transports, gauge covariance, adjoints, destructive interference, and the error inequality.
- The local block-SVD correction makes the absolute budget mandatory even when a relative tolerance is enabled. With singular values \(1,0.11,0.08\), relative tolerance \(0.2\), and absolute budget \(0.1\), the old condition discarded both smaller values and produced spectral error \(0.11\). The corrected condition retains \(0.11\) and discards \(0.08\).
- The finite tests exercise the mathematics and implementation. Physical C10 outputs retain their existing independent validation requirements.

## References

[1] C. Anantharaman and S. Popa, *An introduction to II1 factors*, draft, Theorems 7.3.7 and 10.2.8. https://www.math.ucla.edu/~popa/Books/IIunV15.pdf

[2] S. Bachmann, S. Michalakis, B. Nachtergaele, and R. Sims, “Automorphic Equivalence within Gapped Phases of Quantum Lattice Systems,” *Communications in Mathematical Physics* **309**, 835–871 (2012). https://arxiv.org/abs/1102.0842

[3] D. Buchholz, “Gauss' law and the infraparticle problem,” *Physics Letters B* **174**, 331–334 (1986). https://doi.org/10.1016/0370-2693(86)91110-X

The prescribed C10 path measure and dressing appear in \`snapshot-2026-10-17/task_evidence/TASK1/code/microscopic_closure/construct_c10_compact_path_dressing.py\`. The full physical execution status is recorded in \`results/physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json\`.
