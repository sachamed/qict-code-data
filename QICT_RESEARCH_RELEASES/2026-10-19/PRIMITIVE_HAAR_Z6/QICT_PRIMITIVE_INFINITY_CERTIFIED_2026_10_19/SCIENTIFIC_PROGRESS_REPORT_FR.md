# Quantum Information Copy Time: convergence primitive et compatibilité du quotient central

## Convergence de l'état fondamental à une plaquette

Pour l'Hamiltonien non déroulé

\[
H=8r^2+4-2(S+S^\dagger),\qquad r\in\mathbb Z,
\]

la partie cinétique de voisinage est positive et la contribution diagonale est confinante. Le spectre est discret, le fondamental est simple et strictement positif, et le principe min–max établit \(E_1\ge8\) et \(E_0\le4\). Il en résulte un écart spectral d'au moins 4.

Les quatre vecteurs primitifs authentifiés aux coupures de flux 8, 12, 16 et 20 ont été intégrés à l'espace commun \(\ell^2(\mathbb Z)\) par leurs indices entiers. Les sommes de leur quotient de Rayleigh, leur second moment et leur résidu dans l'opérateur **infini** ont été évaluées exactement comme nombres rationnels dyadiques. Les termes de bord provenant des flux extérieurs ont été inclus. Les inégalités de Temple et du théorème spectral fournissent les encadrements certifiés suivants.

| Coupure | Borne supérieure de la distance de trace au fondamental infini | 
|---:|---:|
| 8 | \(4.425\times10^{-15}\) |
| 12 | \(8.9\times10^{-17}\) |
| 16 | \(7.5\times10^{-17}\) |
| 20 | \(2.2\times10^{-17}\) |

Les intervalles rationnels d'énergie encadrent une même valeur proche de \(3.0897227917851729035347336249422\). Le calcul des fidélités entre les quatre vecteurs, effectué séparément en arithmétique rationnelle, confirme leur compatibilité dans une représentation commune. La convergence de la suite Galerkin complète résulte en outre de la densité des vecteurs à support fini pour la forme quadratique et de l'écart spectral uniforme.

Cette démonstration concerne un Hamiltonien primitif abélien à une plaquette. Elle ne construit pas les quatre états Kato spatiaux du transfert Floquet chargé à \(\lambda=1\).

## Représentations admissibles sur le quotient \(\mathbb Z_6\)

Une représentation de charges centrales \((Y,2j,t)\), avec \(Y\) entier, \(2j\) entier non négatif et \(t\) la trialité \(SU(3)\), descend au quotient fidèle si et seulement si

\[
Y+3(2j)+2t\equiv 0\pmod 6.
\]

Ce résultat résout le problème de sélection des caractères centraux. Le produit mixte \(e^{i\theta}\operatorname{Tr}(u)\operatorname{Tr}(v)/6\) est invariant sous le générateur du noyau. Les caractères fondamentaux employés séparément dans le potentiel magnétique gelé ne le sont pas. Une formulation de l'action magnétique sur le quotient peut donc être définie à partir de caractères admissibles, mais elle représenterait une action différente ; les équations et l'empreinte de l'action gelée ne seraient plus préservées.

## Dépendances physiques encore ouvertes

L'action canonique ne détermine pas encore un opérateur global \(U_{\mathrm{int}}^{rf}\) exécutable avec toutes les holonomies, cartes C10 et contraintes Gauss/CAR. En l'absence de ces amplitudes, les quatre états Kato physiques, l'embedding conjoint des six sources chirales et les moments CN6 chargés ne peuvent pas être calculés sans introduire une dynamique nouvelle. La localisation et la persistance d'un pôle du secteur impair nécessitent en outre une projection de parité compatible avec cet opérateur et des estimations uniformes de résidu et d'écart spectral. Enfin, la conversion en énergie physique exige une mesure indépendante de \(\tau_*\) et le choix justifié de la branche de quasi-énergie.

Les résultats mathématiques ci-dessus sont intégralement calculés et vérifiables dans leur domaine de validité. Aucune masse physique ni exécution HPC complète n'est déduite de ces certificats primitifs.

## Pôle primitif local en flux infini

L'état fondamental du Hamiltonien primitif est séparé du reste du spectre par un écart d'au moins 4. La fonction de Green locale \(G_{00}(z)=\langle0|(z-H)^{-1}|0\rangle\) possède donc un pôle simple en \(E_0\). Son résidu est \(Z_0=|\langle0|\psi_0\rangle|^2\). L'inégalité \(H\ge8r^2\) et la borne \(E_0\le4\) donnent directement \(Z_0\ge1/2\). L'évaluation rationnelle des facteurs de Gram de la coupure \(N=20\) resserre cette borne :

\[
0.9058301980893397686289319480855\le Z_0\le
0.9058301980893398126289319480856.
\]

Le pôle a été recoupé par trois approches distinctes : encadrement rationnel de Temple et du résidu spectral, diagonalisation tridiagonale du secteur pair, et réduction Schur sous forme de fraction continue. Dans ces deux dernières approches, le contrôle flottant sert de vérification algorithmique et non de substitut à l'encadrement rationnel. Un défaut de précision du solveur par bissection d'une valeur propre sélectionnée, atteignant \(10^{-12}\) au plus grand cutoff testé, a été corrigé par un solveur tridiagonal intégral.

## Non-stationnarité sous la mise à jour de Floquet ordonnée

La dynamique physique discrète utilise des portes ordonnées. Dans le sous-système abélien élémentaire, \(U_{BE}=e^{-i(\pi/63)K_B}e^{-i(\pi/63)K_E}\) n'est pas l'exponentielle du Hamiltonien somme. L'évaluation par le noyau de Bessel exact et par exponentiation sur grille élargie fournit un défaut d'état propre optimisé sur la phase d'environ \(0.03211158\), alors que l'erreur de l'état primitif relativement à son état fondamental infini est inférieure à \(10^{-14}\). Les deux méthodes d'évolution concordent à la précision flottante.

Ce résultat démontre la nécessité d'une construction autonome de l'état Floquet Kato : on ne peut pas transférer le pôle du Hamiltonien primitif au transfert discret ordonné. La porte \(U_{\mathrm{int}}^{rf}\), les cartes C10 et le secteur impair interagissant restent à déterminer pour conclure à une particule physique QICT.

## Obstruction Floquet exacte dans la plaquette abélienne

La différence entre les deux réalisations est aussi démontrable algébriquement. La translation de flux \(T_{126}:|r\rangle\mapsto|r+126\rangle\) commute avec \(K_B\) et avec la porte \(e^{-i\pi K_E/63}\), car

\[
K_E(r+126)-K_E(r)=126(16r+1008).
\]

La translation bilatérale \(T_{126}\) n'a aucun vecteur propre non nul dans \(\ell^2(\mathbb Z)\). Par conséquent, l'évolution Floquet ordonnée \(U_{BE}\) ne peut admettre aucun projecteur spectral de Riesz non nul de rang fini. Le Hamiltonien primitif, lui, ne commute pas avec cette translation et conserve un état fondamental de rang un isolé. Cette séparation établit un résultat exact dans le secteur abélien. Son extension à la dynamique QICT avec matière demande encore la preuve de la commutation de \(U_{\mathrm{int}}^{rf}\) avec cette translation, ou une autre prescription cohérente de sélection d'état.
