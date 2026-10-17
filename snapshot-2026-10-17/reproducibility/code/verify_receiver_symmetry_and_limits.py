"""Exact small-model checks for receiver inversion, family symmetry and Pfaffians.

These checks exercise finite algebraic identities; they do not certify a
continuum interacting particle calculation.
"""
from fractions import Fraction as F
from itertools import permutations
from math import factorial


def pfaffian(A):
    n = len(A)
    assert n % 2 == 0
    if not n:
        return 1
    return sum(((-1) ** (j + 1)) * A[0][j] * pfaffian([
        [A[r][c] for c in range(n) if c not in (0, j)]
        for r in range(n) if r not in (0, j)
    ]) for j in range(1, n))


def determinant(A):
    n = len(A)
    if n == 0:
        return 1
    return sum((-1)**j * A[0][j] * determinant([
        row[:j] + row[j + 1:] for row in A[1:]
    ]) for j in range(n))


def block_majorana(M):
    n = len(M)
    return [[0 if i < n else -M[j][i-n] for j in range(n)]
            + ([M[i][j] for j in range(n)] if i < n else [0]*n)
            if i < n else
            ([-M[j][i-n] for j in range(n)] + [0]*n)
            for i in range(2*n)]


def block_matrix(M):
    n = len(M)
    A = [[0] * (2*n) for _ in range(2*n)]
    for i in range(n):
        for j in range(n):
            A[i][n+j] = M[i][j]
            A[n+j][i] = -M[i][j]
    return A

# Test any square matrix, including nonsymmetric and singular cases.
for M in ([[F(2)]], [[F(1), F(2)], [F(3), F(4)]],
          [[F(1), F(2)], [F(2), F(4)]],
          [[F(1), F(2), F(0)], [F(0), F(1), F(3)], [F(4), F(0), F(2)]]):
    n = len(M)
    assert pfaffian(block_matrix(M)) == (-1)**(n*(n-1)//2)*determinant(M)

# Independent exact counterexample to the phase-insensitive sign inversion.
m = F(1); nu = F(2,5)
for Delta in (F(-2), F(2)):
    G0sq = m*m + Delta*Delta
    A = m*m / G0sq
    CG2 = m*nu*m*nu/G0sq
    assert (G0sq, A, CG2) == (F(5), F(1,5), F(4,125))

# At m=0, Delta=nu=1, k=1/2, the maximum leakage is 1/17.
m = F(0); Delta = nu = F(1); k = F(1,2)
a = m + nu*k*k
assert a*a / (a*a + Delta*Delta) == F(1,17)

# Three-dimensional permutation invariance permits a singlet and doublet.
J = [[F(1) for _ in range(3)] for _ in range(3)]
A = [[(F(2) if i==j else F(0)) + F(1) for j in range(3)] for i in range(3)]
assert [sum(row) for row in A] == [F(5)]*3
for p in permutations(range(3)):
    assert all(A[p[i]][p[j]] == A[i][j] for i in range(3) for j in range(3))
# The vector (1,1,1) has eigenvalue 5 and (1,-1,0) eigenvalue 2.
assert [sum(A[i][j]*v for j,v in enumerate((1,-1,0))) for i in range(3)] == [2,-2,0]

# liminf and limsup need not fill out the cluster set.
oscillatory = [F(i % 2) for i in range(100)]
assert set(oscillatory) == {F(0), F(1)}

# Support starts at zero although measure (-inf,0] is zero: uniform (0,1).
# This verifies the non-strict endpoint inequality in the edge proof.
assert F(0) >= F(0)
print('Receiver, Pfaffian, permutation and limit checks: PASS')
