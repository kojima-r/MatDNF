"""DNF simplifications."""

from typing import cast

import cupy as cp
from numpy import bool_, integer
from numpy.typing import NBitBase, NDArray


def simp_dnf[T: NBitBase](dnf: NDArray[integer[T]]) -> NDArray[integer[T]]:
    """DNF simplification.

    Logical case:

    - ()v(a1 & a2)                           => true                  empty elimi.
    - (..a3 & ~a3..)v(a1 & a2)               => (a1 & a2)             anti-tautology elimi.
    - a1 v (~a1 & a2 & a4) v ...             => a1 v (a2 & a4) v ...  unit propa.
    - (a1 & a2 & a4) v (~a1 & a2 & a4) v ... => (a2 & a4) v ...       resolution
    - (a1 & a2 & a4) v (a2 & a4) v ...       => (a2 & a4) v ...       subsumption

    Continuous case:

    - V2_k = sum(1 - min_1(DNF*[1-V1;V1]))   : anti-tautology elimi. applicable
    - I2_k_learned_DNF = (V2_k>=1)

    Args:
        dnf: DNF to be simplified.

    Returns:
        Simplified DNF.
    """
    if len(dnf) == 1:  # single conjunction
        return dnf

    dnf = remove_a_not_a(dnf)
    dnf = _unit_propagation(dnf)
    dnf = _resolution(dnf)
    dnf = _remove_empty_rows(dnf)
    dnf = _subsumption(dnf)
    return dnf


def remove_a_not_a[T: NBitBase](dnf: NDArray[integer[T]]) -> NDArray[integer[T]]:
    """Remove ..A&~A.. conjunction.

    Args:
        dnf: DNF to be simplified.

    Returns:
        Simplified DNF.
    """
    n = dnf.shape[1] // 2
    bb = dnf[:, :n] + dnf[:, n:]
    no_A_notA = ~((bb == 2).any(axis=1))  # noqa: PLR2004
    return dnf[no_A_notA]


def _unit_propagation[T: NBitBase](dnf: NDArray[integer[T]]) -> NDArray[integer[T]]:
    """DNF simplification by unit propagation.

    Args:
        dnf: DNF to be simplified.

    Returns:
        Simplified DNF.
    """
    xp = cp.get_array_module(dnf)
    n = xp.floor_divide(dnf.shape[1], 2)
    # 1: positive literal, -1: negative literal
    aa = dnf[:, :n] - dnf[:, n:]

    # unit propagation
    while True:
        # TODO: Vectorizable?
        for b_k in aa:  # k-th monomial
            if xp.abs(b_k).sum() != 1:
                continue
            b1 = b_k != 0  # unit clause b_k's {1,-1}-position (var = 1 or -1)
            # c1(i,:) indicates if aa(i,:)'s {1,-1}-positons = -(b_k's {1,-1}-position)
            c1 = aa[:, b1].ravel() == -b_k[b1]
            if c1.sum() > 0:  # column in aa that can be resolved.
                aa[c1, b1] = 0  # resolve |d1| monomials with ~b_k
                break
        else:
            break
    return xp.hstack([aa == 1, aa == -1]).astype(dnf.dtype)


def _resolution[T: NBitBase](dnf: NDArray[integer[T]]) -> NDArray[integer[T]]:
    """DNF simplification by resolution.

    Args:
        dnf: DNF to be simplified.

    Returns:
        Simplified DNF.
    """
    xp = cp.get_array_module(dnf)
    n = xp.floor_divide(dnf.shape[1], 2)
    # 1: positive literal, -1: negative literal
    aa = dnf[:, :n] - dnf[:, n:]

    # TODO: Can be vectorized?
    while True:
        for k in range(len(aa)):
            bb = aa[k]  # monomial
            b0 = bb == 0  # bb's 0-positions
            b1 = bb != 0  # bb's {1, -1}-positions

            # c0[i, :] = [1..1] => aa[i, :]'s 0-positions = bb's 0-positions
            c0 = aa[:, b0] == bb[b0]
            # c1[i, :] indicates if aa[i, :]'s {1,-1}-positons = bb's {1,-1}-positions
            c1 = aa[:, b1] == bb[b1]
            # c2[i, :] indicates bb's {1,-1}-positions by {0,1}
            c2 = xp.abs(aa[:, b1])  # TODO: Isn't this redundant?
            # bb's complementary monomial differs from bb just by 1 lieteral
            nb1 = b1.sum() - 1

            # (all(c0')' & all(c2')')(i)=1 <=> bb's {1,0,-1}-positions = i-th row's {1,0,-1}-positions
            # TODO: Why only delete one row at a time...
            d0 = c0.all(axis=1) & c2.all(axis=1) & (c1.sum(axis=1) == nb1)
            (d1,) = xp.where(d0)
            if len(d1) > 0:
                del_r = d1[0]
                del_c = (xp.abs(aa[k] - aa[del_r])).astype(xp.bool_)
                # k-th row and del_r-th row are resolved upon the variable del_c
                aa[k, del_c] = 0
                # delete r-th row from aa and keep k-th row
                aa = xp.delete(aa, del_r, 0)
                break
        else:
            break
    return xp.hstack([aa == 1, aa == -1]).astype(dnf.dtype)


def _remove_empty_rows[T: NBitBase](dnf: NDArray[integer[T]]) -> NDArray[integer[T]]:
    """Remove [0..0] row as it behaves as a false disjunct in a DNF.

    Args:
        dnf: DNF to be simplified.

    Returns:
        Simplified DNF.
    """
    xp = cp.get_array_module(dnf)
    return xp.unique(dnf[dnf.sum(axis=1) != 0], axis=0)


def _subsumption[T: NBitBase](dnf: NDArray[integer[T]]) -> NDArray[integer[T]]:
    """DNF simplification by subsumption.

    Args:
        dnf: DNF to be simplified.

    Returns:
        Simplified DNF.
    """
    xp = cp.get_array_module(dnf)
    if len(dnf) == 1:  # single conjunction
        return dnf

    # remove [0..0] row as it behaves as a false disjunct in a DNF
    dnf = xp.unique(dnf[dnf.sum(axis=1) != 0], axis=0)
    _dnf = dnf.copy()

    # subsumption <- very inefficient
    while True:
        for k in range(len(_dnf)):
            b_k = _dnf[k]
            # indicates rows of dnf subsumed by b_k
            zz = cast(NDArray[bool_], xp.all((_dnf - b_k) >= 0, axis=1))
            # zz[k] = 0  # TODO: More efficient if here.
            if zz.sum() > 1:  # b_k subsumes other conj.s  Av(A&B) <=> A
                zz[k] = 0
                (xx,) = xp.where(zz)
                # remove subsumed rows
                _dnf = xp.delete(_dnf, xx, axis=0)
                break
        else:
            break
    return xp.unique(_dnf, axis=0)
