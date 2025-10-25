"""Losses and error metrics."""

import cupy as cp
from numpy import bool_, floating, integer
from numpy.typing import NBitBase, NDArray


# TODO: Update descriptions for v_k, i2_k, and v_k_th.
def classification_error[T: NBitBase, U: NBitBase](
    i_out: NDArray[integer[T]],
    v_k: NDArray[floating[U]],
    split_v_k: int = 20,
) -> tuple[integer[T], floating[U]]:
    """Compute minimum classification error  Er_k=|I2-(V_k>=V_k_th)|.

    Args:
        i_out: 0-1 row vector representing target truth values corresponding to i_in.
        v_k: Number of satisfied disjuncts.
        split_v_k: Number of threshold bins for disjunction V_k = D_k*M.

    Returns:
        er_k: Minimum error by predicted I2_k(= V_k >= V_k_th).
        v_k_th: Thresholded v_k value
    """
    xp = cp.get_array_module(i_out, v_k)
    ls_v_k = xp.linspace(v_k.min(), v_k.max(), split_v_k, dtype=v_k.dtype)

    d_mat = v_k >= ls_v_k[:, None]  # (split_v_k, l)
    error_v_k = xp.abs(i_out - d_mat).sum(axis=-1)  # (split_v_k,)

    y = xp.argmin(error_v_k)
    return error_v_k[y], ls_v_k[y]


def approximation_error[T: NBitBase, U: NBitBase](
    c: NDArray[floating[T]],
    d_k: NDArray[floating[T]],
    d_i_in: NDArray[integer[U]],
    i_out: NDArray[integer[U]],
    split_c: int = 10,
    split_d_k: int = 10,
) -> tuple[integer[U], NDArray[integer[U]], NDArray[bool_]]:
    """Compute minimum approximation error Er_k_th by {D_k_th, C_th}.
    
    Args:
        c: C-part of learned DNF (continuous space).   # (h, 2n) where n=#variables and h is the maximum number of conjunctions in a DNF.
        d_k: D-part of learned DNF (continuous space). # (h,) 
        d_i_in: Dualized matrix of l data points in n variables to be classified.
        i_out: 0-1 row vector representing target truth values corresponding to i_in.
        split_c: Number of threshold bins for h conjunctions C.
        split_d_k: Number of threshold bins for disjunction D_k.

    Returns:
        er_k_th: The minimum approximation error.
        c_th: (h x 2n) 0-1 Mat for conjunction.
        d_k_th: (1 x h) 0-1 Mat for disjunction.
    """
    xp = cp.get_array_module(c, d_k, d_i_in, i_out)
    ls_c = xp.linspace(c.min(), c.max(), split_c, dtype=c.dtype)
    ls_d_k = xp.linspace(d_k.min(), d_k.max(), split_d_k, dtype=d_k.dtype)

    d_mat = d_k >= ls_d_k[:, None]  # (split_d_k, h)
    c_mat = c >= ls_c[:, None, None]  # (split_c, h, 2n)
    b_mat = (c_mat @ d_i_in) == c_mat.sum(axis=-1, keepdims=True)  # (split_c, h, l)
    e_mat = d_mat @ b_mat  # (split_c, split_d_k, l)
    error_cd = xp.abs(i_out - e_mat).sum(axis=-1)

    # Get thresholds from error_cd
    er_k_th_index = xp.argmin(error_cd)
    yw, w = xp.unravel_index(er_k_th_index, error_cd.shape)
    er_k_th = error_cd[yw, w]
    c_th = (c >= ls_c[yw]).astype(er_k_th.dtype)
    d_k_th = d_k >= ls_d_k[w]

    return er_k_th, c_th, d_k_th


def logi_conseq[T: NBitBase](
    dnf: NDArray[integer[T]],
    i_out: NDArray[integer[T]],
    i_in: NDArray[integer[T]],
) -> tuple[bool, NDArray[integer[T]] | None]:
    """Logical consequence. |= phi_0 => phi
    s.t.
      phi:= dnf
      phi_0:= (i_in, i_out)
    
    If i_in_true |= dnf, conseq = 1.
    O.w. conseq = 0 and couner_example = I_in_counter
    
    Args:
        dnf: #clause x (2x#variables)
        i_out: 0-1 row vector (L,) representing target truth values corresponding to i_in with L: interpretation.  
        i_in: 0-1 matrix (N,L) with N: #variables

    Returns:
        bool: logical equivalence
        profile: #variable x non-equivalent interpretations
        
    """
    xp = cp.get_array_module(dnf, i_out, i_in)
    i_in_true = i_in[:, i_out.astype(bool_)]
    # For when DNF is empty
    try:
        m = 1 - (dnf @ xp.vstack([1 - i_in_true, i_in_true])).min(axis=0)
    except ValueError:
        return False, None

    # D = np.ones(1, learned_dnf_s.shape[0])
    # i_out_true = D @ M >= 1
    # True iff i_in_true |= learned_DNF
    if not (i_out_true := m >= 1).all():
        return False, i_in_true[:, ~i_out_true]
    return True, None


def logi_equiv[T: NBitBase](
    dnf: NDArray[integer[T]],
    i_out: NDArray[integer[T]],
    i_in: NDArray[integer[T]],  # 
) -> tuple[bool, NDArray[integer[T]] | None]:
    """Logical equivalence. |= phi_0 <=> phi
    s.t.
      phi:= dnf
      phi_0:= (i_in, i_out)
      
    Args:
        dnf: #clause x (2x#variables)
        i_out: 0-1 row vector (L,) representing target truth values corresponding to i_in with L: interpretation.  
        i_in: 0-1 matrix (N,L) with N: #variables

    Returns:
        bool: logical equivalence
        profile: #variable x non-equivalent interpretations
    """
    xp = cp.get_array_module(dnf, i_out, i_in)
    # TODO: Duplicate with `acc_dnf`
    xx = dnf @ xp.vstack([i_in, 1 - i_in]) == dnf.sum(axis=1, keepdims=True)
    yy = xp.sum(xx, axis=0) >= 1

    # True iff (I_in |= learned_DNF) = I_out
    if not (i_out == yy).all():
        return False, i_in[:, i_out != yy]
    return True, None


def pred_classi[T: NBitBase, U: NBitBase](
    d_k: NDArray[floating[T]],
    v_k_th: NDArray[floating[T]],
    i1: NDArray[integer[U]],
    l2: int,
    c: NDArray[floating[T]],
) -> floating[T]:
    """prediction by soft DNF for the learned DNF.
    where y_pred is computed by continual weighted clauses
    
    Args:
        d_k: D-part of learned DNF (continuous space). # (h,) 
        v_k_th: threshold scalar value
        i1: i_in inputs
        l2: =L(#interpretation)
        c: C-part of learned DNF (continuous space).   # (h, 2n) where n=#variables and h is the maximum number of conjunctions in a DNF.
    """
    xp = cp.get_array_module(d_k, v_k_th, i2_k, c)
    xV_k = d_k @ (1 - xp.minimum(c @ xp.vstack([1 - i1, i1]), 1))
    i2_k_learned = (xV_k >= v_k_th).astype(i2_k.dtype)
    return i2_k_learned

def acc_classi[T: NBitBase, U: NBitBase](
    d_k: NDArray[floating[T]],
    v_k_th: NDArray[floating[T]],
    i1: NDArray[integer[U]],
    i2_k: NDArray[integer[U]],
    l2: int,
    c: NDArray[floating[T]],
) -> floating[T]:
    """accuracy metric predicted by soft DNF for the learned DNF.
    1/L \sum 1_{y_pred=i2_k} 
    where y_pred is computed by continual weighted clauses
    
    Args:
        d_k: D-part of learned DNF (continuous space). # (h,) 
        v_k_th: threshold scalar value
        i1: i_in inputs
        i2_k: i_out outputs
        l2: =L(#interpretation)
        c: C-part of learned DNF (continuous space).   # (h, 2n) where n=#variables and h is the maximum number of conjunctions in a DNF.
    """
    
    i2_k_learned = pred_classi(d_k, v_k_th, i1, l2, c)
    return 1.0 - xp.abs(i2_k - i2_k_learned).sum() / l2

def pred_dnf(
    dnf: NDArray[integer],
    i1: NDArray[integer],
    l2: int,
) -> floating:
    """prediction by the learned DNF
    where y_pred is computed by discretized clauses
    
    Args:
        dnf: #clause x (2x#variables)
        i1: (i_in) 0-1 matrix (N,L) with N: #variables
        l2: =L(#interpretation)
    """
    xp = cp.get_array_module(dnf, i1, i2_k)
    # TODO: This block pattern is repeated many times -> utils.py
    xx = (dnf @ xp.vstack([i1, 1 - i1])) == dnf.sum(axis=1)[:, None]
    I2_k_learned_b = xx.sum(axis=0) >= 1
    return I2_k_learned_b
    
def acc_dnf(
    dnf: NDArray[integer],
    i1: NDArray[integer],
    i2_k: NDArray[integer],
    l2: int,
) -> floating:
    """accuracy metric predicted by the learned DNF.
    1/L \sum 1_{y_pred=i2_k} 
    where y_pred is computed by discretized clauses
    
    Args:
        dnf: #clause x (2x#variables)
        i1: (i_in) 0-1 matrix (N,L) with N: #variables
        i2_k: (i_out) 0-1 row vector (L,) representing target truth values corresponding to i_in with L: #interpretation.  
        l2: =L(#interpretation)
    """
    I2_k_learned_b = pred_dnf(dnf, i1, l2)
    zz = i2_k - I2_k_learned_b
    return 1.0 - xp.abs(zz).sum() / l2
