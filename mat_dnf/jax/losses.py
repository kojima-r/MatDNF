"""Losses and error metrics."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


@partial(jax.jit, static_argnames="split_v_k")
def classification_error(
    i_out: Int[Array, "l"],
    v_k: Float[Array, "l"],
    split_v_k: int = 20,
) -> tuple[Int[Array, ""], Float[Array, ""]]:
    """Compute minimum classification error  Er_k=|I2-(V_k>=V_k_th)|.

    Args:
        i_out: 0-1 row vector representing target truth values corresponding to i_in.
        v_k: Number of satisfied disjuncts.
        split_v_k: Number of threshold bins for disjunction V_k = D_k*M.

    Returns:
        er_k: Minimum error by predicted I2_k(= V_k >= V_k_th).
        v_k_th: Thresholded v_k value
    """
    ls_v_k = jnp.linspace(v_k.min(), v_k.max(), split_v_k, dtype=v_k.dtype)

    d_mat = v_k >= ls_v_k[:, None]  # (split_v_k, l)
    error_v_k = jnp.abs(i_out - d_mat).sum(axis=-1)  # (split_v_k,)

    y = jnp.argmin(error_v_k)
    return error_v_k[y], ls_v_k[y]


@partial(jax.jit, static_argnames=("split_c", "split_d_k"))
def approximation_error(
    c: Float[Array, "h 2*n"],
    d_k: Float[Array, "h"],
    d_i_in: Int[Array, "2*n l"],
    i_out: Int[Array, "l"],
    split_c: int = 10,
    split_d_k: int = 10,
) -> tuple[Int[Array, ""], Bool[Array, "h 2*n"], Bool[Array, "h"]]:
    """Compute minimum approximation error Er_k_th by {D_k_th, C_th}.

    Args:
        c: C-part of learned DNF (continuous space).
        d_k: D-part of learned DNF (continuous space).
        d_i_in: Dualized matrix of l data points in n variables to be classified.
        i_out: 0-1 row vector representing target truth values corresponding to i_in.
        split_c: Number of threshold bins for h conjunctions C.
        split_d_k: Number of threshold bins for disjunction D_k.

    Returns:
        er_k_th: The minimum approximation error.
        c_th: (h x 2n) 0-1 Mat for conjunction.
        d_k_th: (1 x h) 0-1 Mat for disjunction.
    """
    ls_c = jnp.linspace(c.min(), c.max(), split_c, dtype=c.dtype)
    ls_d_k = jnp.linspace(d_k.min(), d_k.max(), split_d_k, dtype=d_k.dtype)

    d_mat = d_k >= ls_d_k[:, None]  # (split_d_k, h)
    c_mat = c >= ls_c[:, None, None]  # (split_c, h, 2n)
    b_mat = (c_mat @ d_i_in) == c_mat.sum(axis=-1, keepdims=True)  # (split_c, h, l)
    e_mat = d_mat @ b_mat  # (split_c, split_d_k, l)
    error_cd = jnp.abs(i_out - e_mat).sum(axis=-1)

    # Get thresholds from error_cd
    er_k_th_index = jnp.argmin(error_cd)
    yw, w = jnp.unravel_index(er_k_th_index, error_cd.shape)
    er_k_th = error_cd[yw, w]
    c_th = (c >= ls_c[yw]).astype(er_k_th.dtype)
    d_k_th = d_k >= ls_d_k[w]

    return er_k_th, c_th, d_k_th


def logi_conseq(
    dnf: Int[Array, "h 2*n"],
    i_out: Int[Array, "l"],
    i_in: Int[Array, "n l"],
) -> tuple[bool, Int[Array, ""] | None]:
    """Logical consequence.

    I_in(n x l): l=2^n complete assignments over n variables
    I_out(1 x l): truth values of the original DNF against I_in
    => (I_in I_out) = complete spec. of Boolean func., DNF

    If i_in_true |= dnf, conseq = 1.
    O.w. conseq = 0 and couner_example = I_in_counter
    """
    i_in_true = i_in[:, i_out.astype(jnp.bool_)]
    # For when DNF is empty
    try:
        m = 1 - (dnf @ jnp.vstack([1 - i_in_true, i_in_true])).min(axis=0)
    except ValueError:
        return False, None

    # D = np.ones(1, learned_dnf_s.shape[0])
    # i_out_true = D @ M >= 1
    # True iff i_in_true |= learned_DNF
    if not (i_out_true := m >= 1).all():
        return False, i_in_true[:, ~i_out_true]
    return True, None


def logi_equiv(
    dnf: Int[Array, "h 2*n"],
    i_out: Int[Array, "l"],
    i_in: Int[Array, "n l"],
) -> tuple[bool, Int[Array, ""] | None]:
    """Logical equivalence."""
    # TODO: Duplicate with `acc_dnf`
    xx = dnf @ jnp.vstack([i_in, 1 - i_in]) == dnf.sum(axis=1, keepdims=True)
    yy = jnp.sum(xx, axis=0) >= 1

    # True iff (I_in |= learned_DNF) = I_out
    if not (i_out == yy).all():
        return False, i_in[:, i_out != yy]
    return True, None

def pred_classi(
    d_k: Float[Array, "h"],
    i1: Int[Array, "n l"],
    l2: int,
    c: Float[Array, "h 2*n"],
) -> Float[Array, ""]:
    """Some kind of metric for the learned I2."""
    xV_k = d_k @ (1 - jnp.minimum(c @ jnp.vstack([1 - i1, i1]), 1))
    return xV_k
    
def acc_classi(
    d_k: Float[Array, "h"],
    v_k_th: Float[Array, ""],
    i1: Int[Array, "n l"],
    i2_k: Int[Array, "l"],
    l2: int,
    c: Float[Array, "h 2*n"],
) -> Float[Array, ""]:
    """Some kind of metric for the learned I2."""
    xV_k = pred_classi(d_k, i1, l2, c)
    i2_k_learned = (xV_k >= v_k_th).astype(i2_k.dtype)
    return 1.0 - jnp.abs(i2_k - i2_k_learned).sum() / l2


def pred_dnf(
    dnf: Int[Array, "h 2*n"],
    i1: Int[Array, "n l"],
    l2: int,
) -> Float[Array, ""]:
    """Some kind of metric for the learned DNF."""
    # TODO: This block pattern is repeated many times -> utils.py
    xx = (dnf @ jnp.vstack([i1, 1 - i1])) == dnf.sum(axis=1)[:, None]
    I2_k_learned_b = xx.sum(axis=0) >= 1
    return I2_k_learned_b

def acc_dnf(
    dnf: Int[Array, "h 2*n"],
    i1: Int[Array, "n l"],
    i2_k: Int[Array, "l"],
    l2: int,
) -> Float[Array, ""]:
    """Some kind of metric for the learned DNF."""
    # TODO: This block pattern is repeated many times -> utils.py
    I2_k_learned_b = pred_dnf(dnf, i1, l2)
    zz = i2_k - I2_k_learned_b
    return 1.0 - jnp.abs(zz).sum() / l2
