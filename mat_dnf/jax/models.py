"""Definitions and training loops for models."""

import math
from functools import partial, reduce
from typing import Literal, Self

import equinox as eqx
import optax
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from ..utils import should_break
from .losses import approximation_error, classification_error
from .optimizers import adam, normalize_leaf


class MatDNF(eqx.Module):
    """MatDNF neural network."""

    c: Float[Array, "h 2*n"]
    d_k: Float[Array, "h"]
    aa: int = eqx.field(static=True)

    def __init__(self, c: Float[Array, "h 2*n"], d_k: Float[Array, "h"], aa: int = 4):
        """Initialize MatDNF NN with initial C and D_k arrays."""
        self.c = c
        self.d_k = d_k
        self.aa = aa

    @classmethod
    def create_random(
        cls,
        h: int,
        n: int,
        aa: int = 4,
        *,
        key: PRNGKeyArray,
    ):
        """Create a MatDNF NN with random C and D_k of the specified shape.

        Args:
            h: Maximum number of conjunctions in a DNF.
            n: Number of variables.
            aa: Scaling factor for the random initialization.
            seed: Random number seed.
            key: A JAX PRNG key.
        """
        c_key, d_k_key = random.split(key)
        c = jnp.sqrt(aa / (h * 2 * n)) * random.normal(c_key, (h, 2 * n)) + 0.5
        d_k = jnp.sqrt(aa / h) * random.normal(d_k_key, (h,)) + 0.5
        return cls(c=c, d_k=d_k, aa=aa)

    def __call__(self, i_in_d: Int[Array, "2*n l"]) -> Float[Array, "l"]:
        """Compute v_k."""
        # Single hidden layer
        N = self.c @ i_in_d  # (h, l), O(h*2n*l)
        M = 1 - jnp.minimum(N, 1)  # (h, l); O(h*l)  # ReLU
        # Linear output layer
        return self.d_k @ M  # O(l,), U(h*l) Analogue disjunction output

    def perturbate(self, key: PRNGKeyArray) -> Self:
        """Equation (5)."""
        c0_key, d0_k_key = random.split(key)

        C0 = (  # (h, 2*n)
            jnp.sqrt(self.aa / reduce(lambda x, y: x * y, self.c.shape))
            * random.uniform(c0_key, self.c.shape)
            + 0.5
        )
        c = 0.5 * (self.c + C0)

        D0_k = (  # (h,)
            jnp.sqrt(self.aa / reduce(lambda x, y: x * y, self.d_k.shape))
            * random.uniform(d0_k_key, self.d_k.shape)
            + 0.5
        )
        d_k = 0.5 * (self.d_k + D0_k)

        return type(self)(c=c, d_k=d_k, aa=self.aa)


@eqx.filter_jit  # JIT-compile this function
@partial(eqx.filter_value_and_grad, has_aux=True)  # Autograd + (value, f, r)
def j_loss(
    model: MatDNF, i_in_d: Int[Array, "2*n l"], i_out: Int[Array, "l"], l2: float = 0.1
):
    """The J cost function (Equation 3)."""
    v_k = model(i_in_d)

    Y = model.c * (1 - model.c)  # (h x l), O(h*2n*l)
    Z = model.d_k * (1 - model.d_k)

    W = v_k  # V_k = D_k*(1-min(C*[1-I1;I1],1))
    # NOTE: Column-wise dot product a'la MATLAB
    f = (
        jnp.vecdot(i_out, 1 - jnp.minimum(W, 1)).sum()
        + jnp.vecdot(1 - i_out, jnp.maximum(W, 0)).sum()
    )
    r = l2 * 0.5 * (jnp.vecdot(Y, Y, axis=0).sum() + jnp.vecdot(Z, Z, axis=0).sum())

    return f + r, (f, r, v_k)  # O(h*2n*l)


@eqx.filter_jit
def train_step(
    model: MatDNF,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    i_in_d: PyTree,
    i_out: PyTree,
    l2: float,
):
    """Train for a single step."""
    ((loss_value, (f, r, v_k)), grads) = j_loss(
        model, i_in_d=i_in_d, i_out=i_out, l2=l2
    )
    updates, opt_state = optim.update(grads, opt_state, model)  # type: ignore
    model = eqx.apply_updates(model, updates)
    return loss_value, f, r, v_k, model, opt_state


@eqx.filter_jit
def train_step_sam(
    model: MatDNF,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    i_in_d: PyTree,
    i_out: PyTree,
    l2: float,
):
    """Train for a single step with SAM.

    To match the MATLAB implementation, SAM must be set to opaque mode,
    requiring manual specification of ``grad_fn``.
    """
    ((loss_value, (f, r, v_k)), grads) = j_loss(
        model, i_in_d=i_in_d, i_out=i_out, l2=l2
    )
    updates, opt_state = optim.update(
        grads,
        opt_state,
        model,
        grad_fn=lambda params, _: j_loss(params, i_in_d=i_in_d, i_out=i_out, l2=l2)[1],  # type: ignore
    )
    model = eqx.apply_updates(model, updates)
    return loss_value, f, r, v_k, model, opt_state


LOSS_THRESHOLD = 10
"""Loss threshold for switching the number of bins in approximation_error."""


def train_mat_dnf(
    model: MatDNF,
    i_in: Int[Array, "n l"],
    i_out: Int[Array, "l"],
    er_max: int,
    alpha: float,
    max_itr: int,
    max_try: int,
    extra_itr: int = 0,
    fold: int = 0,
    mode: Literal["dnf", "classifier"] = "dnf",
    use_sam: bool = False,
    use_perturbation: bool = True,
    *,
    key: PRNGKeyArray,
) -> tuple[
    MatDNF,
    Float[Array, ""],
    Int[Array, "<h 2*n"],
]:
    """Mat_DNF training loop.

    Mat_DNF learns a DNF or classifier that maps i_in as closely as possible to i_out and
    returns a DNF formula "learned_DNF" or classifier (c, d) with v_k_th
    from which the predicted output i_out_learned for i_in is computed.

    Args:
        model: MatDNF model to be trained.
        i_in: (n, l) 0-1 matrix of l data points in n variables to be classified.
        i_out: (l,) 0-1 row vector representing target truth values corresponding to i_in.
        er_max: Maximum acceptable approximation error.
        alpha: Learning rate.
        max_itr: Maximum number of iterations.
        max_try: Maximum number of trials to learn a DNF, each with perturbation.
        extra_itr: Number of over-iterations.
        fold: Index for trial number.
        mode: Choose "dnf" or classifier" mode.
        use_sam: Use Sharpness-Aware-Minimization.
        use_perturbation: Use perturbation.
        key: a PRNG key used as the random key.

    Returns:
        model: Learned MatDNF model.
        v_k_th: Threshold for classification such that
            I2_k_learned = (V_k>=V_k_th) where V_k = D_k*(1-min_1(C*[1-I1;I1]))  in {0,1}
        learned_dnf (h', 2n): 0-1 matrix (h'=<h) representing a DNF that approximately gives I2_k when evaluated by I1
            I2_k_learned = ([1..1](1-min_1(learned_DNF*[1-I1;I1])))>=1  in {0,1}

    About "dnf" and "classifier" modes:

    - DNF:
        We learn a DNF=(C:conjunctions,D_k:disjuntion) from (I1,I2_k) by first minimizing J2:

            J2 = (I2_k <> 1-min_1(V_k)) + (1-I2_k <> max_0(V_k)) + (l2/2)*|| C.*(1-C) ||^2 + (l2/2)*|| D.*(1-D) ||^2 => O(n*l)+O(h*2n)

            N = C*[1-I1;I1]     (h x 2n)*(2n x l) = (h x l): continuous #false literal in h1 conjunctinons by I1 => O(h*2n*l)
            M = 1-min_1(N)      (h x l): continuous truth values of h conjunctions by I1 => O(h*l)
            V_k = D_k*M         (1 x h)*(h x l) = (1 x l): continuous truth values of DNF=(C,D_k) by I1:(n x l) => O(1*h*l)

        and then thresholding (C,D_k) to leaned_DNF giving a miminum classification error Er_k_th = |I2_k-I2_k_learned| where

            V_k = sum(1-min_1(leaned_DNF*[1-I1;I1]),1)
            I2_k_learned = (V_k>=1)
                where learned_DNF is a matrix [C1;..;Cm] representing an m-conjunction DNF = C1 v..v Cm

    - Classifier:
        We learn a classifier (C,D_k) from (I1,I2_k) by minimizing J2 while computing threshold V_k_th
        giving miminum classification error Er_k = |I2_k-I2_k_learned| where

            V_k = D_k*(1-min(C*[1-I1;I1],1))
            I2_k_learned = (V_k>=V_k_th)
    """
    l2 = 0.1
    rho = 0.001  # SAM's adversarial SGD rho

    # Duals
    d_i_in = jnp.block([[i_in], [1 - i_in]])
    i_in_d = jnp.block([[1 - i_in], [i_in]])

    # Initial values
    loss_value = math.inf  # Initial cost function value;
    c_extra_itr = 0  # Count how many times E_kr_th=0 happens
    c_th = jnp.zeros(shape=model.c.shape, dtype=i_in.dtype)
    d_k_th = jnp.zeros(shape=model.d_k.shape, dtype=jnp.bool_)
    v_k_th = model.c.dtype.type(math.inf)

    if use_sam:
        optim = optax.contrib.sam(
            adam(alpha=alpha),
            optax.chain(normalize_leaf(), optax.sgd(rho)),
            opaque_mode=True,
        )
        _train_step = train_step_sam
    else:
        optim = adam(alpha=alpha)
        _train_step = train_step

    for i in range(max_try):
        opt_state = optim.init(model)  # type: ignore

        er_k_th = -1
        f = jnp.array(1)  # Anything but 0
        er_k = er_max + 1  # So that it won't just quit

        for j in range(max_itr):
            # Compute layers, loss value, and Jacobian
            _loss_value, f, r, v_k, _model, opt_state = _train_step(
                model=model,
                optim=optim,
                opt_state=opt_state,
                i_in_d=i_in_d,
                i_out=i_out,
                l2=l2,
            )

            print(
                f"trial={fold} i={i} j={j}: (f={f:.3f} r={r:.3f})  "
                f"Er_k={er_k} Er_k_th={er_k_th}/{len(i_out)}  "
                f"|V_k|={jnp.abs(v_k).sum():.2f}  D_k:[{model.d_k.max():.2f}..{model.d_k.min():.2f}] "
                f"C:[{model.c.max():.2f} .. {model.c.min():.2f}] c_extra_itr={c_extra_itr}"
            )
            # J:loss, Er_k:error by (classi), Er_k_th:error by (DNF)(over-itr), V_k:continuous truth values, c_extra_itr:count after erro=0

            # Compute minimum classification error er_k=|i_out-(v_k>=v_k_th)|
            er_k, v_k_th = classification_error(i_out, v_k)

            # Compute minimum approximation error er_k_th by {d_k_th, c_th}
            er_k_th = -1
            if mode == "dnf":
                if loss_value > LOSS_THRESHOLD:
                    split_c = 2
                    split_d_k = 2
                else:
                    split_c = 10
                    split_d_k = 10
                er_k_th, c_th, d_k_th = approximation_error(
                    model.c, model.d_k, d_i_in, i_out, split_c, split_d_k
                )

            if er_k_th <= er_max or c_extra_itr > 0:
                c_extra_itr += 1
            if jnp.isclose(f, 0.0):
                v_k_th = model.c.dtype.type(1)

            if should_break(
                mode=mode,
                f=f.item(),
                er_k=er_k,
                er_k_th=er_k_th,
                er_max=er_max,
                extra_itr=extra_itr,
                c_extra_itr=c_extra_itr,
            ):
                break

            # * NOTE: The updates are made here to match MATLAB/Octave output.
            # * However, this approach returns the model at previous step;
            # * is this the intended behaviour?
            model = _model
            loss_value = _loss_value

        if should_break(
            mode=mode,
            f=f.item(),
            er_k=er_k,
            er_k_th=er_k_th,
            er_max=er_max,
            extra_itr=extra_itr,
            c_extra_itr=c_extra_itr,
        ):
            break

        if use_perturbation:
            key, subkey = random.split(key)
            model = model.perturbate(subkey)

    # Compute learned DNF
    learned_dnf = jnp.array([])
    if mode == "dnf":
        # (h' x 2n): h'(=<h) disjuncts in n variables
        dnf_th = c_th[d_k_th]
        # Rows of non_A_notA in DNF2_th
        n = dnf_th.shape[1] // 2
        bb = dnf_th[:, :n] + dnf_th[:, n:]
        no_A_notA = ~((bb == 2).any(axis=1))  # noqa: PLR2004
        learned_dnf = dnf_th[no_A_notA]

    return model, v_k_th, learned_dnf
