"""Core Mat_DNF algorithm."""

import math
from functools import reduce
from types import ModuleType
from typing import Literal, Self

import cupy as cp
import numpy as np
from numpy import float32, float64, floating, integer
from numpy.random import Generator
from numpy.typing import NBitBase, NDArray

from ..simplifications import remove_a_not_a
from ..utils import should_break
from .losses import approximation_error, classification_error
from .optimizers import Adam

RNGDType = float32 | float64 | Literal["float32"] | Literal["float64"]


class MatDNF[T: NBitBase]:
    """MatDNF neural network."""

    def __init__(self, c: NDArray[floating[T]], d_k: NDArray[floating[T]], aa: int = 4):
        """Initialize MatDNF NN with initial C and D_K arrays."""
        self._c = c
        self._d_k = d_k
        self.aa = aa

    @property
    def c(self) -> NDArray[floating[T]]:
        """C array."""
        return self._c

    @property
    def d_k(self) -> NDArray[floating[T]]:
        """D_k array."""
        return self._d_k

    @classmethod
    def create_random(
        cls,
        h: int,
        n: int,
        aa: int = 4,
        dtype: RNGDType = "float64",
        xp: ModuleType = np,
        *,
        rng: Generator,
    ):
        """Create a MatDNF NN with random C and D_k of the specified shape.

        Args:
            h: Maximum number of conjunctions in a DNF.
            n: Number of variables.
            aa: Scaling factor for the random initialization.
            dtype: DType of MatDNF arrays.
            xp: Array module. CuPy or NumPy.
            rng: A NumPy or CuPy random number generator.
        """
        c = (
            xp.sqrt(aa / (h * 2 * n), dtype=dtype)
            * rng.standard_normal(size=(h, 2 * n), dtype=dtype)
            + 0.5
        )
        d_k = (
            xp.sqrt(aa / h, dtype=dtype) * rng.standard_normal((h,), dtype=dtype) + 0.5
        )
        return cls(c=c, d_k=d_k, aa=aa)

    def __call__[U: NBitBase](
        self, i_in_d: NDArray[integer[U]]
    ) -> tuple[
        NDArray[floating[T]],
        tuple[NDArray[floating[T]], NDArray[integer[U]]],
    ]:
        """Compute v_k.

        Also returns auxiliary arrays to avoid repeated computation.
        """
        xp = cp.get_array_module(self.c, self.d_k, i_in_d)
        # Single hidden layer
        N = self.c @ i_in_d  # (h, l); O(h*2n*l)
        M = 1 - xp.minimum(N, 1)  # (h, l); O(h*l)  # ReLU
        # Linear output layer
        v_k = self.d_k @ M  # (l,); O(h*l) Analogue disjunction output
        return v_k, (N, M)

    def perturbate(self, rng: Generator) -> Self:
        """Equation (5)."""
        xp = cp.get_array_module(self.c, self.d_k)
        c0 = (  # (h, 2*n)
            xp.sqrt(self.aa / reduce(lambda x, y: x * y, self.c.shape))
            * rng.random(self.c.shape)
            + 0.5
        )
        c = 0.5 * (self.c + c0)
        d_k0 = (  # (h,)
            xp.sqrt(self.aa / reduce(lambda x, y: x * y, self.d_k.shape))
            * rng.random(self.d_k.shape)
            + 0.5
        )
        d_k = 0.5 * (self.d_k + d_k0)

        return type(self)(c=c, d_k=d_k, aa=self.aa)


def j_loss[T: NBitBase, U: NBitBase](
    model: MatDNF[T],
    i_in_d: NDArray[integer[U]],
    i_out: NDArray[integer[U]],
    l2: float = 0.1,
):
    """The J cost function (Equation 3).

    Also returns auxiliary arrays to avoid repeated computation.
    """
    xp = cp.get_array_module(model.c, model.d_k)

    v_k, (N, M) = model(i_in_d)

    Y = model.c * (1 - model.c)  # (h, l); O(h*2n*l)
    Z = model.d_k * (1 - model.d_k)

    # Compute loss value and Jacobian
    W = v_k  # v_k = d_k * (1 - min(c * [1-i_in; i_in], 1))
    f = xp.dot(i_out, 1 - xp.minimum(W, 1)) + xp.dot(1 - i_out, xp.maximum(W, 0))
    r = l2 * 0.5 * (xp.einsum("ij,ij->j", Y, Y).sum() + xp.dot(Z, Z))
    return f + r, (v_k, N, M, Y, Z, f, r)  # O(h*2n*l)


def grad_j_loss[T: NBitBase, U: NBitBase](
    model: MatDNF[T],
    i_in_d: NDArray[integer[U]],
    i_out: NDArray[integer[U]],
    v_k: NDArray[floating[T]],
    N: NDArray[floating[T]],
    M: NDArray[integer[U]],
    Y: NDArray[floating[T]],
    Z: NDArray[floating[T]],
    l2: float = 0.1,
):
    """Gradient of J cost function."""
    xp = cp.get_array_module(model.c, model.d_k)

    W = v_k  # v_k = d_k * (1 - min(c * [1-i_in; i_in], 1))
    X = -((W <= 1) * i_out) + (W >= 0) * (1 - i_out)
    J_c = -((N <= 1) * xp.outer(model.d_k, X)) @ i_in_d.T + l2 * (1 - 2 * model.c) * Y
    J_d_k = X @ M.T + l2 * (1 - 2 * model.d_k) * Z
    return J_c, J_d_k


LOSS_THRESHOLD = 10
"""Loss threshold for switching the number of bins in approximation_error."""


def train_mat_dnf[T: NBitBase, U: NBitBase](
    model: MatDNF[T],
    i_in: NDArray[integer[U]],
    i_out: NDArray[integer[U]],
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
    rng: Generator,
) -> tuple[
    MatDNF[T],
    floating[T],
    NDArray[integer[U]],
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
        mode: Choose "dnf" or "classifier" mode.
        use_sam: Use Sharpness-Aware-Minimization.
        use_perturbation: Use perturbation.
        rng: NumPy random number generator.

    Returns:
        model: Learned MatDNF model.
        v_k_th: Threshold for classification such that
            I2_k_learned = (V_k>=V_k_th) where V_k = D_k*(1-min_1(C*[1-I1;I1]))  in {0,1}
        learned_dnf: (h', 2n) 0-1 matrix (h'=<h) representing a DNF that approximately gives I2_k when evaluated by I1
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

    xp = cp.get_array_module(model.c, model.d_k, i_in, i_out)

    # Duals
    d_i_in = xp.vstack([i_in, 1 - i_in])
    i_in_d = xp.vstack([1 - i_in, i_in])

    # Initial values.
    loss_value = math.inf  # Initial cost function value;
    c_extra_itr = 0  # Count how many times E_kr_th=0 happens
    c_th = xp.zeros(shape=model.c.shape, dtype=i_in.dtype)
    d_k_th = xp.zeros(shape=model.d_k.shape, dtype=xp.bool_)
    v_k_th = model.c.dtype.type(math.inf)

    for i in range(max_try):
        c_optim = Adam(alpha=alpha, xp=xp)
        d_k_optim = Adam(alpha=alpha, xp=xp)

        er_k_th = -1
        f = 1  # Anything but 0
        er_k = er_max + 1  # So that it won't just quit

        for j in range(max_itr):
            # Compute layers, loss value, and Jacobian
            _loss_value, (v_k, N, M, Y, Z, f, r) = j_loss(
                model=model, i_in_d=i_in_d, i_out=i_out, l2=l2
            )
            J_c, J_d_k = grad_j_loss(
                model=model,
                i_in_d=i_in_d,
                i_out=i_out,
                v_k=v_k,
                N=N,
                M=M,
                Y=Y,
                Z=Z,
                l2=l2,
            )
            if use_sam:
                # Adversarial SGD
                e_hat_c = rho / xp.sqrt(xp.sum(J_c**2)) * J_c
                e_hat_d_k = rho / xp.sqrt(xp.sum(J_d_k**2)) * J_d_k
                model_hat = MatDNF(c=model.c + e_hat_c, d_k=model.d_k + e_hat_d_k)
                _, (v_k_hat, N_hat, M_hat, Y_hat, Z_hat, _, _) = j_loss(
                    model=model_hat, i_in_d=i_in_d, i_out=i_out, l2=l2
                )
                J_c, J_d_k = grad_j_loss(
                    model=model_hat,
                    i_in_d=i_in_d,
                    i_out=i_out,
                    v_k=v_k_hat,
                    N=N_hat,
                    M=M_hat,
                    Y=Y_hat,
                    Z=Z_hat,
                    l2=l2,
                )

            # Update parameter values
            _c = c_optim.update(model.c, J_c)
            _d_k = d_k_optim.update(model.d_k, J_d_k)

            # Logs and metrics
            print(
                f"trial={fold} i={i} j={j}: (f={f:.3f} r={r:.3f})  "
                f"Er_k={er_k} Er_k_th={er_k_th}/{len(i_out)}  "
                f"|V_k|={xp.abs(v_k).sum():.2f}  D_k:[{model.d_k.max():.2f}..{model.d_k.min():.2f}] "
                f"C:[{model.c.max():.2f} .. {model.c.min():.2f}] c_extra_itr={c_extra_itr}"
            )
            # J:loss, er_k:error by (classi), er_k_th:error by (DNF)(over-itr), v_k:continuous truth values, c_extra_itr:count after erro=0

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
            if xp.isclose(f, 0.0):
                v_k_th = model.c.dtype.type(1)

            if should_break(
                mode=mode,
                f=f,
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
            model = MatDNF(c=_c, d_k=_d_k, aa=model.aa)
            loss_value = _loss_value

        if should_break(
            mode=mode,
            f=f,
            er_k=er_k,
            er_k_th=er_k_th,
            er_max=er_max,
            extra_itr=extra_itr,
            c_extra_itr=c_extra_itr,
        ):
            break

        if use_perturbation:
            model.perturbate(rng=rng)

    # Compute learned DNF
    learned_dnf = xp.array([])
    if mode == "dnf":
        # (h', 2n): h'(<=h) disjuncts in n variables
        dnf_th = c_th[d_k_th]
        # Rows of non_A_notA in DNF2_th
        n = dnf_th.shape[1] // 2
        bb = dnf_th[:, :n] + dnf_th[:, n:]
        no_A_notA = ~((bb == 2).any(axis=1))  # noqa: PLR2004
        learned_dnf = dnf_th[no_A_notA]

    return model, v_k_th, learned_dnf,  (c_th, d_k_th)
