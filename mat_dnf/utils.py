"""Random problem generations, log prints, etc."""

import math
from pathlib import Path
from typing import Literal, overload

import cupy as cp
import numpy as np
from numpy import bool_, floating, generic, int64, number
from numpy.random import Generator
from numpy.typing import DTypeLike, NBitBase, NDArray


class MeanLogger:
    """Averages MatDNF runs, following implementation in `readme_for_test.m`."""

    def __init__(self, dr: float, l: int):
        """Initialize empty mean logger."""
        # Parameters only used for printing; to match MATLAB.
        self._dr = dr
        self._l = l

        self._arr_test_size: list[int] = []
        self._arr_time: list[float] = []
        self._arr_acc_classi: list[float] = []
        self._arr_acc_dnf: list[float] = []
        self._arr_conseq: list[bool] = []
        self._arr_equiv: list[bool] = []

    def append(
        self,
        test_size: int,
        time: float,
        acc_classi: float,
        acc_dnf: float,
        conseq: bool,
        equiv: bool,
    ):
        """Append to internal lists."""
        self._arr_test_size.append(test_size)
        self._arr_time.append(time)
        self._arr_acc_classi.append(acc_classi)
        self._arr_acc_dnf.append(acc_dnf)
        self._arr_conseq.append(conseq)
        self._arr_equiv.append(equiv)

    def __str__(self):
        """String representation, matching `readme_for_tst.m`."""
        mean_acc_dnf = np.array(self._arr_acc_dnf).mean()
        mean_acc_classi = np.array(self._arr_acc_classi).mean()
        mean_test_size = np.array(self._arr_test_size).mean()
        mean_conseq = np.array(self._arr_conseq).mean()
        mean_equiv = np.array(self._arr_equiv).mean()
        mean_time = np.array(self._arr_time).mean()
        std_time = np.array(self._arr_time).std()

        # All lists have equal length by construction.
        n_trials = len(self._arr_test_size)

        return (
            f"dr = {self._dr}, average over {n_trials} trials\n"
            f"exact_acc_DNF   exact_acc_classi   |test_size|/|I1|   conseq    equiv    time(s)\n"
            f"{mean_acc_dnf:0.3f}           {mean_acc_classi:0.3f}              "
            f"{mean_test_size:0.1f}/{self._l}            {mean_conseq:0.3f}"
            f"     {mean_equiv:0.3f}    {mean_time:0.6f} ± {std_time:0.6f}\n"
        )


def eval_dnf(
    d: NDArray[number], c: NDArray[number], i_in: NDArray[number]
) -> NDArray[number]:
    """Evaluate DNF by I_in.

    % D(1 x h),C(h x 2n): binary {0,1}
    % I_in(n x l): l assignments over n variables
    % I_out(1 x l): truth values of (D,C) by I_in
    """
    xp = cp.get_array_module(d, c, i_in)
    # s = xp.vstack([1 - i_in, i_in])
    s = xp.vstack((1 - i_in, i_in))
    M = 1 - xp.minimum(c @ s, 1)
    return (d @ M >= 1).astype(d.dtype)


@overload
def all_bit_seq[T: number](n: int, dtype: np.dtype[T]) -> NDArray[T]:
    pass


@overload
def all_bit_seq(n: int) -> NDArray[int64]:
    pass


def all_bit_seq[T: number](n: int, dtype: DTypeLike = int64) -> NDArray[T]:
    """Vectors of all possible bit strings of length n."""
    if n < 1:
        raise ValueError("Bit string length must be >= 1.")
    return np.stack(np.meshgrid(*([[0, 1]] * n)), dtype=dtype).reshape(n, -1)


@overload
def gen_dnf[T: generic](
    n: int,
    h_gen: int,
    d_size: int,
    c_max: int,
    rng: np.random.Generator,
    dtype: np.dtype[T],
) -> tuple[NDArray[T], NDArray[T]]:
    pass


@overload
def gen_dnf(
    n: int,
    h_gen: int,
    d_size: int,
    c_max: int,
    rng: np.random.Generator,
) -> tuple[NDArray[int64], NDArray[int64]]:
    pass


def gen_dnf(
    n: int,
    h_gen: int,
    d_size: int,
    c_max: int,
    rng: np.random.Generator,
    dtype: DTypeLike = int64,
):
    """Generate a random DNF formula F.

    F in {a1...an} = (rand_D(1 x h_gen),rand_C(h_gen x 2n)).
    with d_size disjuncts where each disjunct contains at most c_max literals
    a half of which is negative on average

    REQUIRED: n >= c_max, h_gen >= d_size
    [D C] = gen_DNF(n,h_gen=10,d_size=10,c_max=5); x = C(find(D),:); simp_DNF(x)
    [D C] = gen_DNF(n,h_gen=10,d_size=3,c_max=5); x = C(find(D),:); simp_DNF(x)

    Args:
        n: Number of variables.
        h_gen: Most number of disjuncts?
        d_size: Number of disjuncts.
        c_max: Maximum number of literals in each disjuncts.
        rng: Random number generator.
        dtype: Desired dtype of the generated D and C matrices.
    """
    rand_c = np.zeros((h_gen, 2 * n), dtype=bool_)
    rand_d = np.zeros(h_gen, dtype=bool_)

    c_max0 = min(c_max, n)  # max conjunction size <= number of variables

    for i in range(h_gen):
        c_size = rng.integers(
            low=1, high=c_max0, endpoint=True
        )  # conjunction size <= c_max0
        y = rng.choice(n, c_size, replace=False)  # y = [y_1..y_c_size] in {1..n}
        w = rng.random(c_size) > 0.5  # noqa: PLR2004
        rand_c[i, y[w]] = True  # positive literal, rand_C[i, :] = y_1&..&~y_c_size
        rand_c[i, y[~w] + n] = True  # negative literal, rand_C(i,:) = y_1&..&~y_c_size
    z = rng.choice(h_gen, d_size, replace=False)
    rand_d[z] = True

    return rand_d.astype(dtype), rand_c.astype(dtype)


@overload
def n_parity_function[T: number](
    rng: Generator, n: int, add_noise: bool, dtype: np.dtype[T]
) -> tuple[NDArray[T], NDArray[T]]:
    pass


@overload
def n_parity_function(
    rng: Generator, n: int, add_noise: bool
) -> tuple[NDArray[int64], NDArray[int64]]:
    pass


def n_parity_function[T: number](
    rng: Generator, n: int, add_noise: bool, dtype: np.dtype[T] = np.dtype(int64)
) -> tuple[NDArray[T], NDArray[T]]:
    """Create N-parity function."""
    i0 = all_bit_seq(n, dtype=dtype)
    l = i0.shape[1]

    i1 = i0[:, rng.permutation(l)]
    if add_noise:
        i1 = np.vstack([i1, (rng.random((n, l)) < 0.5).astype(dtype)])  # noqa: PLR2004
        i1 = i1[:, rng.permutation(i1.shape[1])]
        i2_k = np.remainder(np.sum(i1[:n], axis=0), 2)
        # * NOTE: Is it supposed to work even with (2n, l) size?
        i1 = i1[:n]
    else:
        i2_k = np.remainder(np.sum(i1, axis=0), 2)
    return i1, i2_k


@overload
def random_function[T: number](
    rng: Generator, n: int, add_noise: bool, dtype: np.dtype[T]
) -> tuple[NDArray[T], NDArray[T]]:
    pass


@overload
def random_function(
    rng: Generator, n: int, add_noise: bool
) -> tuple[NDArray[int64], NDArray[int64]]:
    pass


def random_function[T: number](
    rng: Generator, n: int, add_noise: bool, dtype: np.dtype[T] = np.dtype(int64)
) -> tuple[NDArray[T], NDArray[T]]:
    """Create random function."""
    i0 = all_bit_seq(n, dtype=dtype)
    l = i0.shape[1]

    i1 = i0[:, rng.permutation(l)]
    if add_noise:
        i1 = np.vstack([i1, (rng.random((n, l)) < 0.5).astype(dtype)])  # noqa: PLR2004
        # * NOTE: Otherwise the noise bits are only at the end...
        i1 = i1[:, rng.permutation(i1.shape[1])]
        # * NOTE: Is it supposed to work even with (2n, l) size?
        i1 = i1[:n]

    i2_k = (rng.random(l) < 0.5).astype(dtype)  # noqa: PLR2004
    return i1, i2_k


@overload
def random_dnf[T: number](
    rng: Generator,
    n: int,
    h_gen: int,
    d_size: int,
    c_max: int,
    add_noise: bool,
    dtype: np.dtype[T],
) -> tuple[NDArray[T], NDArray[T]]:
    pass


@overload
def random_dnf(
    rng: Generator, n: int, h_gen: int, d_size: int, c_max: int, add_noise: bool
) -> tuple[NDArray[int64], NDArray[int64]]:
    pass


def random_dnf[T: number](
    rng: Generator,
    n: int,
    h_gen: int,
    d_size: int,
    c_max: int,
    add_noise: bool,
    dtype: np.dtype[T] = np.dtype(int64),
) -> tuple[NDArray[T], NDArray[T]]:
    """Create random DNF."""
    d0, c0 = gen_dnf(
        n=n,
        h_gen=h_gen,
        d_size=d_size,
        c_max=c_max,
        rng=rng,
        dtype=dtype,
    )

    i0 = all_bit_seq(n, dtype=dtype)
    l = i0.shape[1]

    if add_noise:
        i1 = np.vstack([i0, (rng.random((n, l)) < 0.5).astype(i0.dtype)])  # noqa: PLR2004
        i1 = i1[:, rng.permutation(i1.shape[1])]
        i2_k = eval_dnf(d0, c0, i1[:n])
        # * NOTE: Is it supposed to work even with (2n, l) size?
        i1 = i1[:n]
    else:
        i1 = i0[:, rng.permutation(l)]
        i2_k = eval_dnf(d0, c0, i1)
    return i1.astype(dtype), i2_k.astype(dtype)


@overload
def read_nth_dnf[T: number](
    fname: str | Path, i: int, delimiter: str, dtype: np.dtype[T]
) -> tuple[NDArray[T], NDArray[T]]: ...


@overload
def read_nth_dnf(
    fname: str | Path, i: int, delimiter: str
) -> tuple[NDArray[int64], NDArray[int64]]: ...


def read_nth_dnf[T: number](
    fname: str | Path, i: int, delimiter: str = ",", dtype: DTypeLike = int64
) -> tuple[NDArray[number], NDArray[number]]:
    """Read nth DNF from a CSV."""
    dnfs = np.loadtxt(fname, delimiter=delimiter, dtype=dtype)
    return dnfs[:, :-1], dnfs[i][1:]


def should_break(
    mode: Literal["dnf", "classifier"],
    f: float | floating[NBitBase],
    er_k: int | number,
    er_k_th: int | number,
    er_max: int | number,
    extra_itr: int,
    c_extra_itr: int,
) -> bool:
    """Terminating condition for MatDNF training loop."""
    if mode == "dnf":  # DNF
        if er_k_th <= er_max:
            if extra_itr > 0:  # over-itr
                if er_k_th <= er_max and c_extra_itr >= extra_itr:
                    return True
            elif er_k_th <= er_max:
                return True
    elif math.isclose(f, 0.0) or er_k <= er_max:  # classifier
        return True
    return False
