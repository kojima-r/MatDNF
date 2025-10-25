"""Test utility functions."""

import numpy as np
import pytest
from numpy import bool_, int8, int16, int32, int64, random, testing
from numpy.typing import DTypeLike

from mat_dnf.simplifications import simp_dnf
from mat_dnf.utils import all_bit_seq, eval_dnf, gen_dnf

from .conftest import iter_matlab_arrays, zip_strict_nonempty


@pytest.mark.parametrize("x", [-1, 0, 1, 3, 4, 11])
def test_all_bit_seq(x: int):
    """Test generation of all bit sequence."""
    if x >= 1:
        assert np.unique(all_bit_seq(x), axis=-1).shape[-1] == 2**x
    else:
        with pytest.raises(ValueError):
            all_bit_seq(x)


@pytest.mark.parametrize("dtype", [int8, int16, int32, int64])
def test_eval_dnf(dtype: DTypeLike):
    """Test DNF evaluation."""
    for c0, d0, i1, matlab_i2_k in zip_strict_nonempty(
        iter_matlab_arrays("c0", dtype),
        iter_matlab_arrays("d0", dtype),
        iter_matlab_arrays("i1", dtype),
        iter_matlab_arrays("i2_k", bool_),
    ):
        python_i2 = eval_dnf(d0, c0, i1)

        assert python_i2.dtype == dtype
        testing.assert_allclose(python_i2, matlab_i2_k)


def test_gen_dnf():
    """Test generation of random DNF."""
    n = 5
    h_gen = 10
    d_size = 3
    c_max = 5

    rng = random.default_rng()

    for _ in range(10):
        d0, c0 = gen_dnf(n=n, h_gen=h_gen, d_size=d_size, c_max=c_max, rng=rng)

        assert d0.shape == (h_gen,)
        assert d0.sum() == d_size

        assert c0.shape == (h_gen, 2 * n)
        for row in c0:
            p_row, n_row = row[:n], row[n:]
            assert (~np.logical_and(p_row, n_row)).all()


@pytest.mark.parametrize("dtype", [int8, int16, int32, int64])
def test_simp_dnf(dtype: DTypeLike):
    """Test simplification of DNFs."""
    for c0, d0, matlab_simp_dnf in zip_strict_nonempty(
        iter_matlab_arrays("c0", dtype),
        iter_matlab_arrays("d0", bool_),
        iter_matlab_arrays("dnf", bool_),
    ):
        matlab_simp_dnf = (  # noqa: PLW2901
            matlab_simp_dnf.reshape(1, -1)
            if matlab_simp_dnf.ndim == 1
            else matlab_simp_dnf
        )
        python_simp_dnf = simp_dnf(c0[d0])

        assert python_simp_dnf.dtype == dtype
        testing.assert_allclose(python_simp_dnf, matlab_simp_dnf)
