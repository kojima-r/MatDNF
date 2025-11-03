"""Test NumPy/CuPy implementation of Mat_DNF."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy import bool_, float64, int8, int16, int32, int64, testing
from numpy.typing import DTypeLike, NDArray

from mat_dnf.numpy.losses import (
    acc_classi,
    acc_dnf,
    approximation_error,
    classification_error,
    logi_conseq,
    logi_equiv,
)
from mat_dnf.numpy.models import (
    MatDNF,
    train_mat_dnf,
)

from .conftest import iter_matlab_arrays, zip_strict_nonempty


@pytest.mark.parametrize("dtype", [int8, int16, int32, int64])
def test_acc_classi(dtype: DTypeLike):
    """Test for some kind of metric for the learned I2."""
    for d_k, v_k_th, i1, i2_k, matlab_exact_acc_classi, c in zip_strict_nonempty(
        iter_matlab_arrays("d_k", float64),
        iter_matlab_arrays("v_k_th", float64),
        iter_matlab_arrays("i1", dtype),
        iter_matlab_arrays("i2_k", dtype),
        iter_matlab_arrays("exact_acc_classi", float64),
        iter_matlab_arrays("c", float64),
    ):
        # Manual type hint fix for non-homogenous iterables
        d_k = cast(NDArray[float64], d_k)
        v_k_th = cast(NDArray[float64], v_k_th)
        c = cast(NDArray[float64], c)

        l2 = i1.shape[1]
        exact_acc_classi = acc_classi(d_k, v_k_th, i1, i2_k, l2, c)
        assert exact_acc_classi == matlab_exact_acc_classi


@pytest.mark.parametrize("dtype", [int8, int16, int32, int64])
def test_acc_dnf(dtype: DTypeLike):
    """Test for some kind of metric for the learned DNF."""
    for learned_dnf_s, i1, i2_k, matlab_acc_dnf in zip_strict_nonempty(
        iter_matlab_arrays("learned_dnf_s", dtype),
        iter_matlab_arrays("i1", dtype),
        iter_matlab_arrays("i2_k", dtype),
        iter_matlab_arrays("exact_acc_dnf", float64),
    ):
        l2 = i1.shape[1]
        exact_acc_dnf = acc_dnf(learned_dnf_s, i1, i2_k, l2)
        assert exact_acc_dnf == matlab_acc_dnf


@pytest.mark.parametrize("sample_dir", ["sample_0", "sample_1", "sample_2"])
def test_approximation_error(sample_dir: str):
    """Test for approximation error."""
    array_dir = Path("tests/resources/approximation_error") / sample_dir
    assert array_dir.exists()

    input_dir, _, input_filenames = next((array_dir / "input").walk())
    output_dir, _, output_filenames = next(array_dir.walk())

    input_arrays = {
        (array_dir / f).stem: np.load(input_dir / f) for f in input_filenames
    }
    output_arrays = {
        (array_dir / f).stem: np.load(output_dir / f) for f in output_filenames
    }

    er_k_th, c_th, d_k_th = approximation_error(**input_arrays)
    testing.assert_allclose(er_k_th, output_arrays["er_k_th"])
    testing.assert_allclose(c_th, output_arrays["c_th"])
    testing.assert_allclose(d_k_th, output_arrays["d_k_th"])


# TODO: Partially share logic with test_approximation_error
@pytest.mark.parametrize("sample_dir", ["sample_0", "sample_1"])
def test_classification_error(sample_dir: str):
    """Test for classification error."""
    array_dir = Path("tests/resources/classification_error") / sample_dir
    assert array_dir.exists()

    input_dir, _, input_filenames = next((array_dir / "input").walk())
    output_dir, _, output_filenames = next(array_dir.walk())

    input_arrays = {
        (array_dir / f).stem: np.load(input_dir / f) for f in input_filenames
    }
    output_arrays = {
        (array_dir / f).stem: np.load(output_dir / f) for f in output_filenames
    }

    er_k, v_k_th = classification_error(**input_arrays)
    testing.assert_allclose(er_k, output_arrays["er_k"])
    testing.assert_allclose(v_k_th, output_arrays["v_k_th"])


@pytest.mark.parametrize("dtype", [int8, int16, int32, int64])
def test_logi_conseq(dtype: DTypeLike):
    """Test for logical consequence."""
    i = 0
    for (
        learned_dnf_s,
        i1,
        i2_k,
        matlab_cnsq,
        matlab_counter_examples,
    ) in zip_strict_nonempty(
        iter_matlab_arrays("learned_dnf_s", dtype),
        iter_matlab_arrays("i1", dtype),
        iter_matlab_arrays("i2_k", dtype),
        iter_matlab_arrays("cnsq", bool_),
        iter_matlab_arrays("cnsq_counter_examples", bool_),
    ):
        i += 1
        cnsq, counter_examples = logi_conseq(learned_dnf_s, i2_k, i1)
        assert cnsq == matlab_cnsq

        if cnsq:
            assert counter_examples is None
            assert matlab_counter_examples.size == 0
        else:
            assert counter_examples is not None
            assert counter_examples.dtype == dtype
            assert matlab_counter_examples.size != 0

            # * NOTE: Fix for when n == 1
            if matlab_counter_examples.ndim == 1:
                matlab_counter_examples = matlab_counter_examples[:, None]  # noqa: PLW2901

            testing.assert_allclose(counter_examples, matlab_counter_examples)


# TODO: Add test where there are _no_ counter examples.
@pytest.mark.parametrize("dtype", [int8, int16, int32, int64])
def test_logi_equiv(dtype: DTypeLike):
    """Test for logical equivalence."""
    for (
        learned_dnf_s,
        i1,
        i2_k,
        matlab_eqv,
        matlab_counter_examples,
    ) in zip_strict_nonempty(
        iter_matlab_arrays("learned_dnf_s", dtype),
        iter_matlab_arrays("i1", dtype),
        iter_matlab_arrays("i2_k", dtype),
        iter_matlab_arrays("eqv", bool_),
        iter_matlab_arrays("eqv_counter_examples", bool_),
    ):
        eqv, counter_examples = logi_equiv(learned_dnf_s, i2_k, i1)
        assert eqv == matlab_eqv
        if eqv:
            assert counter_examples is None
            assert matlab_counter_examples.size == 0
        else:
            assert counter_examples is not None
            assert counter_examples.dtype == dtype
            assert matlab_counter_examples.size != 0

            testing.assert_allclose(counter_examples, matlab_counter_examples)


@pytest.mark.parametrize("use_sam", [True, False])
def test_train_mat_dnf(use_sam: bool):
    """Test whether the trained Mat DNF is consistent with Octave."""
    if use_sam:
        directory = Path("tests/resources/mat_dnf_sam")
    else:
        directory = Path("tests/resources/mat_dnf")

    for (
        i1_dr,
        i2_k,
        er_max,
        alpha,
        max_itr,
        max_try,
        c_init,
        d_k_init,
        matlab_c,
        matlab_d_k,
        matlab_v_k_th,
        matlab_learned_dnf,
    ) in zip_strict_nonempty(
        iter_matlab_arrays("i1_dr", int64, directory=directory),
        iter_matlab_arrays("i2_k_dr", int64, directory=directory),
        iter_matlab_arrays("er_max", int64, directory=directory),
        iter_matlab_arrays("alpha", float64, directory=directory),
        iter_matlab_arrays("max_itr", int64, directory=directory),
        iter_matlab_arrays("max_try", int64, directory=directory),
        iter_matlab_arrays("c_init", float64, directory=directory),
        iter_matlab_arrays("d_k_init", float64, directory=directory),
        iter_matlab_arrays("c", float64, directory=directory),
        iter_matlab_arrays("d_k", float64, directory=directory),
        iter_matlab_arrays("v_k_th", float64, directory=directory),
        iter_matlab_arrays("learned_dnf", bool_, directory=directory),
    ):
        # Manual type hint fix for non-homogenous iterables
        i1_dr = cast(NDArray[int64], i1_dr)
        i2_k = cast(NDArray[int64], i2_k)
        er_max = cast(NDArray[int64], er_max)
        alpha = cast(NDArray[float64], alpha)
        max_itr = cast(NDArray[int64], max_itr)
        max_try = cast(NDArray[int64], max_try)
        c_init = cast(NDArray[float64], c_init)
        d_k_init = cast(NDArray[float64], d_k_init)

        model = MatDNF(c=c_init, d_k=d_k_init)

        model, v_k_th, learned_dnf,_ = train_mat_dnf(
            model=model,
            i_in=i1_dr,
            i_out=i2_k,
            er_max=er_max.item(),
            alpha=alpha.item(),
            max_itr=max_itr.item(),
            max_try=max_try.item(),
            use_perturbation=False,
            use_sam=use_sam,
            rng=np.random.default_rng(42),
        )

        for py_array, matlab_array in (
            (model.c, matlab_c),
            (model.d_k, matlab_d_k.flatten()),
            (v_k_th, matlab_v_k_th),
            (learned_dnf, matlab_learned_dnf),
        ):
            testing.assert_allclose(py_array, matlab_array)
