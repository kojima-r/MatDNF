"""Test JAX implementation of Mat_DNF."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike
from numpy import bool_, float64, int8, int16, int32, int64, testing

from mat_dnf.jax.losses import (
    acc_classi,
    acc_dnf,
    approximation_error,
    classification_error,
    logi_conseq,
    logi_equiv,
)
from mat_dnf.jax.models import MatDNF, train_mat_dnf
from mat_dnf.utils import all_bit_seq

from .conftest import iter_matlab_arrays, zip_strict_nonempty

jax.config.update("jax_enable_x64", True)


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
        l2 = i1.shape[1]
        exact_acc_classi = acc_classi(
            d_k=jnp.array(d_k),
            v_k_th=jnp.array(v_k_th),
            i1=jnp.array(i1),
            i2_k=jnp.array(i2_k),
            l2=l2,
            c=jnp.array(c),
        )
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
        exact_acc_dnf = acc_dnf(
            dnf=jnp.array(learned_dnf_s),
            i1=jnp.array(i1),
            i2_k=jnp.array(i2_k),
            l2=l2,
        )
        assert exact_acc_dnf == matlab_acc_dnf


@pytest.mark.parametrize("x", [-1, 0, 1, 3, 4, 11])
def test_all_bit_seq(x: int):
    """Test generation of all bit sequence."""
    if x >= 1:
        assert jnp.unique(all_bit_seq(x), axis=-1).shape[-1] == 2**x
    else:
        with pytest.raises(ValueError):
            all_bit_seq(x)


@pytest.mark.parametrize("sample_dir", ["sample_0", "sample_1", "sample_2"])
def test_approximation_error(sample_dir: str):
    """Test for approximation error."""
    array_dir = Path("tests/resources/approximation_error") / sample_dir
    assert array_dir.exists()

    input_dir, _, input_filenames = next((array_dir / "input").walk())
    output_dir, _, output_filenames = next(array_dir.walk())

    input_arrays = {
        (array_dir / f).stem: jnp.load(input_dir / f) for f in input_filenames
    }
    input_arrays = {k: v if v.ndim > 0 else v.item() for k, v in input_arrays.items()}
    output_arrays = {
        (array_dir / f).stem: jnp.load(output_dir / f) for f in output_filenames
    }

    er_k_th, c_th, d_k_th = approximation_error(**input_arrays)  # type: ignore
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
        (array_dir / f).stem: jnp.load(input_dir / f) for f in input_filenames
    }
    input_arrays = {k: v if v.ndim > 0 else v.item() for k, v in input_arrays.items()}
    output_arrays = {
        (array_dir / f).stem: jnp.load(output_dir / f) for f in output_filenames
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
        cnsq, counter_examples = logi_conseq(
            dnf=jnp.array(learned_dnf_s),
            i_out=jnp.array(i2_k),
            i_in=jnp.array(i1),
        )
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
        eqv, counter_examples = logi_equiv(
            dnf=jnp.array(learned_dnf_s),
            i_out=jnp.array(i2_k),
            i_in=jnp.array(i1),
        )
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
        iter_matlab_arrays("i1_dr", np.int64, directory=directory),
        iter_matlab_arrays("i2_k_dr", np.int64, directory=directory),
        iter_matlab_arrays("er_max", np.float64, directory=directory),
        iter_matlab_arrays("alpha", np.float64, directory=directory),
        iter_matlab_arrays("max_itr", np.int64, directory=directory),
        iter_matlab_arrays("max_try", np.int64, directory=directory),
        iter_matlab_arrays("c_init", np.float64, directory=directory),
        iter_matlab_arrays("d_k_init", np.float64, directory=directory),
        iter_matlab_arrays("c", np.float64, directory=directory),
        iter_matlab_arrays("d_k", np.float64, directory=directory),
        iter_matlab_arrays("v_k_th", np.float64, directory=directory),
        iter_matlab_arrays("learned_dnf", np.bool_, directory=directory),
    ):
        key = jax.random.key(42)
        key, subkey = jax.random.split(key)

        model = MatDNF(c=jnp.array(c_init), d_k=jnp.array(d_k_init))

        model, v_k_th, learned_dnf = train_mat_dnf(
            model=model,
            key=subkey,
            fold=0,
            i_in=jnp.array(i1_dr),
            i_out=jnp.array(i2_k),
            er_max=jnp.array(er_max).item(),
            alpha=jnp.array(alpha).item(),
            max_itr=jnp.array(max_itr).item(),
            max_try=jnp.array(max_try).item(),
            use_perturbation=False,
            use_sam=use_sam,
        )

        for py_array, matlab_array in (
            (model.c, matlab_c),
            (model.d_k[None, :], matlab_d_k),
            (v_k_th, matlab_v_k_th),
            (learned_dnf, matlab_learned_dnf),
        ):
            # TODO: More elegant dimension fix
            if matlab_array.ndim == 1:
                matlab_array = matlab_array[None, :]  # noqa: PLW2901
            testing.assert_allclose(py_array, matlab_array)
