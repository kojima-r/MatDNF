# MatDNF

A Python implementation of MatDNF, an explainable neural network
for learning Boolean Networks (BNs) by minimizing a logically inspired,
non-negative cost function to zero.
Mat_DNF represents Boolean functions in disjunctive normal form (DNFs),
encoding them as pairs of binary matrices and learning them using a single-layer neural network.
This structure ensures that every parameter in the network has a clear
interpretation as a conjunction or literal in the learned DNF.

This package provides three different implementations of Mat_DNF:

1. **NumPy**
   - Provided in [`mat_dnf.numpy`](./mat_dnf/numpy/).
   - Runs on CPU with low overhead, suitable for small-scale problems.
2. **CuPy**
   - Also provided in [`mat_dnf.numpy`](./mat_dnf/numpy/), but is automatically selected when CuPy arrays are used.
   - Leverages GPU acceleration, which is beneficial for large-scale problems, but incurs memory transfer overhead.
   - Identical in usage to NumPy implementation.
3. **JAX**
   - Provided in [`mat_dnf.jax`](./mat_dnf/jax/), with nearly identical usage except for differences in RNG handling.
   - Supports multi-GPU (see JAX's [sharded computation](https://docs.jax.dev/en/latest/sharded-computation.html) and [multi-host environment](https://docs.jax.dev/en/latest/multi_process.html))
     and TPU execution (e.g. in Google Colab).
   - Uses graph-mode automatic differentiation with optional JIT compilation,
     allowing low-overhead and flexible modifications to the network and cost function.
   - Incurs additional compilation overhead (typically <1s), which may be significant for small problems.

All implementations have been tested to produce numerically identical results (within a small tolerance)
to the original MATLAB/Octave implementation (see unit tests in [tests](./tests/)). In general:

- For small problems: NumPy $\gt$ CuPy $\gt$ JAX(CPU) $\gt$ JAX(GPU)
- For large problems: JAX(GPU) $\gt$ CuPy $\gg$ JAX(CPU) $\gt$ NumPy

See example usage in [notebooks/readme_for_test.ipynb](notebooks/readme_for_test.ipynb)

## Table of contents

- [mat-dnf](#mat-dnf)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Benchmarks, etc.](#benchmarks-etc)
    - [Computer specification](#computer-specification)
    - [Initial implementation bottleneck](#initial-implementation-bottleneck)
      - [Run time comparison for different implementations of `approximation_error()`](#run-time-comparison-for-different-implementations-of-approximation_error)
    - [Default random DNF in `readme_for_test`](#default-random-dnf-in-readme_for_test)
    - [Genome-wide AND/OR BNs for Budding Yeast (E_MTAB01)](#genome-wide-andor-bns-for-budding-yeast-e_mtab01)
  - [Known limitations](#known-limitations)

## Installation

```console
$ pip install [-e] .
```

Requires Python 3.12+

Depending on the system configuration, different binary wheels may be required for
both [CuPy](https://cupy.dev/) (GPU-accelerated NumPy and SciPy) and [JAX](https://docs.jax.dev/en/latest/index.html>)
(array library with automatic differentiation and JIT compilation).
These dependencies should be managed through the project's [pyproject.toml](./pyproject.toml)
using a package manager to automatically resolve potential conflicts.

This package was developed with [uv](https://docs.astral.sh/uv/) package manager.
In `uv`, you can add a dependency by running:

```console
$ uv add [package_name]
```

To remove a dependency, use:

```console
$ uv remove [package_name]
```

## Documentation

Requires `sphinx`, `sphinx-autoapi`, and `pydata-sphinx-theme`
(see `dependency-groups.docs` at [pyproject.toml](./pyproject.toml)).
Documentation, including the API reference generated via AutoAPI, can be built by running:

```console
$ cd docs/source
$ sphinx-build -b html . ../build
```

## Benchmarks, etc.

### Computer specification

- CPU: AMD Ryzen 9 7950X (16-Core)
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- 128GB RAM

Note:

- All implementations, including NumPy and Octave, distribute computations accross
  multiple CPU cores by default, as indicated by their CPU usage in `top`.
- Package versions:
  - GNU Octave 6.4.0
  - NumPy 2.2.1
  - CuPy 13.3.0
  - JAX 0.5.0

### Initial implementation bottleneck

A direct one-to-one translation from MATLAB/Octave to Python reveals that
`approximation_error()` is the primary bottleneck:

![py-spy flamegraph generated from `profiling.py`](./docs/source/_static/profile.svg)
_py-spy flamegraph generated from `profiling.py`_

![py-spy flamegraph generated from `profiling02.py`](./docs/source/_static/profile_02.svg)
_py-spy flamegraph generated from `profiling02.py`_

To improve efficiency, the latest implementation of `approximation_error()` and `classification_error()`
vectorizes the inner loop using NumPy broadcasting.
However, since vectorization trades off memory usage for speed,
memory constraints may become a limiting factor in large-scale problems.
If you encounter memory issues, you can switch to a more memory-efficient implementation
by modifying [`losses.py`](./mat_dnf/numpy/losses.py).

For `approximation_error()`, change lines 62-66, from

```python
# Current vectorized implementation in `losses.py` (line  62-66)
d_mat = d_k >= ls_d_k[:, None]  # split_d_k x h
c_mat = c >= ls_c[:, None, None]  # split_c x h x 2n
b_mat = (c_mat @ d_i_in) == c_mat.sum(axis=2, keepdims=True)  # split_c x h x l
e_mat = d_mat @ b_mat  # split_c x split_d_k x l
error_cd = np.abs(i_out - e_mat).sum(axis=2)
```

to

```python
# Alternative implementation: more memory-efficient but slower
for s in range(split_d_k):
    d = d_k >= ls_d_k[s]  # 1 x h
    for t in range(split_c):
        _c = c >= ls_c[t]  # h x 2n
        b = (_c @ d_i_in) == _c.sum(axis=1, keepdims=True)  # h x l
        e = (d @ b) >= 1  # 1 x l
        error_cd[t, s] = np.abs(i_out - e).sum()
```

For `classification error()`, change lines 28-29, from

```python
# Current vectorized implementation in `losses.py` (line  28-29)
d_mat = v_k >= ls_v_k[:, None]  # (split_v_k, l)
error_v_k = xp.abs(i_out - d_mat).sum(axis=-1)  # (split_v_k,)
```

to

```python
# Alternative implementation: more memory-efficient but slower
for s in range(split_v_k):
    d = v_k >= ls_v_k[s]  # 1 x l
    error_v_k[s] = np.abs(i_out - d).sum()
```

Both implementations have been tested for correctness but have not been validated with JAX.
Since JAX generally requires more careful handling of control flow, refer
to the [JAX Control Flow Guide](https://docs.jax.dev/en/latest/control-flow.html#control-flow) for further details.

#### Run time comparison for different implementations of `approximation_error()`

Using MATLAB/Octave arrays generated with the default settings from `readme_for_test.m`:

- Random DNF with no-noise
- `n=5`, `alpha=0.1`;
- `max_try=20`, `max_itr=500`, `h=1000`, `Er_max=0`
- `dr=0.5`
- `i_max=10`
- `d_size=3`, `h_gen=10`, `c_max=5`

| Implementation | Time              | Note                                                       |
| -------------- | ----------------- | ---------------------------------------------------------- |
| NumPy          | 1.5 ms ± 20.5 μs  | Fully vectorized implementation                            |
| CuPy           | 1.04 ms ± 35.5 μs | GPU-accelerated; includes memory transfer overhead         |
| JAX (CPU)      | 1.76 ms ± 7.93 μs | Includes overhead from JIT compilation                     |
| JAX (GPU)      | 106 μs ± 6.38 μs  | Includes both memory transfer and JIT compilation overhead |

See [./notebooks/approximation_error.ipynb](./notebooks/approximation_error.ipynb) for the data source of the table above.

### Default random DNF in `readme_for_test`

- `n=5`; `alpha=0.1`; `max_try=20`; `max_itr=500`; `h=1000`; `Er_max=0`;
- `dr=0.5`; `i_max=10`;
- `d_size=3`; `h_gen=10`; `c_max=5`;

For this problem size, low overhead of CPU execution is more important.

| Implementation | Time          |
| -------------- | ------------- |
| MATLAB/Octave  | 0.020 ± 0.003 |
| NumPy          | 0.008 ± 0.004 |
| CuPy           | 0.126 ± 0.267 |
| JAX (CPU)      | 0.214 ± 0.105 |
| JAX (GPU)      | 0.590 ± 0.300 |

### Genome-wide AND/OR BNs for Budding Yeast (E_MTAB01)

**NOTE**: Either I implemented the CSV-reading function incorrectly,
or I did not know how to set the learning parameters,
but I failed to get the DNF learning running correctly for this problem.

However, given the problem size ($I_1$: 10298 $\times$ 40, $I_2$: 40),
I believe this example is still useful for demonstrating how GPU acceleration
can be beneficial for larger matrices.
Since the exit condition was never triggered, I set
`i_max = 4`, `max_try = 4`, and `max_itr = 50`, resulting in a _fixed_ total of 800 iterations,
with 200 iterations per `Mat_DNF()` / `train_mat_dnf()` invocation.

| Implementation  | Time              |
| --------------- | ----------------- |
| MATLAB / Octave | 741.655 ± 3.069 s |
| NumPy           | 382.977 ± 4.319 s |
| JAX (CPU)       | 262.768 ± 0.460 s |
| CuPy            | 8.930 ± 0.406 s   |
| JAX (GPU)       | 3.127 ± 0.524 s   |

The above table is based on results generated by these scripts:

- [readme_for_test.ipynb](./notebooks/readme_for_test.ipynb)
- [test_emtab.m](./matlab/test-generators/test_emtab.m)

## Known limitations

- DNF simplification is not implemented in JAX because it involves extensive slicing,
  and JAX slicing always creates a copy instead of an array view,
  making it noticeably slower than NumPy/CuPy in practice.
  Furthermore, dynamic slicing is not supported under JIT compilation,
  requiring significant code modifications.
- In some cases, the `c_th` and `d_k_th` matrices (thresholded $\tilde{C}$ and $\tilde{D_k}$?)
  converge to all True / 1, causing the "remove ..A&~A.. conjunction" step in
  DNF simplification to reduce the learned DNF to an empty DNF.
  I'm not sure whether this is a bug or intended behavior.

## Citation
```
@article{sato2023differentiable,
  title={Differentiable learning of matricized DNFs and its application to Boolean networks},
  author={Sato, Taisuke and Inoue, Katsumi},
  journal={Machine Learning},
  volume={112},
  number={8},
  pages={2821--2843},
  year={2023},
  publisher={Springer}
}
```
