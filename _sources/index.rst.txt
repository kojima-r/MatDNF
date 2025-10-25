``mat-dnf`` User Guide
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


A Python implementation of Mat_DNF, an explainable neural network
for learning Boolean Networks (BNs) by minimizing a logically inspired,
non-negative cost function to zero.
Mat_DNF represents Boolean functions in disjunctive normal form (DNFs),
encoding them as pairs of binary matrices and learning them using a single-layer neural network.
This structure ensures that every parameter in the network has a clear
interpretation as a conjunction or literal in the learned DNF.

This package provides three different implementations of Mat_DNF:


- **NumPy**  

  - Provided in :mod:`mat_dnf.numpy <mat_dnf.numpy>`.  

  - Runs on CPU with low overhead, suitable for small-scale problems.  


- **CuPy**  

  - Also provided in :mod:`mat_dnf.numpy <mat_dnf.numpy>`, but is automatically selected when CuPy arrays are used.  

  - Leverages GPU acceleration, which is beneficial for large-scale problems, but incurs memory transfer overhead.  

  - Identical in usage to NumPy implementation.  


- **JAX**  
  - Provided in :mod:`mat_dnf.jax <mat_dnf.jax>`, with nearly identical usage except for differences in RNG handling.  

  - Supports multi-GPU (see JAX's `sharded computation <https://docs.jax.dev/en/latest/sharded-computation.html>`_ and `multi-host environment <https://docs.jax.dev/en/latest/multi_process.html>`_) 
and TPU execution (e.g. in Google Colab).  

  - Uses graph-mode automatic differentiation with optional JIT compilation, allowing low-overhead and flexible modifications to the network and cost function.  

  - Incurs additional compilation overhead (typically <1s), which may be significant for small problems.  



All implementations have been tested to produce numerically identical results (within a small tolerance)
to the original MATLAB/Octave implementation (see unit tests in :file:`./tests`). In general:

- For small problems: NumPy > CuPy > JAX(CPU) > JAX(GPU)
- For large problems: JAX(GPU) > CuPy >> JAX(CPU) > NumPy

See example usage in :file:`./notebooks/readme_for_test`.

Installation
------------

.. code-block:: console

   $ pip install [-e] .

Requires Python 3.12+

Depending on the system configuration, different binary wheels may be required for
both `CuPy <https://cupy.dev/>`_ (GPU-accelerated NumPy and SciPy) and `JAX <https://docs.jax.dev/en/latest/index.html>`_
(array library with automatic differentiation and JIT compilation).
These dependencies should be managed through the project's :file:`pyproject.toml`
using a package manager to automatically resolve potential conflicts.

This package was developed with `uv <https://docs.astral.sh/uv/>`_ package manager.
In ``uv``, you can add a dependency by running:

.. code-block:: console

   $ uv add [package_name]

To remove a dependency, use:

.. code-block:: console

   $ uv remove [package_name]


Benchmarks, etc.
----------------

Computer Specification
^^^^^^^^^^^^^^^^^^^^^^

- CPU: AMD Ryzen 9 7950X (16-Core)
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- RAM: 128GB

Notes:

- All implementations, including NumPy and Octave, distribute computations across
  multiple CPU cores by default, as indicated by their CPU usage in ``top``.

- Package versions:

  - GNU Octave 6.4.0

  - NumPy 2.2.1

  - CuPy 13.3.0

  - JAX 0.5.0


Initial implementation bottleneck
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A direct one-to-one translation from MATLAB/Octave to Python reveals that
``approximation_error()`` is the primary bottleneck:


.. figure:: _static/profile.svg
   :alt: Flamegraph generated from ``profiling.py``
   :align: center
   :width: 600px

   ``py-spy`` flamegraph generated from ```profiling.py``


.. figure:: _static/profile_02.svg
   :alt: Flamegraph generated from ``profiling02.py``
   :align: center
   :width: 600px

   ``py-spy`` flamegraph generated from ```profiling_02.py``


To improve efficiency, the latest implementation of ``approximation_error()`` and ``classification_error()``
vectorizes the inner loop using NumPy broadcasting.
However, since vectorization trades off memory usage for speed,
memory constraints may become a limiting factor in large-scale problems.
If you encounter memory issues, you can switch to a more memory-efficient implementation
by modifying :file:`losses.py`.

For ``approximation_error()``, change lines 62-66, from

.. code-block:: python

   # Current vectorized implementation in `losses.py` (line  62-66)
   d_mat = d_k >= ls_d_k[:, None]  # split_d_k x h
   c_mat = c >= ls_c[:, None, None]  # split_c x h x 2n
   b_mat = (c_mat @ d_i_in) == c_mat.sum(axis=2, keepdims=True)  # split_c x h x l
   e_mat = d_mat @ b_mat  # split_c x split_d_k x l
   error_cd = np.abs(i_out - e_mat).sum(axis=2)


to

.. code-block:: python

   # Alternative implementation: more memory-efficient but slower
   for s in range(split_d_k):
       d = d_k >= ls_d_k[s]  # 1 x h
       for t in range(split_c):
           _c = c >= ls_c[t]  # h x 2n
           b = (_c @ d_i_in) == _c.sum(axis=1, keepdims=True)  # h x l
           e = (d @ b) >= 1  # 1 x l
           error_cd[t, s] = np.abs(i_out - e).sum()


For ``classification error()``, change lines 28-29, from

.. code-block:: python

   # Current vectorized implementation in `losses.py` (line  28-29)
   d_mat = v_k >= ls_v_k[:, None]  # (split_v_k, l)
   error_v_k = xp.abs(i_out - d_mat).sum(axis=-1)  # (split_v_k,)


to

.. code-block:: python

   # Alternative implementation: more memory-efficient but slower
   for s in range(split_v_k):
       d = v_k >= ls_v_k[s]  # 1 x l
       error_v_k[s] = np.abs(i_out - d).sum()


Both implementations have been tested for correctness but have not been validated with JAX.
Since JAX generally requires more careful handling of control flow, refer
to the `JAX Control Flow Guide <https://docs.jax.dev/en/latest/control-flow.html#control-flow>`_ for further details.


Run Time Comparison
^^^^^^^^^^^^^^^^^^^

Using MATLAB/Octave arrays generated with the default settings from ``readme_for_test.m``:

- Random DNF with no-noise
- ``n = 5``, ``alpha = 0.1``
- ``max_try = 20``, ``max_itr = 500``, ``h = 1000``, ``Er_max = 0``
- ``dr = 0.5``
- ``i_max = 10``
- ``d_size = 3``, ``h_gen = 10``, ``c_max = 5``


.. list-table:: Run time comparison for different implementations of ``approximation_error()```
   :header-rows: 1
   :widths: 20 25 55

   * - Implementation
     - Time
     - Notes
   * - NumPy
     - 1.5 ms ± 20.5 μs
     - Fully vectorized implementation
   * - CuPy
     - 1.04 ms ± 35.5 μs
     - GPU-accelerated; includes memory transfer overhead
   * - JAX (CPU)
     - 1.76 ms ± 7.93 μs
     - Includes overhead from JIT compilation
   * - JAX (GPU)
     - 106 μs ± 6.38 μs
     - Includes both memory transfer and JIT compilation overhead


See :file:`./notebooks/approximation_error.ipynb` for the data source of the table above.


Default Random DNF in ``readme_for_test``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``n=5``; ``alpha=0.1``; ``max_try=20``; ``max_itr=500``; ``h=1000``; ``Er_max=0``;
- ``dr=0.5``; ``i_max=10``;
- ``d_size=3``; ``h_gen=10``; ``c_max=5``;

For this problem size, minimizing CPU overhead is more important.

.. list-table:: Execution time for default random DNF in ``readme_for_test``
   :header-rows: 1
   :widths: 25 25

   * - Implementation
     - Time
   * - MATLAB/Octave
     - 0.020 ± 0.003 s
   * - NumPy
     - 0.008 ± 0.004 s
   * - CuPy
     - 0.126 ± 0.267 s
   * - JAX (CPU)
     - 0.214 ± 0.105 s
   * - JAX (GPU)
     - 0.590 ± 0.300 s


Genome-wide AND/OR BNs for Budding Yeast (E_MTAB01)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NOTE**: Either I implemented the CSV-reading function incorrectly,
or I did not know how to set the learning parameters,
but I failed to get the DNF learning running correctly for this problem.

However, given the problem size (:math:`I_1`: 10298 × 40, :math:`I_2`: 40),
I believe this example is still useful for demonstrating how GPU acceleration
can be beneficial for larger matrices.
Since the exit condition was never triggered, I set
``i_max = 4``, ``max_try = 4``, and ``max_itr = 50``, resulting in a *fixed* total of 800 iterations,  
with 200 iterations per ``Mat_DNF()`` / ``train_mat_dnf()`` invocation.

.. list-table:: Execution Time for Genome-wide AND/OR BNs (E_MTAB01)
   :header-rows: 1
   :widths: 30 30

   * - Implementation
     - Time
   * - MATLAB / Octave
     - 741.655 ± 3.069 s
   * - NumPy
     - 382.977 ± 4.319 s
   * - JAX (CPU)
     - 262.768 ± 0.460 s
   * - CuPy
     - 8.930 ± 0.406 s
   * - JAX (GPU)
     - 3.127 ± 0.524 s

The above table is based on results generated by these scripts:

- `readme_for_test.ipynb <./notebooks/readme_for_test.ipynb>`_
- `test_emtab.m <./matlab/test-generators/test_emtab.m>`_



Known Limitations
-----------------

- DNF simplification is not implemented in JAX because it involves extensive slicing,
  and JAX slicing always creates a copy instead of an array view,
  making it noticeably slower than NumPy/CuPy in practice.
  Furthermore, dynamic slicing is not supported under JIT compilation,
  requiring significant code modifications.

- In some cases, the ``c_th`` and ``d_k_th`` matrices (thresholded :math:`\tilde{C}` and :math:`\tilde{D_k}`?)
  converge to all True / 1, causing the "remove ..A&~A.. conjunction" step in
  DNF simplification to reduce the learned DNF to an empty DNF.
  I'm not sure whether this is a bug or intended behavior.
