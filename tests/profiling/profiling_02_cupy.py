"""Profile random DNF."""
#!/usr/bin/env python

import time

import cupy as cp
import numpy as np
from numpy import random

from mat_dnf.numpy.generators import all_bit_seq, gen_dnf, random_c, random_d_k
from mat_dnf.numpy.losses import acc_classi, acc_dnf, logi_conseq, logi_equiv
from mat_dnf.numpy.models import (
    eval_dnf,
    simp_dnf,
    train_mat_dnf,
)

n = 5
alpha = 0.1
max_try = 20
max_itr = 500
h = 1000
Er_max = 0
dr = 0.5
i_max = 10

dtype = cp.dtype(cp.int64)
rng = random.default_rng()

i0 = all_bit_seq(n, dtype=dtype)
l = 2**n
i0 = i0[:, rng.permutation(l)]
i0 = cp.asarray(i0)
xp = cp.get_array_module(i0)

arr_acc_classi: list[np.float64] = []
arr_acc_dnf: list[np.float64] = []
arr_test_size: list[int] = []
arr_time: list[float] = []
arr_conseq: list[bool] = []
arr_equiv: list[bool] = []

acc_unk = -1

# Repeat learning i_max times and measure exact_acc_DNF, exact_acc_class, prob(conseq), prob(equive)
for i in range(i_max):
    # Choose learning data
    # Random DNF
    d_size = 3
    h_gen = 10
    c_max = 5
    # No noise
    i1 = i0[:, rng.permutation(l)]
    d0, c0 = gen_dnf(
        n,
        h_gen,
        d_size,
        c_max,
        rng,
        dtype=dtype,
    )

    # TODO: Temporary adapter
    d0 = cp.asarray(d0)
    c0 = cp.asarray(c0)

    i2_k = eval_dnf(d0, c0, i1)

    l2 = i1.shape[1]
    x = np.floor(l2 * dr).astype(np.int64)
    i1_dr = i1[:, :x]
    i1_test = i1[:, x:]

    i2_k_dr = i2_k[:x]
    i2_k_test = i2_k[x:]

    arr_test_size.append(i1_test.shape[1])

    c_init = random_c(rng, h=h, n=n)
    d_k_init = random_d_k(rng, h=h, n=n)

    # TODO: Temporary adapter
    i1_dr = cp.asarray(i1_dr)
    i2_k_dr = cp.asarray(i2_k_dr)
    c_init = cp.asarray(c_init)
    d_k_init = cp.asarray(d_k_init)
    i2_k = cp.asarray(i2_k)
    i1 = cp.asarray(i1)

    s = time.monotonic()
    c, d_k, v_k_th, learned_dnf = train_mat_dnf(
        rng=rng,
        c=c_init,
        d_k=d_k_init,
        i_in=i1_dr,
        i_out=i2_k_dr,
        er_max=Er_max,
        alpha=0.1,
        max_itr=max_itr,
        max_try=max_try,
        fold=i,
        use_perturbation=True,
    )
    e = time.monotonic()
    elapsed_time = e - s
    arr_time.append(elapsed_time)

    arr_acc_classi.append(acc_classi(d_k, v_k_th, i1, i2_k, l2, c))
    # TODO: Temporary casting
    learned_dnf_s = simp_dnf(learned_dnf.astype(np.int64))
    n2 = learned_dnf_s.shape[1]
    learned_dnf_n = (
        learned_dnf_s[:, : n2 // 2].astype(np.int64)
        - learned_dnf_s[:, n2 // 2 :].astype(np.int64)
    ).astype(np.bool_)

    arr_acc_dnf.append(acc_dnf(learned_dnf_s, i1, i2_k, l2))

    cnsq, _ = logi_conseq(learned_dnf_s, i2_k, i1)
    eqv, _ = logi_equiv(learned_dnf_s, i2_k, i1)
    arr_conseq.append(cnsq)
    arr_equiv.append(eqv)

print(f"dr = {dr}, average over {i_max} trials")
print(
    f"exact_acc_DNF   exact_acc_classi   |test_size|/|I1|   conseq    equiv    time(s)"
)
print(
    f"{xp.array(arr_acc_dnf).mean():0.3f}           {xp.array(arr_acc_classi).mean():0.3f}              "
    f"{xp.array(arr_test_size).mean():0.1f}/{i1.shape[1]}            {xp.array(arr_conseq).mean():0.3f}"  # type: ignore
    f"     {xp.array(arr_equiv).mean():0.3f}    {xp.array(arr_time).mean():0.6f}"
)
