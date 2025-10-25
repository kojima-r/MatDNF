"""Profile random DNF."""
#!/usr/bin/env python

import time

import jax
from jax import numpy as jnp
from jaxtyping import Scalar

from mat_dnf.jax.generators import all_bit_seq, gen_dnf

# TODO: Implement for JAX
# from mat_dnf.jax.losses import acc_classi, acc_dnf, logi_conseq, logi_equiv
from mat_dnf.jax.models import (
    MatDNF,
    eval_dnf,
    train_mat_dnf,
    # simp_dnf,  TODO: Implement for JAX
)

jax.config.update("jax_enable_x64", True)

n = 5
alpha = 0.1
max_try = 20
max_itr = 500
h = 1000
Er_max = 0
dr = 0.5
i_max = 10

dtype = jnp.dtype(jnp.int64)
key = jax.random.key(42)

i0 = all_bit_seq(n, dtype=dtype)
l = 2**n
key, subkey = jax.random.split(key)
i0 = i0[:, jax.random.permutation(subkey, l)]

arr_acc_classi: list[Scalar] = []
arr_acc_dnf: list[Scalar] = []
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
    key, i1_key, gen_key, model_key, train_key = jax.random.split(key, num=5)
    i1 = i0[:, jax.random.permutation(i1_key, l)]
    d0, c0 = gen_dnf(
        key=gen_key,
        n=n,
        h_gen=h_gen,
        d_size=d_size,
        c_max=c_max,
        dtype=dtype,
    )
    i2_k = eval_dnf(d0, c0, i1)

    l2 = i1.shape[1]
    x = jnp.floor(l2 * dr).astype(jnp.int64)
    i1_dr = i1[:, :x]
    i1_test = i1[:, x:]

    i2_k_dr = i2_k[:x]
    i2_k_test = i2_k[x:]

    arr_test_size.append(i1_test.shape[1])

    model = MatDNF.create_random(key=model_key, h=h, n=n, aa=4)

    s = time.monotonic()
    model, v_k_th, learned_dnf = train_mat_dnf(
        model=model,
        key=train_key,
        fold=i,
        i_in=i1_dr,
        i_out=i2_k_dr,
        er_max=Er_max,
        alpha=0.1,
        max_itr=max_itr,
        max_try=max_try,
        use_perturbation=True,
    )
    e = time.monotonic()
    elapsed_time = e - s
    arr_time.append(elapsed_time)

    # TODO: Write JAX version
    # arr_acc_classi.append(acc_classi(d_k, v_k_th, i1, i2_k, l2, c))

    # TODO: Temporary casting
    # TODO: Write JAX version
    # learned_dnf_s = simp_dnf(learned_dnf.astype(np.int64))
    # n2 = learned_dnf_s.shape[1]
    # learned_dnf_n = (
    #     learned_dnf_s[:, : n2 // 2].astype(np.int64)
    #     - learned_dnf_s[:, n2 // 2 :].astype(np.int64)
    # ).astype(np.bool_)

    # TODO: Write JAX version
    # arr_acc_dnf.append(acc_dnf(learned_dnf_s, i1, i2_k, l2))

    # TODO: Write JAX version
    # cnsq, _ = logi_conseq(learned_dnf_s, i2_k, i1)
    # eqv, _ = logi_equiv(learned_dnf_s, i2_k, i1)
    cnsq = True
    eqv = True
    arr_conseq.append(cnsq)
    arr_equiv.append(eqv)

print(f"dr = {dr}, average over {i_max} trials")
print(
    f"exact_acc_DNF   exact_acc_classi   |test_size|/|I1|   conseq    equiv    time(s)"
)
print(
    f"{jnp.array(arr_acc_dnf).mean():0.3f}           {jnp.array(arr_acc_classi).mean():0.3f}              "
    f"{jnp.array(arr_test_size).mean():0.1f}/{i1.shape[1]}            {jnp.array(arr_conseq).mean():0.3f}"  # type: ignore
    f"     {jnp.array(arr_equiv).mean():0.3f}    {jnp.array(arr_time).mean():0.6f}"
)
