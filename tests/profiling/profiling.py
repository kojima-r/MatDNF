#!/usr/bin/env python
"""Basic profiling for mat_dnf."""

import numpy as np

from mat_dnf.numpy.initializers import Default
from mat_dnf.numpy.models import train_mat_dnf
from mat_dnf.numpy.optimizers import Adam


def main():
    """Single row profiling."""
    dataset_1 = np.loadtxt("../../data/E-MTAB-1908/01_T.bin.csv", delimiter=",").astype(
        np.int64
    )
    row = 0
    I1 = dataset_1[:, :-1]
    I2 = dataset_1[row][1:]

    Er_max = 0
    max_itr = 10
    max_try = 2
    n = I1.shape[0]
    h = I1.shape[1]

    initializer = Default(h=h, n=n, aa=4)
    c_optimizer = Adam()
    d_k_optimizer = Adam()

    _, _, _, _ = train_mat_dnf(
        fold=0,
        i_in=I1,
        i_out=I2,
        h=h,
        er_max=Er_max,
        max_itr=max_itr,
        max_try=max_try,
        c_optim=c_optimizer,
        d_k_optim=d_k_optimizer,
        initializer=initializer,
        use_perturbation=False,
    )


if __name__ == "__main__":
    main()
