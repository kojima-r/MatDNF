"""Optimizers for updating model parameters."""

from abc import ABC, abstractmethod
from types import ModuleType
from typing import final, override

import cupy as cp
import numpy as np
from numpy import floating
from numpy.typing import NBitBase, NDArray


class Optimizer(ABC):
    """Generic optimizer."""

    @abstractmethod
    def update[T: NBitBase](
        self,
        x: NDArray[floating[T]],
        ja_x: NDArray[floating[T]],
    ) -> NDArray[floating[T]]:
        """Update x given gradient ja_x."""


@final
class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        xp: ModuleType = np,
    ):
        """A variant of Adam optimizer, as implemented in the MATLAB code.

        A little different from the typical Adam with an additional "t" parameter.

        Args:
            alpha: Learning rate.
            beta_1: Exponential decay rates for first moment estimate.
            beta_2: Exponential decay rates for second moment estimate.
            epsilon: Small constant to avoid divide by zero.
            xp: Array module. CuPy or NumPy.
        """
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 1
        self.m: NDArray[floating] | None = None
        self.v: NDArray[floating] | None = None
        self.m_hat: NDArray[floating] = xp.array([])
        self.v_hat: NDArray[floating] = xp.array([])

    @override
    def update[T: NBitBase](
        self,
        x: NDArray[floating[T]],
        ja_x: NDArray[floating[T]],
    ) -> NDArray[floating[T]]:
        xp = cp.get_array_module(x, ja_x)

        _m = self.m if self.m is not None else xp.zeros_like(x)
        _v = self.v if self.v is not None else xp.zeros_like(x)

        self.m = self.beta_1 * _m + (1 - self.beta_1) * ja_x
        self.v = self.beta_2 * _v + (1 - self.beta_2) * (ja_x * ja_x)

        self.m_hat = self.m / (1 - xp.exp(self.t * xp.log(self.beta_1)))
        self.v_hat = self.v / (1 - xp.exp(self.t * xp.log(self.beta_2)))
        x = x - self.alpha * (self.m_hat / (xp.sqrt(self.v_hat) + self.epsilon))

        self.t += 1
        return x
