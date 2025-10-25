"""Optimizers for updating model parameters."""

from typing import NamedTuple

import jax
import optax
from jax import numpy as jnp
from jaxtyping import PyTree
from optax import tree_utils as otu


class AdamState(NamedTuple):
    """State for the MATLAB code's variant of Adam optimizer."""

    t: jax.Array
    m: PyTree
    v: PyTree
    m_hat: PyTree
    v_hat: PyTree


def adam(
    alpha: float = 0.1,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> optax.GradientTransformation:
    """A variant of Adam optimizer, as implemented in the MATLAB code.

    A little different from the typical Adam with an additional "t" parameter.
    """

    def init_fn(params: PyTree) -> AdamState:
        m = otu.tree_zeros_like(params)
        v = otu.tree_zeros_like(params)
        m_hat = otu.tree_zeros_like(params)
        v_hat = otu.tree_zeros_like(params)
        return AdamState(
            t=jnp.array(1),
            m=m,
            v=v,
            m_hat=m_hat,
            v_hat=v_hat,
        )

    def update_fn(
        updates: PyTree, state: AdamState, params: PyTree | None = None
    ) -> tuple[PyTree, AdamState]:
        del params

        m = jax.tree.map(lambda g, t: beta_1 * t + (1 - beta_1) * g, updates, state.m)
        v = jax.tree.map(
            lambda g, t: beta_2 * t + (1 - beta_2) * (g * g), updates, state.v
        )
        m_hat = jax.tree.map(lambda t: t / (1 - jnp.exp(state.t * jnp.log(beta_1))), m)
        v_hat = jax.tree.map(lambda t: t / (1 - jnp.exp(state.t * jnp.log(beta_2))), v)

        updates = jax.tree.map(
            lambda m_hat_, v_hat_: -alpha * (m_hat_ / (jnp.sqrt(v_hat_) + epsilon)),
            m_hat,
            v_hat,
        )

        return updates, AdamState(t=state.t + 1, m=m, v=v, m_hat=m_hat, v_hat=v_hat)

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def normalize_leaf() -> optax.GradientTransformation:
    """Normalize the gradient on per-leaf basis.

    This is necessary to match MATLAB implementation,
    which normalizes C and D_k separately.
    """

    def init_fn(params: PyTree):
        del params
        return optax.EmptyState()

    def update_fn(
        updates: optax.EmptyState, state: PyTree, params: PyTree | None = None
    ):
        del params
        updates = jax.tree.map(lambda g: g / jnp.sqrt(jnp.sum(g**2)), updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore
