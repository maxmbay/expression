"""Conditional Gaussian on the four module scores. Diagonal variance plus a rank-3 factor."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

DIM = 4
RANK = 3


class ConditionalGaussian(eqx.Module):
    net: eqx.nn.MLP

    def factors(self, condition: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        raw = self.net(condition)
        mean = raw[:DIM]
        log_std = jnp.clip(raw[DIM : 2 * DIM], -5.0, 2.0)
        factor = 0.1 * raw[2 * DIM :].reshape(DIM, RANK)
        return mean, log_std, factor

    def log_prob(self, score: jax.Array, condition: jax.Array) -> jax.Array:
        mean, log_std, factor = self.factors(condition)
        covariance = jnp.diag(jnp.exp(2 * log_std)) + factor @ factor.T
        covariance = covariance + jnp.eye(DIM) * 1e-3
        residual = score - mean
        precision = jnp.linalg.solve(covariance, residual)
        _sign, logdet = jnp.linalg.slogdet(covariance)
        return -0.5 * (residual @ precision + logdet + DIM * jnp.log(2 * jnp.pi))

    def sample(self, condition: jax.Array, key: jax.Array, n: int) -> jax.Array:
        mean, log_std, factor = self.factors(condition)
        noise_key, factor_key = jr.split(key)
        diagonal = jr.normal(noise_key, (n, DIM)) * jnp.exp(log_std)
        low_rank = jr.normal(factor_key, (n, RANK)) @ factor.T
        return mean + diagonal + low_rank


def train_gaussian(
    conditions: jax.Array,
    scores: jax.Array,
    seed: int,
    epochs: int = 8,
    batch_size: int = 1024,
) -> ConditionalGaussian:
    """Fit on training cells. Returns the weights with the best holdout negative log-likelihood."""
    key = jr.PRNGKey(seed)
    key, init_key, split_key = jr.split(key, 3)
    model = ConditionalGaussian(
        eqx.nn.MLP(conditions.shape[1], DIM * 2 + DIM * RANK, 64, 2, key=init_key)
    )
    holdout = jr.bernoulli(split_key, 0.1, (scores.shape[0],))
    train_h, train_y = conditions[~holdout], scores[~holdout]
    valid_h, valid_y = conditions[holdout], scores[holdout]
    optimizer = optax.adamw(3e-4, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, score, condition):
        def loss(model):
            batched = jax.vmap(model.log_prob)(score, condition)
            return -jnp.mean(batched)

        value, grads = eqx.filter_value_and_grad(loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, value

    @eqx.filter_jit
    def holdout_loss(model, score, condition):
        return -jnp.mean(jax.vmap(model.log_prob)(score, condition))

    best_model, best_loss = model, float("inf")
    n = int(train_y.shape[0])
    for epoch in range(epochs):
        key, perm_key = jr.split(key)
        order = jr.permutation(perm_key, n)
        losses = []
        for start in range(0, n - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            model, opt_state, value = step(model, opt_state, train_y[batch], train_h[batch])
            losses.append(float(value))
        current = (
            float(holdout_loss(model, valid_y, valid_h)) if int(valid_y.shape[0]) else float("inf")
        )
        train_mean = float(np.mean(losses)) if losses else float("nan")
        print(
            f"  gaussian epoch {epoch + 1}: train {train_mean:.3f} holdout {current:.3f}",
            flush=True,
        )
        if current < best_loss:
            best_loss = current
            best_model = model
    return best_model


def sample_gaussian(
    model: ConditionalGaussian, condition: jax.Array, seed: int, n: int
) -> jax.Array:
    return model.sample(condition, jr.PRNGKey(seed), n)
