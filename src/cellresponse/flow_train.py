"""Conditional coupling flow on module-score residuals.

The target is the training-cell score minus that plate's reference mean, scaled by the
training residual standard deviation. Sampling adds the reference mean back. Test-treated
cells are not read. The flow is kept only if its validation top-1 utility beats the
shared shift; otherwise the shared shift stays the selected model.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import polars as pl
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import coupling_flow

from cellresponse.baselines.run import (
    _condition_vector,
    _contexts,
    _curves,
    _directions,
    _fingerprints,
    _gaussian_rows,
    _ids_by_condition,
    _matrix,
    _score_predictions,
)
from cellresponse.baselines.scores import ensure_scores
from cellresponse.data.report import md_table
from cellresponse.split import HELD_OUT_PLATE


def run(config: dict) -> None:
    prepared = Path(config["paths"]["prepared"]) / config["name"]
    seed = int(config["cohort"]["seed"])
    genes = pl.read_parquet(prepared / "genes.parquet")
    scores = ensure_scores(prepared, genes)
    splits = pl.read_parquet(prepared / "splits.parquet")
    joined = scores.join(splits, on="cell_id", how="inner")
    curves = _curves(joined)
    matrix, row_of = _matrix(joined)
    compounds = pl.read_parquet(prepared / "compounds.parquet")
    fingerprints = _fingerprints(compounds)
    projection = np.random.default_rng(seed).normal(size=(2048, 32)).astype(np.float32) / np.sqrt(
        2048
    )
    dev = json.loads(Path("queries/dev.json").read_text())
    treated = _ids_by_condition(
        splits.filter(
            (pl.col("protocol_a") == "validation")
            & ~pl.col("is_control")
            & (pl.col("plate_num") != HELD_OUT_PLATE)
        )
    )
    contexts = _contexts(dev, matrix, row_of, treated, fingerprints)
    train = joined.filter(
        (pl.col("protocol_a") == "train")
        & ~pl.col("is_control")
        & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    reference = joined.filter(
        (pl.col("control_role") == "reference") & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    features, targets = _gaussian_rows(train, reference, fingerprints, projection, seed)
    residual = targets - features[:, :4]
    scale = np.maximum(residual.std(axis=0), 1e-3).astype(np.float32)
    standardized = (residual / scale).astype(np.float32)
    print(f"[flow] training on {standardized.shape[0]:,} cells", flush=True)
    flow, nll = _train(standardized, features.astype(np.float32), seed)
    _numerical_check(flow, features[0].astype(np.float32))
    predicted = _predict(flow, contexts, projection, scale, seed)
    rows, dist_rows = _score_predictions(
        contexts, {"flow": predicted}, curves, _directions(seed), seed
    )
    table = pl.DataFrame(rows)
    summary = table.group_by("goal_id").agg(
        top1=pl.col("top1_observed").mean(),
        spearman=pl.col("spearman").drop_nans().mean(),
        nothing=pl.col("picked_nothing").mean(),
    )
    overall = float(table["top1_observed"].mean())
    baseline = pl.read_csv("reports/tables/baseline_queries.csv")
    shared = float(baseline.filter(pl.col("model") == "shared_shift")["top1_observed"].mean())
    selected = "flow" if overall > shared else "shared_shift"
    report = "\n".join(
        [
            "# Flow",
            "",
            "A 4-layer rational-quadratic coupling flow models the residual of the four module",
            "scores around the reference mean. It is conditioned on the reference mean and spread,",
            "a 32-dimensional projection of the Morgan fingerprint, and the dose.",
            "",
            f"Holdout NLL {nll:.3f}. Mean top-1 observed utility {overall:.4f}.",
            f"The shared shift's validation top-1 is {shared:.4f}. Selected model: `{selected}`.",
            "",
            md_table(summary.sort("goal_id")),
            "",
            "Inverse consistency on one condition was finite, with absolute error below 1e-3.",
            "",
        ]
    )
    Path("reports/flow.md").write_text(report)
    print(report, flush=True)
    print(f"[flow] selected {selected}", flush=True)
    _ = dist_rows


def _train(standardized: np.ndarray, features: np.ndarray, seed: int, epochs: int = 6):
    key = jr.key(seed)
    flow = coupling_flow(
        key,
        base_dist=Normal(jnp.zeros(4), jnp.ones(4)),
        transformer=RationalQuadraticSpline(knots=8, interval=6),
        cond_dim=features.shape[1],
        flow_layers=4,
        nn_width=128,
        nn_depth=2,
    )
    y = jnp.asarray(standardized)
    h = jnp.asarray(features)
    hold = jr.bernoulli(jr.key(seed + 1), 0.1, (y.shape[0],))
    diff, static = eqx.partition(flow, eqx.is_inexact_array)
    optimizer = optax.adamw(3e-4, weight_decay=1e-4)
    opt_state = optimizer.init(diff)

    @eqx.filter_jit
    def step(diff, opt_state, score, condition):
        def loss(diff):
            model = eqx.combine(diff, static)
            return -jnp.mean(model.log_prob(score, condition))

        value, grads = eqx.filter_value_and_grad(loss)(diff)
        updates, opt_state = optimizer.update(grads, opt_state, diff)
        diff = eqx.apply_updates(diff, updates)
        return diff, opt_state, value

    best, best_loss = eqx.combine(diff, static), float("inf")
    n = int(y.shape[0])
    batch = 512
    for epoch in range(epochs):
        key, perm_key = jr.split(key)
        order = np.asarray(jr.permutation(jr.fold_in(perm_key, epoch), n))
        losses = []
        for start in range(0, n - batch + 1, batch):
            idx = order[start : start + batch]
            diff, opt_state, value = step(diff, opt_state, y[idx], h[idx])
            losses.append(float(value))
        flow = eqx.combine(diff, static)
        current = float(-jnp.mean(flow.log_prob(y[hold], h[hold])))
        print(
            f"  flow epoch {epoch + 1}: train {np.mean(losses):.3f} holdout {current:.3f}",
            flush=True,
        )
        if current < best_loss:
            best_loss = current
            best = flow
    return best, best_loss


def _numerical_check(flow, condition: np.ndarray) -> None:
    h = jnp.asarray(condition)
    base = jnp.zeros(4)
    transformed, logdet = flow.bijection.transform_and_log_det(base, condition=h)
    recovered, inv_logdet = flow.bijection.inverse_and_log_det(transformed, condition=h)
    error = float(jnp.max(jnp.abs(recovered - base)))
    if not np.isfinite(error) or error > 1e-3 or not np.isfinite(float(logdet + inv_logdet)):
        raise RuntimeError(f"flow inverse check failed: error {error}, logdet {float(logdet)}")
    print(f"[flow] inverse error {error:.2e}", flush=True)


def _predict(flow, contexts, projection, scale, seed) -> dict:
    conditions, keys = [], []
    for context in contexts:
        ref_mean = context["reference"].mean(axis=0)
        ref_sd = context["reference"].std(axis=0)
        row = {f"ref_{name}": ref_mean[i] for i, name in enumerate(_names())}
        row.update({f"sd_{name}": ref_sd[i] for i, name in enumerate(_names())})
        for candidate in context["panel"]:
            row["dose_uM"] = candidate["dose_uM"]
            conditions.append(_condition_vector(row, candidate["fingerprint"], projection))
            keys.append((context["cell_line"], context["plate_num"], candidate["compound_key"]))
    stacked = jnp.asarray(np.asarray(conditions, np.float32))

    def draw(condition, key):
        residual = flow.sample(key, (256,), condition=condition)
        return residual * jnp.asarray(scale) + condition[:4]

    samples = jax.vmap(draw)(stacked, jr.split(jr.key(seed), stacked.shape[0]))
    return {key: np.asarray(sample) for key, sample in zip(keys, samples, strict=True)}


def _names() -> list[str]:
    from cellresponse.baselines.scores import score_names

    return score_names()
