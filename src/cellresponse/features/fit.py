"""Fit the version-0.1 representation on a fit mask, then test whether it keeps goal scores.

HVGs are the 2,000 genes whose raw-count variance sits furthest above a quadratic
mean-variance trend. Moments come from a balanced subsample: at most 32 cells from
each training condition, ranked by a fixed 64-bit mix of the cell id, so a large
condition cannot dominate. PCA is fit on a second balanced subsample of at most
200,000 cells. Components are centered. PC scales are the training standard
deviations, floored at 1e-4.

The gate scores each goal on the member genes that landed in the HVG set. Raw and
PCA-reconstructed event fractions are compared on protocol-A validation conditions.
Thresholds are quantiles of the protocol-A training reference-vehicle scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cellresponse.goals import goals, passes_event, vocabulary_columns
from cellresponse.split import HELD_OUT_PLATE

N_HVG = 2_000
N_COMPONENTS = 64
HVG_PER_CONDITION = 32
PCA_MAX_CELLS = 200_000
PC_SCALE_FLOOR = 1e-4
MIN_GENE_MEAN = 1e-3
MIN_GATE_CELLS = 32
MIN_RANKED_COMPOUNDS = 10
MIN_GENE_COVERAGE = 0.5
GATE_EVENT_TOLERANCE = 0.05
GATE_RANK_CORRELATION = 0.8

HVG_METHOD = (
    "Seurat-v3-style residual: on a balanced subsample of at most 32 cells per "
    "training condition, compute the mean and variance of raw counts, fit a degree-2 "
    "polynomial of log10(variance) on log10(mean) for genes with mean at least 0.001, "
    "and keep the 2,000 genes with the largest residual. Ties break toward the lower column."
)


def mix64(cell_ids: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic rank key. Not used for the split itself."""
    z = cell_ids.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def choose_hvg(
    count_sum: np.ndarray, count_sumsq: np.ndarray, n_cells: int, n_hvg: int = N_HVG
) -> np.ndarray:
    """Return vocabulary column indices of the selected genes, sorted by column."""
    mean = count_sum / n_cells
    var = np.maximum(count_sumsq / n_cells - mean**2, 0.0)
    eligible = (mean >= MIN_GENE_MEAN) & (var > 0)
    if int(eligible.sum()) < n_hvg:
        raise RuntimeError(
            f"only {int(eligible.sum())} genes have enough counts to rank {n_hvg} HVGs"
        )
    log_mean = np.log10(mean[eligible])
    log_var = np.log10(var[eligible])
    residual = np.full(mean.shape, -np.inf)
    # A quadratic trend needs spread in the mean. With none, rank by raw variance.
    if np.unique(np.round(log_mean, 5)).size < 3:
        fitted = np.zeros_like(log_var)
    else:
        fitted = np.polyval(np.polyfit(log_mean, log_var, deg=2), log_mean)
    residual[eligible] = log_var - fitted
    order = np.lexsort((np.arange(residual.size), -residual))
    return np.sort(order[:n_hvg].astype(np.int32))


def balanced_subsample(
    frame: pl.DataFrame, seed: int, per_condition: int, max_cells: int
) -> pl.DataFrame:
    ranked = frame.with_columns(rank=pl.Series("rank", mix64(frame["cell_id"].to_numpy(), seed)))
    chosen = ranked.sort("rank").group_by("condition", maintain_order=True).head(per_condition)
    if chosen.height > max_cells:
        chosen = chosen.sort("rank").head(max_cells)
    return chosen.drop("rank")


def fit_representation(
    prepared: Path, splits: pl.DataFrame, genes: pl.DataFrame, fit_column: str, seed: int, out: Path
) -> dict:
    """Fit HVGs and PCA on `fit_column` and write a compressed npz. Held-out cells stay out."""
    frame = _fit_frame(prepared, splits, fit_column)
    print(
        f"[{fit_column}] {frame.height:,} fitting cells, "
        f"{frame['condition'].n_unique():,} conditions",
        flush=True,
    )
    lut, known, n_genes = _token_lut(genes)
    hvg_sample = balanced_subsample(frame, seed, HVG_PER_CONDITION, PCA_MAX_CELLS)
    count_sum, count_sumsq, n_hvg_cells = _count_moments(prepared, hvg_sample, n_genes, lut, known)
    hvg_columns = choose_hvg(count_sum, count_sumsq, n_hvg_cells)
    per_condition = max(1, min(256, PCA_MAX_CELLS // frame["condition"].n_unique()))
    pca_sample = balanced_subsample(frame, seed + 1, per_condition, PCA_MAX_CELLS)
    print(
        f"[{fit_column}] PCA on {pca_sample.height:,} cells x {hvg_columns.size} genes", flush=True
    )
    matrix = _load_matrix(
        prepared, pca_sample, _hvg_positions(genes, hvg_columns, lut.shape[0]), known
    )
    from sklearn.decomposition import PCA

    pca = PCA(
        n_components=N_COMPONENTS, svd_solver="randomized", random_state=seed, iterated_power=4
    )
    latent = pca.fit_transform(matrix)
    pc_mean = latent.mean(axis=0).astype(np.float32)
    pc_scale = np.maximum(latent.std(axis=0), PC_SCALE_FLOOR).astype(np.float32)
    token_of_column = np.full(n_genes, -1, np.int32)
    token_of_column[genes["column"].to_numpy()] = genes["token_id"].to_numpy()
    np.savez_compressed(
        out,
        hvg_columns=hvg_columns,
        hvg_token_ids=token_of_column[hvg_columns],
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pc_mean=pc_mean,
        pc_scale=pc_scale,
    )
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"[{fit_column}] PCA explains {explained:.3f} of HVG variance -> {out}", flush=True)
    return {
        "fit_column": fit_column,
        "path": str(out),
        "n_fit_cells": frame.height,
        "n_conditions": frame["condition"].n_unique(),
        "n_hvg_cells": n_hvg_cells,
        "n_pca_cells": int(pca_sample.height),
        "hvg_per_condition": HVG_PER_CONDITION,
        "pca_per_condition": per_condition,
        "variance_explained": round(explained, 4),
        "pc_scale_floor": PC_SCALE_FLOOR,
        "pc_scale_min": float(pc_scale.min()),
        "method": HVG_METHOD,
    }


def representation_gate(
    prepared: Path, splits: pl.DataFrame, genes: pl.DataFrame, fit: dict
) -> dict:
    """Compare raw and reconstructed validation event fractions. Thresholds are training vehicle."""
    bundle = np.load(fit["path"])
    hvg_columns = bundle["hvg_columns"]
    components = bundle["pca_components"].astype(np.float64)
    pca_mean = bundle["pca_mean"].astype(np.float64)
    lut, known, _ = _token_lut(genes)
    positions = _hvg_positions(genes, hvg_columns, lut.shape[0])
    column_pos = {int(column): index for index, column in enumerate(hvg_columns)}
    membership = []
    for goal in goals():
        columns, missing = vocabulary_columns(goal, genes)
        kept = [int(column) for column in columns if int(column) in column_pos]
        membership.append(
            (
                goal,
                np.asarray(kept and [column_pos[column] for column in kept], dtype=np.int32),
                missing,
                len(columns),
            )
        )
    scored = _gate_frame(prepared, splits)
    print(f"[gate] scoring {scored.height:,} cells", flush=True)
    threshold_scores, validation_scores, validation_meta = _stream_scores(
        prepared, scored, positions, known, components, pca_mean, membership
    )
    meta = pl.concat(validation_meta)
    results = []
    for goal, cols, missing, n_vocabulary in membership:
        if cols.size == 0:
            results.append(_unrepresented(goal, n_vocabulary, missing))
            continue
        raw = np.concatenate(validation_scores[goal.goal_id][0])
        recon = np.concatenate(validation_scores[goal.goal_id][1])
        baseline = np.concatenate(threshold_scores[goal.goal_id])
        threshold = float(np.quantile(baseline, goal.quantile))
        table = meta.with_columns(
            raw_event=pl.Series(passes_event(raw, threshold, goal.direction)),
            recon_event=pl.Series(passes_event(recon, threshold, goal.direction)),
        )
        summary = _goal_summary(
            goal, table, len(cols), n_vocabulary, missing, threshold, int(baseline.size)
        )
        results.append(summary)
        print(
            f"[gate] {goal.goal_id}: median |Δ| {summary['median_abs_delta']:.4f}, "
            f"spearman {summary['median_spearman']}, coverage {summary['coverage']:.2f}",
            flush=True,
        )
    passed = all(item["passes"] for item in results)
    return {
        "decision": "pca64" if passed else "direct_module_scores",
        "goals": results,
        "n_validation_cells": meta.height,
    }


def _unrepresented(goal, n_vocabulary: int, missing: list[str]) -> dict:
    return {
        "goal_id": goal.goal_id,
        "role": goal.role,
        "direction": goal.direction,
        "quantile": goal.quantile,
        "threshold": None,
        "n_threshold_cells": 0,
        "n_vocabulary_genes": n_vocabulary,
        "n_hvg_genes": 0,
        "n_missing_genes": len(missing),
        "missing_genes": missing,
        "coverage": 0.0,
        "n_conditions": 0,
        "median_abs_delta": float("nan"),
        "p90_abs_delta": float("nan"),
        "frac_abs_delta_gt_0.05": float("nan"),
        "median_spearman": float("nan"),
        "n_ranked_blocks": 0,
        "n_blocks_no_variation": 0,
        "passes": False,
    }


def _goal_summary(goal, table, n_kept, n_vocabulary, missing, threshold, n_threshold) -> dict:
    from scipy.stats import spearmanr

    counted = (
        table.group_by("cell_line", "plate_num", "compound_key")
        .agg(
            n=pl.len(),
            raw_event=pl.col("raw_event").mean(),
            recon_event=pl.col("recon_event").mean(),
        )
        .filter(pl.col("n") >= MIN_GATE_CELLS)
    )
    delta = (counted["raw_event"] - counted["recon_event"]).abs()
    rhos = []
    skipped = 0
    for _, block in counted.group_by("cell_line", "plate_num"):
        if block.height < MIN_RANKED_COMPOUNDS:
            continue
        raw = block["raw_event"].to_numpy()
        recon = block["recon_event"].to_numpy()
        if np.unique(raw).size < 2 or np.unique(recon).size < 2:
            skipped += 1
            continue
        rhos.append(float(spearmanr(raw, recon).statistic))
    median_rho = float(np.median(rhos)) if rhos else float("nan")
    coverage = n_kept / n_vocabulary if n_vocabulary else 0.0
    median_delta = float(delta.median()) if counted.height else float("nan")
    passes = (
        coverage >= MIN_GENE_COVERAGE
        and counted.height > 0
        and median_delta <= GATE_EVENT_TOLERANCE
        and len(rhos) >= 5
        and median_rho >= GATE_RANK_CORRELATION
    )
    return {
        "goal_id": goal.goal_id,
        "role": goal.role,
        "direction": goal.direction,
        "quantile": goal.quantile,
        "threshold": threshold,
        "n_threshold_cells": n_threshold,
        "n_vocabulary_genes": n_vocabulary,
        "n_hvg_genes": n_kept,
        "n_missing_genes": len(missing),
        "missing_genes": missing,
        "coverage": coverage,
        "n_conditions": counted.height,
        "median_abs_delta": median_delta,
        "p90_abs_delta": float(delta.quantile(0.9)) if counted.height else float("nan"),
        "frac_abs_delta_gt_0.05": float((delta > GATE_EVENT_TOLERANCE).mean())
        if counted.height
        else float("nan"),
        "median_spearman": median_rho,
        "n_ranked_blocks": len(rhos),
        "n_blocks_no_variation": skipped,
        "passes": bool(passes),
    }


def _fit_frame(prepared: Path, splits: pl.DataFrame, fit_column: str) -> pl.DataFrame:
    cells = pl.read_parquet(
        prepared / "cells.parquet", columns=["cell_id", "library_size", "fragment"]
    )
    frame = splits.filter(pl.col(fit_column)).join(cells, on="cell_id", how="left")
    if frame["library_size"].null_count() or frame.filter(pl.col("library_size") <= 0).height:
        raise RuntimeError("fit cells are missing a positive library size")
    return frame.with_columns(
        scale=pl.lit(10_000.0) / pl.col("library_size"),
        condition=(
            pl.when(pl.col("is_control"))
            .then(
                pl.concat_str(
                    [pl.col("cell_line"), pl.lit("vehicle"), pl.col("plate_num")], separator="|"
                )
            )
            .otherwise(
                pl.concat_str(
                    [
                        pl.col("cell_line"),
                        pl.col("compound_key"),
                        pl.col("dose_uM"),
                        pl.col("plate_num"),
                    ],
                    separator="|",
                )
            )
        ),
    )


def _gate_frame(prepared: Path, splits: pl.DataFrame) -> pl.DataFrame:
    cells = pl.read_parquet(
        prepared / "cells.parquet", columns=["cell_id", "library_size", "fragment"]
    )
    threshold = splits.filter(
        (pl.col("control_role") == "reference") & pl.col("fit_a")
    ).with_columns(gate_role=pl.lit("threshold"))
    validation = splits.filter(
        (pl.col("protocol_a") == "validation")
        & ~pl.col("is_control")
        & (pl.col("plate_num") != HELD_OUT_PLATE)
    ).with_columns(gate_role=pl.lit("validation"))
    columns = ["cell_id", "gate_role", "cell_line", "plate_num", "compound_key"]
    return (
        pl.concat([threshold.select(columns), validation.select(columns)], how="vertical")
        .join(cells, on="cell_id", how="left")
        .with_columns(scale=pl.lit(10_000.0) / pl.col("library_size"))
    )


def _token_lut(genes: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    token = genes["token_id"].to_numpy()
    column = genes["column"].to_numpy().astype(np.int32)
    lut = np.full(int(token.max()) + 1, -1, np.int32)
    lut[token] = column
    return lut, lut >= 0, int(column.max()) + 1


def _hvg_positions(genes: pl.DataFrame, hvg_columns: np.ndarray, n_tokens: int) -> np.ndarray:
    token_of_column = np.full(int(genes["column"].max()) + 1, -1, np.int32)
    token_of_column[genes["column"].to_numpy()] = genes["token_id"].to_numpy()
    positions = np.full(n_tokens, -1, np.int32)
    positions[token_of_column[hvg_columns]] = np.arange(hvg_columns.size, dtype=np.int32)
    return positions


def _count_moments(prepared, sample, n_genes, lut, known) -> tuple[np.ndarray, np.ndarray, int]:
    count_sum = np.zeros(n_genes, np.float64)
    count_sumsq = np.zeros(n_genes, np.float64)
    seen = 0
    fragments = list(_by_fragment(sample))
    for index, (fragment, part) in enumerate(fragments, start=1):
        long = _explode(prepared / "fragments" / fragment, part)
        seen += part.height
        if long.is_empty():
            continue
        genes_idx = long["genes"].to_numpy()
        if np.any(~known[genes_idx]):
            raise RuntimeError(f"unknown gene token in {fragment}")
        counts = long["counts"].to_numpy().astype(np.float64, copy=False)
        columns = lut[genes_idx]
        np.add.at(count_sum, columns, counts)
        np.add.at(count_sumsq, columns, counts * counts)
        if index % 400 == 0:
            print(f"  hvg moments {index}/{len(fragments)} fragments", flush=True)
    if seen != sample.height:
        raise RuntimeError(f"saw {seen} cells of {sample.height} in the HVG subsample")
    return count_sum, count_sumsq, seen


def _load_matrix(prepared, sample, positions, known) -> np.ndarray:
    matrix = np.zeros((sample.height, int((positions >= 0).sum())), np.float32)
    cursor = 0
    fragments = list(_by_fragment(sample))
    for index, (fragment, part) in enumerate(fragments, start=1):
        block = _fill_block(prepared / "fragments" / fragment, part, positions, known)
        matrix[cursor : cursor + block.shape[0]] = block
        cursor += block.shape[0]
        if index % 400 == 0:
            print(f"  pca matrix {index}/{len(fragments)} fragments", flush=True)
    if cursor != sample.height:
        raise RuntimeError("PCA matrix rows do not match the subsample")
    return matrix


def _stream_scores(prepared, scored, positions, known, components, pca_mean, membership):
    threshold = {goal.goal_id: [] for goal, *_ in membership}
    validation = {goal.goal_id: ([], []) for goal, *_ in membership}
    meta = []
    fragments = list(_by_fragment(scored))
    for index, (fragment, part) in enumerate(fragments, start=1):
        block = _fill_block(prepared / "fragments" / fragment, part, positions, known).astype(
            np.float64
        )
        recon = (block - pca_mean) @ components.T @ components + pca_mean
        is_threshold = part["gate_role"].to_numpy() == "threshold"
        for goal, cols, _, _ in membership:
            if cols.size == 0:
                raw = np.full(block.shape[0], np.nan)
                hat = raw.copy()
            else:
                raw = block[:, cols].mean(axis=1)
                hat = recon[:, cols].mean(axis=1)
            if np.any(is_threshold):
                threshold[goal.goal_id].append(raw[is_threshold])
            if np.any(~is_threshold):
                validation[goal.goal_id][0].append(raw[~is_threshold])
                validation[goal.goal_id][1].append(hat[~is_threshold])
        if np.any(~is_threshold):
            keep = np.flatnonzero(~is_threshold)
            meta.append(
                part.with_row_index("row")
                .filter(pl.col("row").is_in(keep.tolist()))
                .sort("row")
                .select("cell_line", "plate_num", "compound_key")
            )
        if index % 400 == 0:
            print(f"  gate {index}/{len(fragments)} fragments", flush=True)
    return threshold, validation, meta


def _explode(path: Path, part: pl.DataFrame) -> pl.DataFrame:
    counts = pl.read_parquet(path, columns=["cell_id", "genes", "counts"])
    matched = part.select("cell_id").join(counts, on="cell_id", how="left")
    if matched["genes"].null_count():
        raise RuntimeError(f"{matched['genes'].null_count()} cells missing from {path.name}")
    return matched.explode(["genes", "counts"]).filter(pl.col("counts") > 0)


def _fill_block(
    path: Path, part: pl.DataFrame, positions: np.ndarray, known: np.ndarray
) -> np.ndarray:
    counts = pl.read_parquet(path, columns=["cell_id", "genes", "counts"])
    ordered = part.with_row_index("matrix_row").select("matrix_row", "cell_id", "scale")
    matched = ordered.join(counts, on="cell_id", how="left")
    if matched["genes"].null_count():
        raise RuntimeError(f"cells missing from {path.name}")
    matrix = np.zeros((part.height, int((positions >= 0).sum())), np.float32)
    long = matched.explode(["genes", "counts"]).filter(pl.col("counts") > 0)
    if long.is_empty():
        return matrix
    genes_idx = long["genes"].to_numpy()
    if np.any(~known[genes_idx]):
        raise RuntimeError(f"unknown gene token in {path.name}")
    where = positions[genes_idx]
    keep = where >= 0
    if not np.any(keep):
        return matrix
    values = np.log1p(
        long["counts"].to_numpy()[keep].astype(np.float64) * long["scale"].to_numpy()[keep]
    )
    matrix[long["matrix_row"].to_numpy()[keep], where[keep]] = values.astype(np.float32)
    return matrix


def _by_fragment(frame: pl.DataFrame):
    grouped = list(frame.group_by("fragment"))
    for key, part in sorted(
        grouped, key=lambda item: item[0][0] if isinstance(item[0], tuple) else item[0]
    ):
        fragment = key[0] if isinstance(key, tuple) else key
        yield str(fragment), part
