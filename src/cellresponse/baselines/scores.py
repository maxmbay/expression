"""Module scores from raw counts. Test-treated cells are not written here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from cellresponse.features.fit import _by_fragment
from cellresponse.goals import Goal, goals

SCORE_VERSION = "v1-exclude-protocol-a-test-treated"
PENALTY_GOAL = "apoptosis_constraint"
PENALTY_WEIGHT = 1.0


def score_names() -> list[str]:
    return [goal.goal_id for goal in goals()]


def objective_goals() -> tuple[Goal, ...]:
    return tuple(goal for goal in goals() if goal.role == "objective")


def goal_weights(genes: pl.DataFrame) -> np.ndarray:
    """`weight[token, goal]` makes a sum over expressed genes the module mean."""
    symbol_to_token = dict(
        zip(genes["gene_symbol"].to_list(), genes["token_id"].to_list(), strict=True)
    )
    weight = np.zeros((int(genes["token_id"].max()) + 1, len(goals())), np.float32)
    for index, goal in enumerate(goals()):
        present = [symbol_to_token[symbol] for symbol in goal.symbols if symbol in symbol_to_token]
        if not present:
            raise RuntimeError(f"{goal.goal_id} has no genes in the vocabulary")
        weight[present, index] = 1.0 / len(present)
    return weight


def score_cells(prepared: Path, genes: pl.DataFrame, cell_ids: pl.Series) -> pl.DataFrame:
    """Score these cells. Call this only after the test rankings have been hashed."""
    cells = pl.read_parquet(
        prepared / "cells.parquet", columns=["cell_id", "library_size", "fragment"]
    ).join(pl.DataFrame({"cell_id": cell_ids}), on="cell_id", how="semi")
    if cells.height == 0:
        raise RuntimeError("no cells to score")
    frame = cells.with_columns(scale=pl.lit(10_000.0) / pl.col("library_size"))
    weight = goal_weights(genes)
    known = np.zeros(weight.shape[0], dtype=bool)
    known[genes["token_id"].to_numpy()] = True
    pieces = [
        _score_fragment(prepared / "fragments" / fragment, part, weight, known)
        for fragment, part in _by_fragment(frame)
    ]
    scored = pl.concat(pieces)
    if scored.height != frame.height:
        raise RuntimeError(f"scored {scored.height} of {frame.height} cells")
    return scored


def ensure_scores(prepared: Path, genes: pl.DataFrame) -> pl.DataFrame:
    path = prepared / "module_scores.parquet"
    meta_path = prepared / "module_scores.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("version") == SCORE_VERSION:
            print(f"[scores] reusing {path}", flush=True)
            return pl.read_parquet(path)
    splits = pl.read_parquet(
        prepared / "splits.parquet", columns=["cell_id", "protocol_a", "is_control"]
    )
    keep = splits.filter(~((pl.col("protocol_a") == "test") & ~pl.col("is_control")))
    cells = pl.read_parquet(
        prepared / "cells.parquet", columns=["cell_id", "library_size", "fragment"]
    ).join(keep.select("cell_id"), on="cell_id", how="semi")
    if cells.filter(pl.col("library_size") <= 0).height:
        raise RuntimeError("non-positive library size")
    frame = cells.with_columns(scale=pl.lit(10_000.0) / pl.col("library_size"))
    print(f"[scores] {frame.height:,} cells, test-treated held out", flush=True)
    weight = goal_weights(genes)
    known = np.zeros(weight.shape[0], dtype=bool)
    known[genes["token_id"].to_numpy()] = True
    pieces = []
    fragments = list(_by_fragment(frame))
    for index, (fragment, part) in enumerate(fragments, start=1):
        pieces.append(_score_fragment(prepared / "fragments" / fragment, part, weight, known))
        if index % 400 == 0:
            print(f"  scores {index}/{len(fragments)} fragments", flush=True)
    scored = pl.concat(pieces)
    if scored.height != frame.height:
        raise RuntimeError(f"scored {scored.height} of {frame.height} cells")
    scored.write_parquet(path)
    meta_path.write_text(
        json.dumps({"version": SCORE_VERSION, "cells": scored.height, "columns": score_names()})
        + "\n"
    )
    print(f"[scores] wrote {path}", flush=True)
    return scored


def _score_fragment(
    path: Path, part: pl.DataFrame, weight: np.ndarray, known: np.ndarray
) -> pl.DataFrame:
    counts = pl.read_parquet(path, columns=["cell_id", "genes", "counts"])
    ordered = part.with_row_index("matrix_row").select("matrix_row", "cell_id", "scale")
    matched = ordered.join(counts, on="cell_id", how="left")
    if matched["genes"].null_count():
        raise RuntimeError(f"cells missing from {path.name}")
    scores = np.zeros((part.height, weight.shape[1]), np.float64)
    long = matched.explode(["genes", "counts"]).filter(pl.col("counts") > 0)
    if long.height:
        genes_idx = long["genes"].to_numpy()
        if np.any(~known[genes_idx]):
            raise RuntimeError(f"unknown gene token in {path.name}")
        values = np.log1p(long["counts"].to_numpy().astype(np.float64) * long["scale"].to_numpy())
        rows = long["matrix_row"].to_numpy()
        contrib = weight[genes_idx] * values[:, None]
        for column in range(weight.shape[1]):
            scores[:, column] = np.bincount(rows, weights=contrib[:, column], minlength=part.height)
    return pl.DataFrame({"cell_id": ordered["cell_id"]}).with_columns(
        [
            pl.Series(name, scores[:, index].astype(np.float32))
            for index, name in enumerate(score_names())
        ]
    )
