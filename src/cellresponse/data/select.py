"""Deterministic, row-group-aware cohort selection.

Each well's cells are spread over many row groups (one slice per sublibrary), and line
composition within a sublibrary matches its well up to multinomial noise. So a capped
cohort can be drawn by visiting each well's row groups in a hash-determined order and
stopping once every selected line reaches its cap. Only visited row groups are downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class CohortSpec:
    plates: tuple[int, ...]
    lines: tuple[str, ...]
    compounds: tuple[str, ...] | None = None  # normalized compound keys; None keeps all
    treated_cap: int | None = 256  # cells per (well, line)
    control_cap: int | None = None  # cells per (vehicle well, line); None keeps every control
    seed: int = 0

    def cap_for(self, is_control: bool) -> int | None:
        return self.control_cap if is_control else self.treated_cap


def _hash_key(column: str, seed: int) -> pl.Expr:
    return pl.col(column).hash(seed=seed)


def candidate_cells(
    obs_path: Path, locations: pl.DataFrame, samples: pl.DataFrame, spec: CohortSpec
) -> pl.DataFrame:
    chosen = samples.filter(pl.col("plate_num").is_in(spec.plates))
    if spec.compounds is not None:
        chosen = chosen.filter(pl.col("is_control") | pl.col("compound_key").is_in(spec.compounds))
    cells = (
        pl.scan_parquet(obs_path)
        .with_row_index("obs_row")
        .filter(
            (pl.col("pass_filter") == "full")
            & pl.col("sample").is_in(chosen["sample"].implode())
            & pl.col("cell_line").cast(pl.String).is_in(list(spec.lines))
        )
        .select("obs_row", "sample", pl.col("cell_line").cast(pl.String), "BARCODE_SUB_LIB_ID")
        .collect(engine="streaming")
    )
    return cells.join(locations, on="obs_row", how="left", validate="1:1").join(
        chosen.select("sample", "is_control"), on="sample", how="left"
    )


def plan_row_groups(cells: pl.DataFrame, spec: CohortSpec) -> pl.DataFrame:
    """Return the (sample, rg_id) pairs to visit."""
    per_group = (
        cells.group_by("sample", "is_control", "cell_line", "rg_id")
        .agg(n=pl.len())
        .with_columns(key=_hash_key("rg_id", spec.seed))
        .sort("sample", "cell_line", "key")
        .with_columns(cum=pl.col("n").cum_sum().over("sample", "cell_line"))
    )
    per_group = per_group.with_columns(cap=_cap_expr(spec))
    reach = per_group.group_by("sample", "cell_line").agg(
        reach=pl.when(pl.col("cap").first().is_null())
        .then(pl.col("key").max())
        .otherwise(pl.col("key").filter(pl.col("cum") >= pl.col("cap")).min().fill_null(pl.col("key").max()))
    )
    cutoff = reach.group_by("sample").agg(cutoff=pl.col("reach").max())
    groups = per_group.select("sample", "rg_id", "key").unique()
    return groups.join(cutoff, on="sample").filter(pl.col("key") <= pl.col("cutoff")).select("sample", "rg_id")


def select_cells(cells: pl.DataFrame, visit: pl.DataFrame, spec: CohortSpec) -> pl.DataFrame:
    """Hash-sample up to the cap per (well, line) among cells in visited row groups."""
    visited = cells.join(visit, on=["sample", "rg_id"], how="semi")
    ranked = visited.with_columns(
        key=_hash_key("BARCODE_SUB_LIB_ID", spec.seed), cap=_cap_expr(spec)
    ).with_columns(rank=pl.col("key").rank("ordinal").over("sample", "cell_line"))
    keep = pl.col("cap").is_null() | (pl.col("rank") <= pl.col("cap"))
    return ranked.filter(keep).drop("key", "rank", "cap")


def _cap_expr(spec: CohortSpec) -> pl.Expr:
    return (
        pl.when(pl.col("is_control"))
        .then(pl.lit(spec.control_cap, dtype=pl.Int64))
        .otherwise(pl.lit(spec.treated_cap, dtype=pl.Int64))
    )


def footprint_summary(index: pl.DataFrame, visit: pl.DataFrame, selected: pl.DataFrame) -> dict:
    index = index.sort("shard", "row_group").with_row_index("rg_id").with_columns(pl.col("rg_id").cast(pl.Int32))
    groups = index.join(visit.select("rg_id").unique(), on="rg_id", how="semi")
    shards = groups["shard"].unique()
    shard_bytes = index.filter(pl.col("shard").is_in(shards.implode())).group_by("shard").agg(
        pl.col("compressed_bytes").sum()
    )
    return {
        "row_groups": groups.height,
        "shards": shards.len(),
        "rowgroup_download_gb": round(groups["compressed_bytes"].sum() / 1e9, 2),
        "whole_shard_download_gb": round(shard_bytes["compressed_bytes"].sum() / 1e9, 2),
        "selected_cells": selected.height,
        "selected_control_cells": int(selected["is_control"].sum()),
    }
