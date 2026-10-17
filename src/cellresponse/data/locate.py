"""Exact cell -> (shard, row group, row) locations, reconstructed from metadata and verified.

Hypothesis checked here rather than assumed: the expression rows are the obs rows with
`pass_filter == "full"`, grouped in contiguous sublibrary blocks whose order is read from
row-group footer statistics, with obs order preserved inside each block. The reconstruction
is accepted only if predicted min/max statistics match every row-group footer exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

VERIFIED_COLUMNS = {
    "BARCODE_SUB_LIB_ID": "BARCODE_SUB_LIB_ID",
    "sample": "sample",
    "plate": "plate",
    "drug": "drug",
    "cell_line_id": "cell_line",
}


def _suffix(column: str) -> pl.Expr:
    return pl.col(column).str.split("-").list.last()


def sublibrary_order(index: pl.DataFrame) -> list[str]:
    pure = index.sort("shard", "row_group").filter(
        _suffix("BARCODE_SUB_LIB_ID_min") == _suffix("BARCODE_SUB_LIB_ID_max")
    )
    sequence = pure.select(_suffix("BARCODE_SUB_LIB_ID_min").alias("sub"))["sub"].to_list()
    order: list[str] = []
    for sub in sequence:
        if not order or order[-1] != sub:
            if sub in order:
                raise ValueError(f"sublibrary {sub} appears in non-contiguous expression blocks")
            order.append(sub)
    return order


def build_locations(obs_path: Path, index: pl.DataFrame) -> pl.DataFrame:
    """Return obs_row, shard_id, row_group, row for every retained cell."""
    index = index.sort("shard", "row_group").with_row_index("rg_id")
    order = sublibrary_order(index)
    ranks = pl.DataFrame({"sublibrary": order, "sub_rank": np.arange(len(order), dtype=np.int32)})
    full = (
        pl.scan_parquet(obs_path)
        .with_row_index("obs_row")
        .filter(pl.col("pass_filter") == "full")
        .select("obs_row", "sublibrary")
        .collect(engine="streaming")
    )
    missing = set(full["sublibrary"].unique()) - set(order)
    if missing:
        raise ValueError(
            f"{len(missing)} sublibraries have no pure row group, e.g. {sorted(missing)[:3]}"
        )
    full = full.join(ranks, on="sublibrary", how="left").sort(
        "sub_rank", "obs_row", maintain_order=True
    )
    if full.height != int(index["num_rows"].sum()):
        raise ValueError(
            f"{full.height} retained obs rows vs {index['num_rows'].sum()} expression rows"
        )
    starts = np.concatenate([[0], np.cumsum(index["num_rows"].to_numpy())[:-1]])
    position = np.arange(full.height, dtype=np.int64)
    rg_id = np.searchsorted(starts, position, side="right") - 1
    shard_id = (
        index["shard"].str.extract(r"train-(\d+)-of").cast(pl.Int32).to_numpy()
    )
    return pl.DataFrame(
        {
            "obs_row": full["obs_row"],
            "rg_id": rg_id.astype(np.int32),
            "shard_id": shard_id[rg_id].astype(np.int16),
            "row_group": index["row_group"].to_numpy()[rg_id].astype(np.int16),
            "row": (position - starts[rg_id]).astype(np.int32),
        }
    )


def verify_locations(obs_path: Path, index: pl.DataFrame, locations: pl.DataFrame) -> pl.DataFrame:
    """Compare predicted per-row-group min/max with footer statistics; return mismatching groups."""
    index = index.sort("shard", "row_group").with_row_index("rg_id")
    # Expression shards store drug names stripped of the trailing spaces present in obs.
    obs = pl.scan_parquet(obs_path).with_row_index("obs_row").select(
        "obs_row",
        *[pl.col(c).cast(pl.String) for c in set(VERIFIED_COLUMNS.values()) - {"drug"}],
        pl.col("drug").str.strip_chars(),
    )
    predicted = (
        locations.lazy()
        .select("obs_row", "rg_id")
        .join(obs, on="obs_row", how="left")
        .group_by("rg_id")
        .agg(
            pl.len().alias("n"),
            *[pl.col(src).min().alias(f"{dst}_pmin") for dst, src in VERIFIED_COLUMNS.items()],
            *[pl.col(src).max().alias(f"{dst}_pmax") for dst, src in VERIFIED_COLUMNS.items()],
        )
        .collect(engine="streaming")
    )
    joined = index.join(predicted, on="rg_id", how="left")
    checks = [pl.col("num_rows") != pl.col("n")]
    for dst in VERIFIED_COLUMNS:
        checks.append(pl.col(f"{dst}_min") != pl.col(f"{dst}_pmin"))
        checks.append(pl.col(f"{dst}_max") != pl.col(f"{dst}_pmax"))
    return joined.filter(pl.any_horizontal(checks) | pl.col("n").is_null())
