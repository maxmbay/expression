"""Reading and validating sparse expression rows from the pinned Tahoe shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from scipy import sparse

from cellresponse.data.parse import strip_marker
from cellresponse.data.shards import remote_path

METADATA_COLUMNS = ["BARCODE_SUB_LIB_ID", "sample", "plate", "drug", "cell_line_id"]


def shard_name(shard_id: int, total: int = 3388) -> str:
    return f"data/train-{shard_id:05d}-of-{total:05d}.parquet"


def read_row_group(revision: str, shard_id: int, row_group: int, fs: HfFileSystem | None = None):
    fs = fs or HfFileSystem()
    with fs.open(remote_path(revision, shard_name(shard_id)), "rb", block_size=1 << 22) as handle:
        return pq.ParquetFile(handle).read_row_group(
            row_group, columns=["genes", "expressions", *METADATA_COLUMNS]
        )


def to_csr(table, token_to_col: dict[int, int]) -> tuple[sparse.csr_matrix, pl.DataFrame, dict]:
    """Parse rows into a cells x genes CSR matrix; unknown tokens are counted, never silently kept."""
    genes_col = table.column("genes").combine_chunks()
    expr_col = table.column("expressions").combine_chunks()
    offsets = genes_col.offsets.to_numpy()
    all_genes = genes_col.values.to_numpy()
    all_counts = expr_col.values.to_numpy()
    if not np.array_equal(offsets, expr_col.offsets.to_numpy()):
        raise ValueError("genes and expressions list offsets differ")
    data, indices, indptr = [], [], [0]
    unknown_tokens = 0
    unknown_counts = 0.0
    totals = np.zeros(table.num_rows)
    for i in range(table.num_rows):
        genes, counts = strip_marker(
            all_genes[offsets[i] : offsets[i + 1]], all_counts[offsets[i] : offsets[i + 1]]
        )
        cols = np.fromiter((token_to_col.get(int(g), -1) for g in genes), dtype=np.int64, count=genes.size)
        known = cols >= 0
        unknown_tokens += int((~known).sum())
        unknown_counts += float(counts[~known].sum())
        totals[i] = counts.sum()
        order = np.argsort(cols[known])
        indices.append(cols[known][order])
        data.append(counts[known][order])
        indptr.append(indptr[-1] + int(known.sum()))
    matrix = sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=(table.num_rows, len(token_to_col)),
        dtype=np.float32,
    )
    obs = pl.from_arrow(table.select(METADATA_COLUMNS)).with_columns(total_counts_all_tokens=totals)
    stats = {"unknown_token_entries": unknown_tokens, "unknown_token_counts": unknown_counts}
    return matrix, obs, stats


def gene_vocabulary(raw: Path) -> tuple[pl.DataFrame, dict[int, int]]:
    genes = pl.read_parquet(raw / "metadata/gene_metadata.parquet").sort("token_id")
    if genes["token_id"].is_duplicated().any():
        raise ValueError("duplicate token IDs in gene metadata")
    return genes, {int(t): i for i, t in enumerate(genes["token_id"])}
