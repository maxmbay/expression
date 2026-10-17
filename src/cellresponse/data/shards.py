"""Row-group index of the expression shards, built from Parquet footers only.

The expression shards are ordered by plate and sample, and every row group carries
min/max statistics for those columns. Reading footers lets us fetch only the row
groups that contain selected wells instead of streaming the whole atlas.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from cellresponse.data.source import REPO_ID, list_files

STAT_COLUMNS = ("plate", "sample", "drug", "cell_line_id", "BARCODE_SUB_LIB_ID")


def remote_path(revision: str, repo_path: str) -> str:
    return f"datasets/{REPO_ID}@{revision}/{repo_path}"


def _stat(column, kind: str) -> str | None:
    stats = column.statistics
    if stats is None or not stats.has_min_max:
        return None
    value = stats.min if kind == "min" else stats.max
    return value.decode() if isinstance(value, bytes) else value


def footer_rows(fs: HfFileSystem, revision: str, repo_path: str) -> list[dict]:
    with fs.open(remote_path(revision, repo_path), "rb", block_size=1 << 16) as handle:
        metadata = pq.ParquetFile(handle).metadata
    names = [metadata.schema.column(i).path for i in range(metadata.num_columns)]
    rows = []
    for g in range(metadata.num_row_groups):
        group = metadata.row_group(g)
        columns = {names[i]: group.column(i) for i in range(group.num_columns)}
        row = {
            "shard": repo_path,
            "row_group": g,
            "num_rows": group.num_rows,
            "compressed_bytes": sum(c.total_compressed_size for c in columns.values()),
            "expression_bytes": sum(
                c.total_compressed_size
                for n, c in columns.items()
                if n.startswith(("genes", "expressions"))
            ),
        }
        for name in STAT_COLUMNS:
            row[f"{name}_min"] = _stat(columns[name], "min")
            row[f"{name}_max"] = _stat(columns[name], "max")
        rows.append(row)
    return rows


def build_shard_index(
    revision: str, out: Path, workers: int = 16, limit: int | None = None
) -> pl.DataFrame:
    shards = [f.path for f in list_files(revision, prefix="data") if f.path.endswith(".parquet")]
    if limit is not None:
        shards = shards[:limit]
    done: dict[str, list[dict]] = {}
    if out.exists():
        previous = pl.read_parquet(out)
        for shard, frame in previous.group_by("shard"):
            done[shard[0]] = frame.to_dicts()
    todo = [s for s in shards if s not in done]
    fs = HfFileSystem()
    started = time.time()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(footer_rows, fs, revision, s): s for s in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            shard = futures[future]
            try:
                done[shard] = future.result()
            except Exception as error:  # noqa: BLE001 - retried on the next resumable run
                failures.append(f"{shard}: {error!r}")
            if i % 200 == 0 or i == len(todo):
                _write(done, out)
                print(f"footers {i}/{len(todo)} in {time.time() - started:.0f}s", flush=True)
    index = _write(done, out)
    if failures:
        raise RuntimeError(f"{len(failures)} footer reads failed; rerun to resume: {failures[:3]}")
    return index


def _write(done: dict[str, list[dict]], out: Path) -> pl.DataFrame:
    rows = [row for shard in sorted(done) for row in done[shard]]
    frame = pl.DataFrame(rows).sort("shard", "row_group")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    return frame
