"""`cellresponse prepare`: fetch a frozen cohort and write canonical local tables.

Counts stay in per-source-shard fragments (sparse list columns, zstd); `cells.parquet`
points at `(fragment, fragment_row)`. Fragments are written atomically, so an interrupted
run resumes by skipping completed shards.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from cellresponse.data import audit as A
from cellresponse.data.expression import shard_name
from cellresponse.data.parse import strip_marker_batch
from cellresponse.data.remote import RemoteParquet
from cellresponse.data.select import CohortSpec, candidate_cells, plan_row_groups, select_cells
from cellresponse.data.source import sha256sum

OBS_FIELDS = [
    "BARCODE_SUB_LIB_ID",
    "plate",
    "sample",
    "cell_line",
    "cell_name",
    "sublibrary",
    "BARCODE",
    "tscp_count",
    "gene_count",
    "mread_count",
    "pcnt_mito",
    "S_score",
    "G2M_score",
    "phase",
    "pass_filter",
]
FRAGMENT_SCHEMA = pa.schema(
    [
        ("cell_id", pa.uint32()),
        ("BARCODE_SUB_LIB_ID", pa.string()),
        ("genes", pa.list_(pa.int32())),
        ("counts", pa.list_(pa.int32())),
    ]
)


def spec_from_config(config: dict) -> CohortSpec:
    cohort = config["cohort"]
    compounds = cohort.get("compounds")
    return CohortSpec(
        plates=tuple(cohort["plates"]),
        lines=tuple(cohort["lines"]),
        compounds=tuple(compounds) if compounds else None,
        treated_cap=cohort.get("treated_cap"),
        control_cap=cohort.get("control_cap"),
        seed=cohort.get("seed", 0),
    )


def run(config: dict) -> None:
    started = time.time()
    revision = config["source"]["revision"]
    raw = Path(config["paths"]["raw"])
    interim = Path(config["paths"]["interim"])
    out = Path(config["paths"]["prepared"]) / config["name"]
    fragments = out / "fragments"
    fragments.mkdir(parents=True, exist_ok=True)
    spec = spec_from_config(config)
    obs_path = raw / "metadata/obs_metadata.parquet"

    samples = A.load_samples(raw)
    locations = pl.read_parquet(interim / "cell_locations.parquet")
    candidates = candidate_cells(obs_path, locations, samples, spec)
    visit = plan_row_groups(candidates, spec)
    chosen = select_cells(candidates, visit, spec).sort("obs_row")
    selection_hash = hashlib.sha256(chosen["obs_row"].to_numpy().tobytes()).hexdigest()
    chosen.write_parquet(out / "selection.parquet")
    print(
        f"[{config['name']}] selected {chosen.height:,} cells from "
        f"{chosen['rg_id'].n_unique():,} row groups in {chosen['shard_id'].n_unique():,} shards",
        flush=True,
    )

    fetch_stats = fetch(
        revision, chosen, fragments, workers=config.get("download", {}).get("workers", 8)
    )
    cells = build_cells(obs_path, samples, chosen, fragments)
    verify_counts(cells, fragments)
    write_tables(config, raw, samples, cells, out)
    write_manifest(config, out, spec, selection_hash, cells, fetch_stats, started)
    print(f"[{config['name']}] done in {(time.time() - started) / 60:.1f} min -> {out}", flush=True)


def fetch(revision: str, chosen: pl.DataFrame, fragments: Path, workers: int) -> dict:
    jobs = {
        int(shard_id[0]): frame.select("obs_row", "BARCODE_SUB_LIB_ID", "row_group", "row")
        for shard_id, frame in chosen.group_by("shard_id")
    }
    todo = sorted(s for s in jobs if not _fragment_path(fragments, s).exists())
    print(
        f"fetch: {len(jobs) - len(todo):,} shards already done, {len(todo):,} to fetch", flush=True
    )
    started, done_bytes, failures = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_shard, revision, s, jobs[s], fragments): s for s in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            shard_id = futures[future]
            try:
                done_bytes += future.result()
            except Exception as error:  # noqa: BLE001 - reported and retried on the next run
                failures.append(f"shard {shard_id}: {error!r}")
            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - started
                rate = done_bytes / max(elapsed, 1e-9) / 1e6
                eta = elapsed / i * (len(todo) - i) / 60
                print(
                    f"fetch {i:,}/{len(todo):,} shards  {done_bytes / 1e9:.2f} GB  "
                    f"{rate:.1f} MB/s  eta {eta:.0f} min  failures {len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"{len(failures)} shards failed; rerun to resume. First: {failures[:3]}")
    return {
        "shards": len(jobs),
        "fetched_this_run": len(todo),
        "bytes_this_run": done_bytes,
        "seconds_this_run": round(time.time() - started, 1),
    }


def _fragment_path(fragments: Path, shard_id: int) -> Path:
    return fragments / f"shard-{shard_id:05d}.parquet"


def _fetch_shard(
    revision: str,
    shard_id: int,
    rows: pl.DataFrame,
    fragments: Path,
    attempts: int = 6,
) -> int:
    for attempt in range(attempts):
        try:
            return _fetch_shard_once(revision, shard_id, rows, fragments)
        except Exception as error:
            if attempt == attempts - 1:
                raise
            wait = min(120, 5 * 2**attempt)
            print(f"retry shard {shard_id} in {wait}s after {error!r:.200}", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


FETCH_COLUMNS = ("BARCODE_SUB_LIB_ID", "genes", "expressions")


def _fetch_shard_once(revision: str, shard_id: int, rows: pl.DataFrame, fragments: Path) -> int:
    remote = RemoteParquet(shard_name(shard_id), revision)
    row_groups = sorted(int(g) for g in rows["row_group"].unique())
    remote.fetch_row_groups(row_groups, FETCH_COLUMNS)
    parquet = remote.parquet_file()
    tables = []
    for (row_group,), wanted in rows.sort("row").group_by("row_group", maintain_order=True):
        table = parquet.read_row_group(
            int(row_group), columns=["BARCODE_SUB_LIB_ID", "genes", "expressions"]
        ).take(pa.array(wanted["row"].to_numpy()))
        barcodes = table.column("BARCODE_SUB_LIB_ID").to_pylist()
        if barcodes != wanted["BARCODE_SUB_LIB_ID"].to_list():
            raise ValueError(f"shard {shard_id} rg {row_group}: barcode mismatch with location map")
        genes = table.column("genes").combine_chunks()
        counts = table.column("expressions").combine_chunks()
        offsets, gene_values, count_values = strip_marker_batch(
            genes.offsets.to_numpy(), genes.values.to_numpy(), counts.values.to_numpy()
        )
        arrow_offsets = pa.array(offsets.astype(np.int32))
        tables.append(
            pa.table(
                {
                    "cell_id": pa.array(wanted["obs_row"].to_numpy(), pa.uint32()),
                    "BARCODE_SUB_LIB_ID": pa.array(barcodes, pa.string()),
                    "genes": pa.ListArray.from_arrays(arrow_offsets, pa.array(gene_values)),
                    "counts": pa.ListArray.from_arrays(arrow_offsets, pa.array(count_values)),
                },
                schema=FRAGMENT_SCHEMA,
            )
        )
    target = _fragment_path(fragments, shard_id)
    partial = target.with_suffix(".part")
    pq.write_table(pa.concat_tables(tables), partial, compression="zstd", compression_level=3)
    partial.replace(target)
    return remote.bytes_fetched


def build_cells(
    obs_path: Path, samples: pl.DataFrame, chosen: pl.DataFrame, fragments: Path
) -> pl.DataFrame:
    obs = (
        pl.scan_parquet(obs_path)
        .with_row_index("obs_row")
        .filter(pl.col("obs_row").is_in(chosen["obs_row"].implode()))
        .select("obs_row", *OBS_FIELDS)
        .with_columns(
            pl.col(c).cast(pl.String) for c in ("cell_line", "cell_name", "phase", "pass_filter")
        )
        .collect(engine="streaming")
    )
    pointers = pl.concat(
        [
            pl.read_parquet(path, columns=["cell_id"])
            .with_row_index("fragment_row")
            .with_columns(fragment=pl.lit(path.name))
            for path in sorted(fragments.glob("shard-*.parquet"))
        ]
    ).rename({"cell_id": "obs_row"})
    wells = samples.select("sample", "plate_num", "compound_key", "drug", "dose_uM", "is_control")
    cells = (
        chosen.select("obs_row", "shard_id", "row_group", "row")
        .join(obs, on="obs_row", how="left", validate="1:1")
        .join(wells, on="sample", how="left", validate="m:1")
        .join(pointers.with_columns(pl.col("obs_row").cast(pl.UInt32)), on="obs_row", how="left")
        .rename({"obs_row": "cell_id", "tscp_count": "library_size"})
        .with_columns(pl.col("drug").str.strip_chars())
        .sort("cell_id")
    )
    missing = cells.filter(pl.col("fragment").is_null()).height
    if missing:
        raise RuntimeError(f"{missing} selected cells have no fetched counts")
    return cells


def verify_counts(cells: pl.DataFrame, fragments: Path) -> None:
    """Landmark check: summed counts equal library size and nnz equals gene_count for every cell."""
    observed = pl.concat(
        [
            pl.scan_parquet(path)
            .select(
                pl.col("cell_id"),
                pl.col("counts").list.sum().alias("total"),
                # gene_count counts expressed genes; a few cells store an explicit zero.
                pl.col("counts").list.eval(pl.element() > 0).list.sum().alias("nnz"),
            )
            .collect()
            for path in sorted(fragments.glob("shard-*.parquet"))
        ]
    )
    joined = cells.select("cell_id", "library_size", "gene_count").join(
        observed, on="cell_id", how="left"
    )
    bad = joined.filter(
        (pl.col("total") != pl.col("library_size")) | (pl.col("nnz") != pl.col("gene_count"))
    )
    if bad.height:
        raise RuntimeError(
            f"{bad.height} cells fail count verification, e.g. {bad.head(3).to_dicts()}"
        )
    print(f"verified counts for {joined.height:,} cells", flush=True)


def write_tables(
    config: dict, raw: Path, samples: pl.DataFrame, cells: pl.DataFrame, out: Path
) -> None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    cells.write_parquet(out / "cells.parquet")

    counts = pl.read_parquet(Path(config["paths"]["interim"]) / "cell_counts.parquet")
    available = (
        counts.filter(pl.col("pass_filter") == "full")
        .group_by("sample", "cell_line")
        .agg(available_cells=pl.col("n").sum())
    )
    excluded = (
        counts.filter(pl.col("pass_filter") == "minimal")
        .group_by("sample", "cell_line")
        .agg(minimal_qc_cells=pl.col("n").sum())
    )
    retained = cells.group_by("sample", "cell_line").agg(retained_cells=pl.len())
    references = (
        samples.filter(pl.col("is_control"))
        .group_by("plate_num")
        .agg(control_reference_wells=pl.col("sample").sort())
    )
    conditions = (
        retained.join(
            samples.select(
                "sample", "plate", "plate_num", "compound_key", "drug", "dose_uM", "is_control"
            ),
            on="sample",
            how="left",
        )
        .join(available, on=["sample", "cell_line"], how="left")
        .join(excluded, on=["sample", "cell_line"], how="left")
        .join(references, on="plate_num", how="left")
        .with_columns(
            pl.col("minimal_qc_cells").fill_null(0),
            pl.col("drug").str.strip_chars(),
            split_role=pl.lit(None, dtype=pl.String),
        )
        .sort("plate_num", "sample", "cell_line")
    )
    conditions.write_parquet(out / "conditions.parquet")

    chem = A.compound_audit(raw, samples)["chemistry"]
    in_cohort = cells.filter(~pl.col("is_control"))["compound_key"].unique()
    drugs = pl.read_parquet(raw / "metadata/drug_metadata.parquet").select(
        "drug", "canonical_smiles"
    )
    compounds = chem.filter(pl.col("compound_key").is_in(in_cohort.implode())).join(
        drugs, on="drug", how="left"
    )
    scaffolds = []
    for smiles in compounds["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            scaffolds.append(None)
            continue
        parent = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=parent) or "acyclic")
    compounds = compounds.with_columns(
        structure_group=pl.col("inchikey_parent").fill_null(pl.col("compound_key")),
        scaffold=pl.Series(scaffolds, dtype=pl.String),
        fingerprint_valid=pl.col("smiles_valid"),
    ).sort("compound_key")
    compounds.write_parquet(out / "compounds.parquet")

    genes = pl.read_parquet(raw / "metadata/gene_metadata.parquet").sort("token_id")
    genes.with_row_index("column").write_parquet(out / "genes.parquet")


def write_manifest(
    config: dict,
    out: Path,
    spec: CohortSpec,
    selection_hash: str,
    cells: pl.DataFrame,
    fetch_stats: dict,
    started: float,
) -> None:
    fragment_files = sorted((out / "fragments").glob("shard-*.parquet"))
    config_bytes = json.dumps(config, sort_keys=True).encode()
    manifest = {
        "name": config["name"],
        "source": {
            "repo_id": config["source"]["repo_id"],
            "revision": config["source"]["revision"],
            "url": f"https://huggingface.co/datasets/{config['source']['repo_id']}/tree/"
            f"{config['source']['revision']}",
            "license": "CC0-1.0 (dataset card); see reports/source_manifest.json",
        },
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "cohort": {
            "plates": list(spec.plates),
            "lines": list(spec.lines),
            "compounds": list(spec.compounds) if spec.compounds else None,
            "treated_cap": spec.treated_cap,
            "control_cap": spec.control_cap,
            "seed": spec.seed,
        },
        "selection_sha256": selection_hash,
        "cells": cells.height,
        "control_cells": int(cells["is_control"].sum()),
        "wells": cells["sample"].n_unique(),
        "fragments": {
            "count": len(fragment_files),
            "bytes": sum(p.stat().st_size for p in fragment_files),
            "sha256": {p.name: sha256sum(p) for p in fragment_files},
        },
        "fetch": fetch_stats,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{pkg: version(pkg) for pkg in ("polars", "pyarrow", "numpy", "huggingface-hub")},
        },
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
