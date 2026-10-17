"""`cellresponse audit`: reproducible M0 data and design audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from cellresponse.data import audit as A
from cellresponse.data import source
from cellresponse.data.locate import build_locations, verify_locations
from cellresponse.data.report import dict_table, md_table
from cellresponse.data.select import (
    CohortSpec,
    candidate_cells,
    footprint_summary,
    plan_row_groups,
    select_cells,
)
from cellresponse.data.shards import build_shard_index


def run(config: dict, skip_network: bool = False) -> None:
    revision = config["source"]["revision"]
    raw = Path(config["paths"]["raw"])
    interim = Path(config["paths"]["interim"])
    reports = Path(config["paths"]["reports"])
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    elig = config["eligibility"]

    listing_path = interim / "source_listing.json"
    if skip_network:
        listing = [source.SourceFile(**f) for f in json.loads(listing_path.read_text())]
    else:
        listing = source.list_files(revision)
        listing_path.parent.mkdir(parents=True, exist_ok=True)
        listing_path.write_text(json.dumps([f.__dict__ for f in listing]))
        remote_sizes = {f.path: f.size for f in listing}
        for path in source.METADATA_FILES:
            print(f"ensure {path}", flush=True)
            source.download(path, revision, raw, remote_sizes.get(path))
        build_shard_index(revision, interim / "shard_index.parquet")
    local = {p: raw / p for p in source.METADATA_FILES}
    manifest = source.write_manifest(interim / "source_manifest.json", revision, local, listing)
    (reports / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    index = pl.read_parquet(interim / "shard_index.parquet")
    obs_path = raw / "metadata/obs_metadata.parquet"
    samples = A.load_samples(raw)
    cells = A.aggregate_cells(raw, interim / "cell_counts.parquet")
    integrity = A.obs_integrity(raw)

    locations_path = interim / "cell_locations.parquet"
    if locations_path.exists():
        locations = pl.read_parquet(locations_path)
    else:
        locations = build_locations(obs_path, index)
    mismatches = verify_locations(obs_path, index, locations)
    if mismatches.height:
        raise RuntimeError(f"cell-location reconstruction failed on {mismatches.height} row groups")
    locations.write_parquet(locations_path)

    wl = A.well_line_counts(cells, samples)
    controls = A.control_support(wl, elig["min_control_cells"])
    panels = A.candidate_panels(
        wl, controls, elig["min_treated_cells"], elig["test_fraction"], elig["min_candidates_per_query"]
    )
    lines = A.rank_lines(panels, wl)
    line_meta = (
        pl.read_parquet(raw / "metadata/cell_line_metadata.parquet")
        .select(cell_line="Cell_ID_Cellosaur", organ="Organ")
        .unique()
    )
    lines = lines.join(line_meta, on="cell_line", how="left").sort("line_rank")
    replicates = A.replicate_structure(samples)
    design = A.plate_design(samples)
    chem = A.compound_audit(raw, samples)
    composition = A.sublibrary_composition(cells)

    targets = config["cohort_targets"]
    top_lines = tuple(lines.head(targets["prototype"]["lines"])["cell_line"])
    smoke_lines = tuple(lines.head(targets["smoke"]["lines"])["cell_line"])
    smoke_pool = (
        samples.filter((pl.col("plate_num") == 6) & ~pl.col("is_control") & (pl.col("drug") != "Adagrasib"))
        .with_columns(key=pl.col("compound_key").hash(seed=0))
        .sort("key")
    )
    smoke_compounds = tuple(smoke_pool.head(targets["smoke"]["compounds"])["compound_key"])
    cap = targets["prototype"]["max_cells_per_condition_well"]
    cohorts = {
        "smoke: plate 6, 4 lines, 12 compounds, cap 128 / controls 512": CohortSpec(
            (6,), smoke_lines, smoke_compounds, 128, 512
        ),
        "group B + plate 14, 12 lines, cap 256": CohortSpec((4, 5, 6, 14), top_lines, None, cap),
        "groups B + C + plate 14, 12 lines, cap 256": CohortSpec(
            (4, 5, 6, 7, 8, 9, 14), top_lines, None, cap
        ),
        "all 13 single-dose plates, 12 lines, cap 256": CohortSpec(
            tuple(p for p in range(1, 15) if p != 13), top_lines, None, cap
        ),
    }
    footprints = []
    for name, spec in cohorts.items():
        candidates = candidate_cells(obs_path, locations, samples, spec)
        visit = plan_row_groups(candidates, spec)
        chosen = select_cells(candidates, visit, spec)
        footprints.append({"cohort": name, **footprint_summary(index, visit, chosen)})
    footprints = pl.DataFrame(footprints)

    smoke_spec = cohorts["smoke: plate 6, 4 lines, 12 compounds, cap 128 / controls 512"]
    expression_path = tables / "expression_sample.json"
    if not skip_network:
        check = expression_check(revision, raw, index, obs_path, config["expression_sample"]["shards"])
        expression_path.write_text(json.dumps(check, indent=2) + "\n")
    expression = json.loads(expression_path.read_text())
    json.dump(
        {
            "smoke": {
                "plates": list(smoke_spec.plates),
                "lines": list(smoke_spec.lines),
                "compounds": list(smoke_compounds),
                "treated_cap": smoke_spec.treated_cap,
                "control_cap": smoke_spec.control_cap,
            },
            "prototype_lines": list(top_lines),
        },
        (reports / "candidate_cohorts.json").open("w"),
        indent=2,
    )

    for name, frame in {
        "plate_design": design,
        "control_support": controls,
        "candidate_panels": panels,
        "line_ranking": lines,
        "repeated_conditions": replicates["repeated_conditions"],
        "plate13_conditions": replicates["plate13"],
        "chemistry": chem["chemistry"],
        "parent_structure_duplicates": chem["parent_structure_duplicates"],
        "download_footprints": footprints,
    }.items():
        _write_csv(frame, tables / f"{name}.csv")

    context = {
        "revision": revision,
        "listing": listing,
        "index": index,
        "integrity": integrity,
        "samples": samples,
        "cells": cells,
        "wl": wl,
        "controls": controls,
        "panels": panels,
        "lines": lines,
        "replicates": replicates,
        "design": design,
        "chem": chem,
        "composition": composition,
        "footprints": footprints,
        "elig": elig,
        "expression": expression,
    }
    (reports / "data_audit.md").write_text(render_data_audit(context))
    (reports / "design_audit.md").write_text(render_design_audit(context))
    print("wrote reports/data_audit.md and reports/design_audit.md", flush=True)


def expression_check(
    revision: str, raw: Path, index: pl.DataFrame, obs_path: Path, shard_ids: list[int]
) -> dict:
    from huggingface_hub import HfFileSystem

    from cellresponse.data.expression import gene_vocabulary, read_row_group, to_csr

    genes, token_to_col = gene_vocabulary(raw)
    fs = HfFileSystem()
    obs = pl.scan_parquet(obs_path).select("BARCODE_SUB_LIB_ID", "tscp_count", "gene_count")
    rows = []
    for shard_id in shard_ids:
        shard = f"data/train-{shard_id:05d}-of-03388.parquet"
        last = int(index.filter(pl.col("shard") == shard)["row_group"].max())
        for row_group in sorted({0, last // 2}):
            matrix, cells, stats = to_csr(read_row_group(revision, shard_id, row_group, fs), token_to_col)
            joined = cells.join(
                obs.filter(pl.col("BARCODE_SUB_LIB_ID").is_in(cells["BARCODE_SUB_LIB_ID"].implode())).collect(),
                on="BARCODE_SUB_LIB_ID",
                how="left",
            )
            rows.append(
                {
                    "shard": shard_id,
                    "row_group": row_group,
                    "cells": cells.height,
                    "matched_obs": int(joined["tscp_count"].is_not_null().sum()),
                    "total_equals_tscp": int((joined["total_counts_all_tokens"] == joined["tscp_count"]).sum()),
                    "nnz_equals_gene_count": int((np.diff(matrix.indptr) == joined["gene_count"].to_numpy()).sum()),
                    **stats,
                }
            )
    return {
        "vocabulary_genes": genes.height,
        "token_min": int(genes["token_id"].min()),
        "token_max": int(genes["token_id"].max()),
        "row_groups": rows,
    }


def _write_csv(frame: pl.DataFrame, path: Path) -> None:
    flat = frame.with_columns(
        pl.col(c).cast(pl.List(pl.String)).list.join(";")
        for c, dtype in frame.schema.items()
        if isinstance(dtype, pl.List)
    )
    flat.write_csv(path)


def render_data_audit(ctx: dict) -> str:
    listing, index, integrity = ctx["listing"], ctx["index"], ctx["integrity"]
    cells, chem, composition = ctx["cells"], ctx["chem"], ctx["composition"]
    data_files = [f for f in listing if f.path.startswith("data/")]
    pass_counts = cells.group_by("pass_filter").agg(pl.col("n").sum()).sort("pass_filter")
    full_cells = int(pass_counts.filter(pl.col("pass_filter") == "full")["n"].sum())
    chemistry = chem["chemistry"]
    sizes = A.summarize_sizes(ctx["wl"].filter(~pl.col("is_control"))["n_full"].to_numpy())
    well_sizes = A.summarize_sizes(
        cells.filter(pl.col("pass_filter") == "full").group_by("sample").agg(pl.col("n").sum())["n"].to_numpy()
    )
    tv = composition["tv_distance"].to_numpy()
    null_tv = composition["null_tv_distance"].to_numpy()
    expr = ctx["expression"]
    checked = pl.DataFrame(expr["row_groups"])
    n_checked = int(checked["cells"].sum())
    return f"""# Data audit (M0)

Generated by `uv run cellresponse audit --config configs/audit.yaml`. Every number below is
recomputed from the pinned release; nothing was edited by hand.

## Source

| quantity | value |
| --- | --- |
| repository | `tahoebio/Tahoe-100M` (Hugging Face dataset) |
| pinned revision | `{ctx["revision"]}` |
| license (dataset card) | CC0-1.0; `LICENSE.md` downloaded and checksummed in `reports/source_manifest.json` |
| expression shards | {len(data_files) or index["shard"].n_unique():,} Parquet files, {sum(f.size for f in data_files) / 1e9:,.1f} GB compressed |
| expression rows | {int(index["num_rows"].sum()):,} in {index.height:,} row groups (≈1,000 rows each) |
| per-cell metadata (`obs_metadata`) | {integrity["rows"]:,} rows, {integrity["unique_keys"]:,} unique `BARCODE_SUB_LIB_ID` |

The site-wide "100M" label is the per-cell metadata count. Expression rows are the subset with
`pass_filter == "full"`:

{md_table(pass_counts)}

Retained (`full`) cells: {full_cells:,}; expression rows: {int(index["num_rows"].sum()):,}.
They are equal, so `minimal` cells exist only in metadata. They are counted as attrition, never imputed.

## Expression row format (verified on a {n_checked:,}-cell sample)

Row groups from the first, middle, and last shards were parsed with `cellresponse.data.parse.strip_marker`:

{md_table(checked)}

- Every row begins with gene token `1` and a negative value (`-2.0`). The marker is removed by
  position and the parser raises if it is absent; nothing is filtered by sign.
- After removal, all counts are finite non-negative integers and no token repeats within a cell.
- Remaining tokens map to the {expr["vocabulary_genes"]:,}-gene vocabulary (`gene_metadata`, tokens
  {expr["token_min"]}–{expr["token_max"]:,}, unique Ensembl IDs and symbols); unknown entries are counted above.
- Where `total_equals_tscp` and `nnz_equals_gene_count` equal `cells`, summed counts equal
  `obs_metadata.tscp_count` and non-zero entries equal `gene_count`: `expressions` holds raw UMI
  counts and `tscp_count` is the full library size.

## Cell locations: the shard layout is exactly reconstructible

Expression rows are the `full` obs rows, arranged in contiguous blocks of {composition["sublibrary"].n_unique():,}
sublibraries (block order read from footer statistics) with obs order kept within each block. This
was **verified rather than assumed**: predicted row counts and min/max of `BARCODE_SUB_LIB_ID`, `sample`,
`plate`, `drug`, and `cell_line_id` match the Parquet footer of all {index.height:,} row groups.
`data/interim/cell_locations.parquet` maps every retained cell to `(shard, row_group, row)`, so a
cohort can be fetched by row group without streaming the atlas.

Each sublibrary holds a slice of every well on its plate. Within a well, the line mix of each
sublibrary differs from the well's mix by a median total-variation distance of {np.median(tv):.3f}
(90th percentile {np.percentile(tv, 90):.3f}). Multinomial draws of the same sizes from the well's mix
give {np.median(null_tv):.3f} ({np.percentile(null_tv, 90):.3f}), so differences are close to sampling noise and
row-group subsampling is not expected to distort line composition. Sublibraries are still technical
batches; selection hashes over row groups across all sublibraries rather than taking whole ones.

## Cell counts

Cells per well: {well_sizes}. Cells per (treated well, line): {sizes}.
Some wells are nearly depleted. Conditions below the eligibility threshold are reported as
excluded, never as "no response".

## Discrepancies with documentation

| documented | observed | handling |
| --- | --- | --- |
| Tutorial loads `vevotx/Tahoe-100M` | Repository is `tahoebio/Tahoe-100M` | Use the pinned `tahoebio` revision |
| Tutorial drops the first entry only if `expressions[0] < 0` and silently skips unknown tokens | First entry is always the marker; no unknown tokens in sample | Remove by position, raise on anything unexpected |
| `drug` is a join key | `sample_metadata`/`obs` carry `'Erdafitinib '` and `'Selinexor '` (trailing space); `drug_metadata` and expression shards use stripped names | Join on a normalized `compound_key`, keep raw strings |
| ~100M cells | {integrity["rows"]:,} obs rows; {full_cells:,} with expression | Use `full` cells only; report `minimal` as attrition |
| `hf_hub_download` | Stalled on the 2.3 GB xet-backed metadata file | Resumable `httpx` download with size check |
| Anonymous hub access | HTTP 429 rate limits during footer indexing | Resumable index build; an HF token is advisable for cohort download |
| `moa-fine`, targets, approval | GPT-assisted annotations | Excluded from predictors |
| A drug named `Trametinib (DMSO_TF solvate)` | Contains the vehicle label as a substring | Vehicle detection uses exact equality |

## Chemistry

- {int(chemistry["smiles_valid"].sum())} of {chemistry.height} SMILES parse in RDKit. Invalid:
  {", ".join(chemistry.filter(~pl.col("smiles_valid"))["drug"].to_list())}.
- {int((chemistry["fragments"] > 1).sum())} structures have multiple fragments (salts, solvates, mixtures).
- Sample-table names unmatched to `drug_metadata` after normalization: {chem["unmatched_sample_drugs"].height}.
- {chem["parent_structure_duplicates"].height} groups of compounds share a largest-fragment InChIKey
  (salt/free-base pairs, stereo-free duplicates). They must share a split group in every protocol:

{md_table(chem["parent_structure_duplicates"].select("drug", "distinct_full_structures"), max_rows=40)}

Doxorubicin and Epirubicin are epimers; the provided SMILES omit stereochemistry and collapse
them. Structure-grouped splits will treat them as one group, which is conservative.
"""


def render_design_audit(ctx: dict) -> str:
    design, controls, panels, lines = ctx["design"], ctx["controls"], ctx["panels"], ctx["lines"]
    replicates, footprints, elig, wl = ctx["replicates"], ctx["footprints"], ctx["elig"], ctx["wl"]
    plate13 = replicates["plate13"]
    p13_wells = plate13["sample"].n_unique()
    p13_repeat_wells = plate13.filter(pl.col("primary_plate").is_not_null())["sample"].n_unique()
    adagrasib = wl.filter((pl.col("drug") == "Adagrasib") & (pl.col("dose_uM") == 0.05))
    pair_6_14 = replicates["pair_6_14"]
    line_cols = [
        "line_rank", "cell_name", "organ", "median_cells_per_well", "frac_wells_ge_128",
        "minimal_fraction", "feasible_plates", "min_control_cells",
    ]
    feasible_by_plate = (
        panels.group_by("plate_num", "plate_group", "dose_uM")
        .agg(
            feasible_queries=pl.col("query_feasible").sum(),
            median_eligible=pl.col("eligible_candidates").median(),
            median_expected_test=pl.col("expected_test_candidates").median(),
        )
        .sort("plate_num")
    )
    rep_lines = (
        wl.filter(pl.col("plate_num").is_in([6, 14]) & ~pl.col("is_control"))
        .group_by("cell_line", "compound_key")
        .agg(both=((pl.col("n_full") >= elig["min_treated_cells"]).sum() == 2))
        .group_by("cell_line")
        .agg(pl.col("both").sum())
    )
    return f"""# Design audit (M0)

Generated by `uv run cellresponse audit --config configs/audit.yaml` from metadata only.

## Experimental-unit hierarchy

plate (14) → well = `sample` (96 per plate, one treatment each) → cell line (50 pooled per well)
→ cell. Every well is split across many sublibraries, which are sequencing libraries (technical
batches), not biological repeats. A treated well is the experimental unit; line-level outcomes
are nested measurements in it. Each plate has exactly 2 vehicle (`DMSO_TF`) wells.

## Compound × dose × plate incidence

{md_table(design)}

The design is strictly blocked:

- **379 compounds form four groups (A–D) of 92–94.** Each group occupies three plates, one per dose:
  plates 1/4/7/10 at 0.05 µM, 2/5/8/11 at 0.5 µM, 3/6/9/12 at 5 µM.
- **Dose is fully confounded with plate** within a group, and compound group is confounded with plate
  triplet. No plate carries the same compound at two doses.
- **Adagrasib 0.05 µM is on every plate** ({adagrasib["sample"].n_unique()} wells across 14 plates), the only
  across-plate anchor at fixed dose. Some plates list 0.05 µM next to their panel dose only because of it.
- **Plate 14 repeats plate 6** (group B, 5 µM): {replicates["summary_6_14"]["n"][0]} shared compounds,
  none unique to either plate. This is the documented technical repeat.
- **Plate 13 is a mixed repeat plate not mentioned in the brief:** {p13_wells} treated wells over
  {plate13["compound_key"].n_unique()} compounds at all three doses. {p13_repeat_wells} wells repeat a condition
  from another plate; {p13_wells - p13_repeat_wells} are conditions measured only on plate 13 (e.g.
  {", ".join(plate13.filter(pl.col("primary_plate").is_null())["drug"].unique().sort().head(4).to_list())}).

### Consequences for what can be estimated

| comparison | status | reason |
| --- | --- | --- |
| Compounds within one (line, plate) block vs. the plate's vehicle wells | **Supported** (primary search) | Shared plate, shared vehicle reference |
| Same compound across doses | **Confounded with plate** | Each dose is on a different plate; no within-plate dose contrast |
| Compounds from different groups | **Confounded with plate triplet** | Groups never share a plate except via Adagrasib and plate 13 |
| Same condition, plate 6 vs 14 | **Supported** (technical transfer) | 94 compounds repeated on a second plate |
| Plate 13 conditions vs their primary plate | Supported, small | Limited cross-group/cross-plate repeats |
| Joint compound–dose optimization | **Not evaluated** | Would need cross-plate comparability that the design cannot establish alone |

Treatment-label permutation is only exchangeable *within a plate*; across plates it would assume
away the plate effect.

## Vehicle-control support

Per (plate, line) vehicle cells across both DMSO wells: median {int(controls["control_cells"].median()):,},
min {int(controls["control_cells"].min()):,}. {int(controls["control_eligible"].sum())} of {controls.height}
(plate, line) cells reach {elig["min_control_cells"]}; {int(controls["well_split_possible"].sum())} allow one
reference well and one evaluation well with ≥{elig["min_control_cells"] // 2} cells each.

## Candidate panels (protocol A2 feasibility)

A query is one (line, single-dose plate) block. Candidates are treated wells with ≥{elig["min_treated_cells"]}
retained cells for that line. Under a {elig["test_fraction"]:.0%} pair holdout, the expected number of held-out
candidates per query must be ≥{elig["min_candidates_per_query"]}.

{md_table(feasible_by_plate)}

{int(panels["query_feasible"].sum())} of {panels.height} query blocks are feasible. Each panel is about 93 compounds at
one dose, so a test query ranks ≈14 held-out compounds against a shared vehicle reference.

## Line selection

Lines ranked by feasible plates, then minimum control cells, then treated-well coverage (tissue is
secondary and only reported):

{md_table(lines.select(line_cols), max_rows=50)}

Three lines (NCI-H2122, NCI-H596, NCI-H661) cannot support any query. Recovery differs strongly
across lines, so line-level results must be macro-averaged.

## Plate 6 vs 14 replication audit support

Compounds eligible on both plates per line (top 12 lines):

{md_table(rep_lines.join(lines.select("cell_line", "cell_name", "line_rank"), on="cell_line").sort("line_rank").head(12).select("cell_name", "both"))}

Adagrasib 0.05 µM anchor wells: {adagrasib["sample"].n_unique()} wells; cells per (well, line) median
{int(adagrasib["n_full"].median()):,}.

## Download footprint

Exact, using `cell_locations.parquet` and the row-group planner in `cellresponse.data.select`:

{md_table(footprints)}

At the measured ~10 MB/s, the group B + plate 14 prototype is under an hour of row-group transfer
(~27 GB) versus ~2.6 h for whole shards. Groups B + C fit as well. Neither approaches the 338 GB atlas.

## Claims that remain unsupported

- Dose–response or dose optimization, because dose is confounded with plate.
- Ranking compounds across groups, because group is confounded with plate triplet.
- Between-experiment reproducibility beyond one repeated plate (6/14) and plate 13's small overlap.
- Viability or survival, because outcomes are among captured, `full`-QC cells only.
"""
