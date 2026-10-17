"""M0 data and design audit computed from Tahoe metadata and footer statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cellresponse.data.parse import VEHICLE, normalize_compound_name, parse_drugname_drugconc

ANCHOR_PLATE_GROUPS = {
    **{p: "A" for p in (1, 2, 3)},
    **{p: "B" for p in (4, 5, 6, 14)},
    **{p: "C" for p in (7, 8, 9)},
    **{p: "D" for p in (10, 11, 12)},
    13: "mixed",
}


def plate_number(column: str = "plate") -> pl.Expr:
    return pl.col(column).str.extract(r"(\d+)$").cast(pl.Int32)


def load_samples(raw: Path) -> pl.DataFrame:
    samples = pl.read_parquet(raw / "metadata/sample_metadata.parquet")
    parsed = [parse_drugname_drugconc(s) for s in samples["drugname_drugconc"]]
    if any(len(p) != 1 for p in parsed):
        raise ValueError("expected exactly one treatment per sample; combinations need a new schema")
    return samples.with_columns(
        plate_num=plate_number(),
        treatment_name=pl.Series([p[0].name for p in parsed]),
        dose_uM=pl.Series([p[0].dose_uM for p in parsed]),
        dose_unit_raw=pl.Series([p[0].unit for p in parsed]),
        is_control=pl.col("drug") == VEHICLE,
        compound_key=pl.col("drug").map_elements(normalize_compound_name, return_dtype=pl.String),
    ).with_columns(
        plate_group=pl.col("plate_num").replace_strict(ANCHOR_PLATE_GROUPS, return_dtype=pl.String),
    )


def aggregate_cells(raw: Path, out: Path) -> pl.DataFrame:
    """Per (plate, sample, line, sublibrary, pass_filter) cell counts from the 100M-row obs table."""
    if out.exists():
        return pl.read_parquet(out)
    counts = (
        pl.scan_parquet(raw / "metadata/obs_metadata.parquet")
        .group_by("plate", "sample", "cell_line", "cell_name", "sublibrary", "pass_filter")
        .agg(
            n=pl.len(),
            tscp_sum=pl.col("tscp_count").sum(),
            genes_sum=pl.col("gene_count").sum(),
            mito_sum=pl.col("pcnt_mito").cast(pl.Float64).sum(),
        )
        .with_columns(pl.col(c).cast(pl.String) for c in ("cell_line", "cell_name", "pass_filter"))
        .collect(engine="streaming")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    counts.write_parquet(out)
    return counts


def obs_integrity(raw: Path) -> dict:
    lf = pl.scan_parquet(raw / "metadata/obs_metadata.parquet")
    row = lf.select(
        rows=pl.len(),
        unique_keys=pl.col("BARCODE_SUB_LIB_ID").n_unique(),
        null_line=pl.col("cell_line").is_null().sum(),
        null_sample=pl.col("sample").is_null().sum(),
        drug_mismatch_rows=pl.len() - pl.col("drug").is_not_null().sum(),
    ).collect(engine="streaming")
    return row.to_dicts()[0]


def well_line_counts(cells: pl.DataFrame, samples: pl.DataFrame) -> pl.DataFrame:
    """Retained (`full`) and excluded (`minimal`) cells per well and line."""
    wide = (
        cells.group_by("sample", "cell_line", "cell_name")
        .agg(
            n_full=pl.col("n").filter(pl.col("pass_filter") == "full").sum(),
            n_minimal=pl.col("n").filter(pl.col("pass_filter") == "minimal").sum(),
            tscp_full=pl.col("tscp_sum").filter(pl.col("pass_filter") == "full").sum(),
        )
        .with_columns(mean_tscp_full=pl.col("tscp_full") / pl.col("n_full"))
    )
    keys = samples.select(
        "sample", "plate", "plate_num", "plate_group", "drug", "compound_key", "dose_uM", "is_control"
    )
    return wide.join(keys, on="sample", how="left", validate="m:1")


def control_support(wl: pl.DataFrame, min_control: int) -> pl.DataFrame:
    controls = wl.filter(pl.col("is_control"))
    per_well = controls.sort("sample").group_by("plate_num", "cell_line").agg(
        control_wells=pl.col("sample"),
        control_cells_per_well=pl.col("n_full"),
        control_cells=pl.col("n_full").sum(),
    )
    return per_well.with_columns(
        min_well_cells=pl.col("control_cells_per_well").list.min(),
        control_eligible=pl.col("control_cells") >= min_control,
        well_split_possible=pl.col("control_cells_per_well").list.min() >= min_control // 2,
    )


def candidate_panels(
    wl: pl.DataFrame, controls: pl.DataFrame, min_treated: int, test_fraction: float, min_candidates: int
) -> pl.DataFrame:
    """Per (line, single-dose plate) query block: eligible treated candidates sharing one reference."""
    treated = wl.filter(~pl.col("is_control") & (pl.col("plate_num") != 13))
    panels = (
        treated.group_by("plate_num", "plate_group", "cell_line", "cell_name")
        .agg(
            dose_uM=pl.col("dose_uM").filter(pl.col("drug") != "Adagrasib").mode().first(),
            wells=pl.len(),
            eligible_candidates=(pl.col("n_full") >= min_treated).sum(),
            median_treated_cells=pl.col("n_full").median(),
        )
        .join(controls, on=["plate_num", "cell_line"], how="left")
        .with_columns(
            expected_test_candidates=(pl.col("eligible_candidates") * test_fraction).floor(),
        )
        .with_columns(
            query_feasible=(pl.col("expected_test_candidates") >= min_candidates)
            & pl.col("control_eligible").fill_null(False),
        )
    )
    return panels.sort("plate_num", "cell_line")


def rank_lines(panels: pl.DataFrame, wl: pl.DataFrame) -> pl.DataFrame:
    treated = wl.filter(~pl.col("is_control"))
    per_line = treated.group_by("cell_line", "cell_name").agg(
        median_cells_per_well=pl.col("n_full").median(),
        frac_wells_ge_128=(pl.col("n_full") >= 128).mean(),
        minimal_fraction=pl.col("n_minimal").sum() / (pl.col("n_full") + pl.col("n_minimal")).sum(),
    )
    feasibility = panels.group_by("cell_line").agg(
        feasible_plates=pl.col("query_feasible").sum(),
        min_control_cells=pl.col("control_cells").min(),
    )
    return (
        per_line.join(feasibility, on="cell_line", how="left")
        .sort(["feasible_plates", "min_control_cells", "frac_wells_ge_128"], descending=True)
        .with_row_index("line_rank", offset=1)
    )


def replicate_structure(samples: pl.DataFrame) -> dict[str, pl.DataFrame]:
    treated = samples.filter(~pl.col("is_control"))
    condition_wells = treated.group_by("compound_key", "dose_uM").agg(
        drug=pl.col("drug").first(),
        n_wells=pl.len(),
        plates=pl.col("plate_num").sort(),
        samples=pl.col("sample").sort_by("plate_num"),
    )
    repeated = condition_wells.filter(pl.col("n_wells") > 1)
    p6 = set(treated.filter(pl.col("plate_num") == 6)["compound_key"])
    p14 = set(treated.filter(pl.col("plate_num") == 14)["compound_key"])
    pair_6_14 = repeated.filter(
        pl.col("plates").list.contains(6) & pl.col("plates").list.contains(14)
    )
    plate13 = treated.filter(pl.col("plate_num") == 13).join(
        treated.filter(pl.col("plate_num") != 13).select(
            "compound_key", "dose_uM", primary_plate="plate_num", primary_sample="sample"
        ),
        on=["compound_key", "dose_uM"],
        how="left",
    )
    summary = pl.DataFrame(
        {
            "comparison": ["plate 6 vs 14 compounds", "plate 6 only", "plate 14 only"],
            "n": [len(p6 & p14), len(p6 - p14), len(p14 - p6)],
        }
    )
    return {
        "repeated_conditions": repeated.sort("n_wells", "drug", descending=[True, False]),
        "pair_6_14": pair_6_14,
        "plate13": plate13.sort("primary_plate", "drug"),
        "summary_6_14": summary,
    }


def plate_design(samples: pl.DataFrame) -> pl.DataFrame:
    """Plate layout; the Adagrasib 0.05 uM anchor well is counted separately from the panel."""
    panel = ~pl.col("is_control") & (pl.col("drug") != "Adagrasib")
    return (
        samples.group_by("plate_num", "plate_group")
        .agg(
            wells=pl.len(),
            vehicle_wells=pl.col("is_control").sum(),
            anchor_wells=(pl.col("drug") == "Adagrasib").sum(),
            panel_compounds=pl.col("compound_key").filter(panel).n_unique(),
            panel_wells=panel.sum(),
            panel_doses_uM=pl.col("dose_uM").filter(panel).unique().sort(),
        )
        .sort("plate_num")
    )


def compound_audit(raw: Path, samples: pl.DataFrame) -> dict[str, pl.DataFrame]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    drugs = pl.read_parquet(raw / "metadata/drug_metadata.parquet")
    chooser = rdMolStandardize.LargestFragmentChooser()
    rows = []
    for name, smiles, cid in drugs.select("drug", "canonical_smiles", "pubchem_cid").iter_rows():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        fragments = len(Chem.GetMolFrags(mol)) if mol is not None else None
        parent = chooser.choose(mol) if mol is not None else None
        rows.append(
            {
                "drug": name,
                "compound_key": normalize_compound_name(name),
                "smiles_valid": mol is not None,
                "fragments": fragments,
                "heavy_atoms_parent": parent.GetNumHeavyAtoms() if parent is not None else None,
                "inchikey_full": Chem.MolToInchiKey(mol) if mol is not None else None,
                "inchikey_parent": Chem.MolToInchiKey(parent) if parent is not None else None,
                "pubchem_cid": cid,
            }
        )
    chem = pl.DataFrame(rows)
    sample_names = samples.filter(~pl.col("is_control")).select("drug", "compound_key").unique()
    raw_name_mismatches = sample_names.join(chem.select("drug"), on="drug", how="anti")
    unmatched = sample_names.join(chem.select("compound_key"), on="compound_key", how="anti")
    key_collisions = chem.group_by("compound_key").agg(pl.col("drug")).filter(pl.col("drug").list.len() > 1)
    parent_duplicates = (
        chem.filter(pl.col("inchikey_parent").is_not_null())
        .group_by("inchikey_parent")
        .agg(pl.col("drug"), pl.col("inchikey_full").n_unique().alias("distinct_full_structures"))
        .filter(pl.col("drug").list.len() > 1)
    )
    treatment_name_variants = (
        samples.filter(~pl.col("is_control"))
        .filter(pl.col("treatment_name") != pl.col("drug"))
        .select("drug", "treatment_name")
        .unique()
    )
    return {
        "chemistry": chem,
        "raw_name_mismatches": raw_name_mismatches,
        "unmatched_sample_drugs": unmatched,
        "name_key_collisions": key_collisions,
        "parent_structure_duplicates": parent_duplicates,
        "treatment_name_variants": treatment_name_variants,
    }


def sublibrary_composition(cells: pl.DataFrame) -> pl.DataFrame:
    """Total-variation distance between each sublibrary's line mix and its well's line mix."""
    full = cells.filter(pl.col("pass_filter") == "full")
    well = full.group_by("sample", "cell_line").agg(n_well=pl.col("n").sum())
    well = well.with_columns(p_well=pl.col("n_well") / pl.col("n_well").sum().over("sample"))
    sub = full.group_by("sample", "sublibrary", "cell_line").agg(n_sub=pl.col("n").sum())
    sub = sub.with_columns(
        total_sub=pl.col("n_sub").sum().over("sample", "sublibrary"),
    ).with_columns(p_sub=pl.col("n_sub") / pl.col("total_sub"))
    joined = sub.join(well.select("sample", "cell_line", "p_well"), on=["sample", "cell_line"])
    tv = joined.group_by("sample", "sublibrary").agg(
        cells=pl.col("total_sub").first(),
        tv_distance=(pl.col("p_sub") - pl.col("p_well")).abs().sum() / 2,
    )
    return tv.join(_multinomial_null_tv(well, tv), on=["sample", "sublibrary"], how="left")


def _multinomial_null_tv(well: pl.DataFrame, tv: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """TV distance expected if each sublibrary were a multinomial draw from its well's line mix."""
    rng = np.random.default_rng(seed)
    probabilities = {
        sample[0]: frame["p_well"].to_numpy() for sample, frame in well.sort("cell_line").group_by("sample")
    }
    rows = []
    for sample, sublibrary, cells in tv.select("sample", "sublibrary", "cells").iter_rows():
        p = probabilities[sample]
        draw = rng.multinomial(cells, p) / cells
        rows.append((sample, sublibrary, float(np.abs(draw - p).sum() / 2)))
    return pl.DataFrame(rows, schema=["sample", "sublibrary", "null_tv_distance"], orient="row")


def footprint(
    index: pl.DataFrame,
    wl: pl.DataFrame,
    samples: pl.DataFrame,
    plates: list[int],
    lines: list[str],
    cap_per_condition: int | None,
    compounds: list[str] | None = None,
    safety: float = 1.5,
) -> dict:
    """Bytes of expression row groups needed for a cohort, with row-group subsampling per well.

    Row groups are a superset filter: each group whose [sample_min, sample_max] range covers a
    selected well is counted. With a cap, only the fraction of a well's groups needed to reach the
    cap for its rarest selected line (times a safety factor) is counted; controls are never capped.
    """
    chosen = samples.filter(pl.col("plate_num").is_in(plates))
    if compounds is not None:
        chosen = chosen.filter(pl.col("is_control") | pl.col("compound_key").is_in(compounds))
    min_line_cells = (
        wl.filter(pl.col("cell_line").is_in(lines))
        .group_by("sample")
        .agg(min_line_cells=pl.col("n_full").min(), line_cells=pl.col("n_full").sum())
    )
    chosen = chosen.join(min_line_cells, on="sample", how="left")
    groups = index.select("sample_min", "sample_max", "compressed_bytes")
    total_bytes = 0.0
    full_bytes = 0.0
    group_hits = 0
    selected_cells = 0.0
    for sample, is_control, min_cells, line_cells in chosen.select(
        "sample", "is_control", "min_line_cells", "line_cells"
    ).iter_rows():
        hits = groups.filter((pl.col("sample_min") <= sample) & (pl.col("sample_max") >= sample))
        well_bytes = float(hits["compressed_bytes"].sum())
        full_bytes += well_bytes
        group_hits += hits.height
        fraction = 1.0
        if cap_per_condition is not None and not is_control and min_cells:
            fraction = min(1.0, safety * cap_per_condition / max(min_cells, 1))
        total_bytes += fraction * well_bytes
        cap_cells = line_cells or 0
        if cap_per_condition is not None and not is_control:
            cap_cells = min(cap_cells, cap_per_condition * len(lines))
        selected_cells += cap_cells
    return {
        "plates": plates,
        "wells": chosen.height,
        "lines": len(lines),
        "row_groups_touched": group_hits,
        "full_well_gb": full_bytes / 1e9,
        "estimated_download_gb": total_bytes / 1e9,
        "approx_selected_cells": int(selected_cells),
    }


def summarize_sizes(values: np.ndarray) -> dict:
    return {
        "min": int(values.min()),
        "p10": int(np.percentile(values, 10)),
        "median": int(np.median(values)),
        "p90": int(np.percentile(values, 90)),
        "max": int(values.max()),
    }
