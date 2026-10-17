"""Freeze the M1b benchmark contract: splits, queries, representation, and the gate."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cellresponse.data.report import md_table
from cellresponse.data.source import sha256sum
from cellresponse.features.fit import fit_representation, representation_gate
from cellresponse.goals import VERSION, goals
from cellresponse.queries import build_documents
from cellresponse.split import HELD_OUT_PLATE, assemble_splits

_ROLES = ("train", "validation", "test")


def run(config: dict) -> None:
    prepared = Path(config["paths"]["prepared"]) / config["name"]
    reports = Path("reports")
    query_dir = Path("queries")
    reports.mkdir(exist_ok=True)
    query_dir.mkdir(exist_ok=True)
    seed = int(config["cohort"]["seed"])

    cells = pl.read_parquet(
        prepared / "cells.parquet",
        columns=[
            "cell_id",
            "sample",
            "cell_line",
            "plate_num",
            "compound_key",
            "dose_uM",
            "is_control",
        ],
    )
    conditions = pl.read_parquet(prepared / "conditions.parquet")
    compounds = pl.read_parquet(prepared / "compounds.parquet")
    splits = assemble_splits(cells, conditions, compounds, seed)
    split_path = prepared / "splits.parquet"
    splits.write_parquet(split_path)
    split_hash = sha256sum(split_path)
    print(f"[split] {splits.height:,} cells -> {split_path}", flush=True)

    documents = build_documents(splits, conditions, seed, split_hash)
    for name, document in documents.items():
        path = query_dir / f"{name}.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        print(
            f"[queries] {len(document['queries'])} {name} queries, "
            f"{len(document['dropped_blocks'])} dropped blocks",
            flush=True,
        )
    (reports / "splits.md").write_text(_splits_report(splits, documents, split_hash, compounds))

    genes = pl.read_parquet(prepared / "genes.parquet")
    fit_a = fit_representation(
        prepared, splits, genes, "fit_a", seed, prepared / "representation_a.npz"
    )
    gate = representation_gate(prepared, splits, genes, fit_a)
    (prepared / "goals_a.json").write_text(json.dumps(gate, indent=2) + "\n")
    fit_c = fit_representation(
        prepared, splits, genes, "fit_c", seed, prepared / "representation_c.npz"
    )
    (reports / "representation.md").write_text(_representation_report(fit_a, fit_c, gate))
    decision = "PCA-64" if gate["decision"] == "pca64" else "direct module scores"
    print(f"[gate] decision: {decision}", flush=True)


def _splits_report(
    splits: pl.DataFrame, documents: dict, split_hash: str, compounds: pl.DataFrame
) -> str:
    pairs = (
        splits.filter(pl.col("protocol_a").is_in(list(_ROLES)))
        .select("cell_line", "structure_group", "protocol_a")
        .unique()
    )
    lines = [
        "# Splits",
        "",
        "Protocol A holds out `(cell line, structure group)` pairs. A structure group is the",
        "largest-fragment InChIKey, so a salt and its free base stay together.",
        "The hash is Blake2s of `seed`, `A`, the line, and the structure group, joined by NUL",
        "bytes. Values below 0.70 are train, below 0.85 validation, and the rest test. If a line",
        "or a structure group would otherwise have no training pair, its lowest-hash held-out pair",
        "moves to train. Every dose, well, and plate of a pair, plate 14 included, shares it.",
        "",
        "Protocol C hashes the structure group alone (`C` instead of `A`) and does not repair:",
        "a held-out group is absent from every line. Adagrasib and the two compounds with no",
        "fingerprint (`Sacubitril/Valsartan`, `Verteporfin`) are excluded from protocol C.",
        "Scaffold holdout is not a second protocol. The most common Bemis-Murcko scaffold in this",
        "cohort is benzene, shared by 17 compounds, and 7 compounds are acyclic, so a scaffold",
        "split would not be a chemical-novelty test.",
        "",
        "Each plate has two vehicle wells. The well with the smaller Blake2s hash is the",
        "reference population; the other is the evaluation population and is disjoint. Plate 14",
        "is outside both fit masks, including its reference well. Fit masks are training treated",
        "cells on the other plates, plus reference vehicle cells on those plates.",
        "",
        f"Split file sha256: `{split_hash}`. Seed `{splits['seed'][0]}`.",
        "",
        "## Protocol A pairs",
        "",
        _role_table(pairs, "protocol_a"),
        "",
        "## Protocol A by line",
        "",
        md_table(_spread(pairs, "cell_line", "protocol_a")),
        "",
        "## Protocol C groups",
        "",
        _role_table(
            splits.filter(pl.col("protocol_c").is_in(list(_ROLES)))
            .select("structure_group", "protocol_c")
            .unique(),
            "protocol_c",
        ),
        "",
        "## Plate 14 treated cells",
        "",
        md_table(_count_rows(splits.filter(pl.col("plate_num") == HELD_OUT_PLATE), "protocol_b")),
        "",
        "## Vehicle wells",
        "",
        md_table(
            splits.filter(pl.col("is_control"))
            .select("plate_num", "sample", "control_role")
            .unique()
            .sort("plate_num", "control_role", "sample")
        ),
        "",
        "## Queries",
        "",
    ]
    for name, document in documents.items():
        n_contexts = len(document["contexts"])
        lines.append(
            f"- `{name}`: {len(document['queries'])} queries over {n_contexts} contexts, "
            f"{len(document['dropped_blocks'])} blocks below the 10-candidate minimum."
        )
    lines += [
        "",
        "Query files name reference and evaluation cell ids, the panel, and the scoring rule.",
        "They do not contain expression, scores, or thresholds. Thresholds are fit from training",
        f"reference-vehicle cells when the representation is fit (`goals` version `{VERSION}`).",
        "",
        f"Compounds in the cohort: {compounds.height}. "
        f"Structure groups: {compounds['structure_group'].n_unique()}.",
        "",
    ]
    return "\n".join(lines)


def _representation_report(fit_a: dict, fit_c: dict, gate: dict) -> str:
    decision = gate["decision"]
    if decision == "pca64":
        outcome = (
            "The gate passes. Inverse search stays in the 64 principal components. "
            "Observed utility still uses the raw module scores, not the reconstruction."
        )
    else:
        outcome = (
            "The gate does not pass. Inverse search will model the four module scores directly. "
            "PCA-64 stays available for baselines that do not need the goal event. "
            "The test set is unchanged."
        )
    rows = []
    for goal in gate["goals"]:
        rows.append(
            {
                "goal": goal["goal_id"],
                "role": goal["role"],
                "genes_in_hvg": f"{goal['n_hvg_genes']}/{goal['n_vocabulary_genes']}",
                "threshold": None if goal["threshold"] is None else round(goal["threshold"], 4),
                "median_abs_delta": None
                if goal["median_abs_delta"] != goal["median_abs_delta"]
                else round(goal["median_abs_delta"], 4),
                "p90_abs_delta": None
                if goal["p90_abs_delta"] != goal["p90_abs_delta"]
                else round(goal["p90_abs_delta"], 4),
                "spearman": None
                if goal["median_spearman"] != goal["median_spearman"]
                else round(goal["median_spearman"], 4),
                "ranked_blocks": goal["n_ranked_blocks"],
                "passes": goal["passes"],
            }
        )
    lines = [
        "# Representation",
        "",
        outcome,
        "",
        "Fitting uses protocol-A training treated cells and reference vehicle cells on plates",
        "other than 14. Validation treated cells and plate 14 are not in the fit.",
        "Protocol C has its own HVG and PCA fit, with the same plate-14 exclusion;",
        "the gate below is the protocol-A validation gate, which is the primary benchmark.",
        "",
        fit_a["method"],
        "",
        "Counts are normalized to 10,000 per cell and transformed with log1p before PCA.",
        "PCA is centered, with 64 components, fit on at most 200,000 balanced cells.",
        f"PC scales are training standard deviations floored at {fit_a['pc_scale_floor']}.",
        "",
        "## Fits",
        "",
        md_table(
            pl.DataFrame(
                [
                    _fit_row(fit_a),
                    _fit_row(fit_c),
                ]
            )
        ),
        "",
        "## Gate",
        "",
        "A goal passes when at least half of its vocabulary genes are among the 2,000 HVGs,",
        "the median absolute difference of raw and reconstructed event fractions is at most 0.05,",
        "and the median Spearman correlation of those fractions across compounds, within a",
        "(line, plate) block of at least 10 compounds, is at least 0.8. Blocks with no variation",
        "are excluded from the correlation. The event threshold is the training reference-vehicle",
        "quantile of the raw score on the same gene subset.",
        "",
        md_table(pl.DataFrame(rows)),
        "",
        f"Decision recorded for later milestones: `{decision}`.",
        "",
    ]
    goal_notes = []
    for goal in goals():
        match = next(item for item in gate["goals"] if item["goal_id"] == goal.goal_id)
        if match["missing_genes"]:
            goal_notes.append(
                f"- {goal.goal_id} is missing {', '.join(match['missing_genes'])} "
                "from the vocabulary."
            )
    if goal_notes:
        lines += [
            "Genes from the Hallmark list that are not in the Tahoe vocabulary:",
            "",
            *goal_notes,
            "",
        ]
    return "\n".join(lines)


def _fit_row(fit: dict) -> dict:
    return {
        "fit": fit["fit_column"],
        "cells": fit["n_fit_cells"],
        "conditions": fit["n_conditions"],
        "hvg_cells": fit["n_hvg_cells"],
        "pca_cells": fit["n_pca_cells"],
        "variance_explained": fit["variance_explained"],
    }


def _role_table(frame: pl.DataFrame, column: str) -> str:
    counts = _count_rows(frame, column)
    total = int(counts["n"].sum()) if counts.height else 0
    fractions = counts.with_columns(fraction=(pl.col("n") / total).round(3)) if total else counts
    return md_table(fractions)


def _count_rows(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.group_by(column).len().sort(column).rename({"len": "n"})


def _spread(frame: pl.DataFrame, index: str, column: str) -> pl.DataFrame:
    rows = []
    for key, part in frame.group_by(index):
        name = key[0] if isinstance(key, tuple) else key
        counts = {role: part.filter(pl.col(column) == role).height for role in _ROLES}
        rows.append({index: name, **counts})
    return pl.DataFrame(rows).sort(index) if rows else pl.DataFrame()
