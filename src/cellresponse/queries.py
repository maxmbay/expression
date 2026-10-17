"""Query manifests. A query names a context, a panel, and a goal. It does not name an outcome.

Dev queries rank protocol-A validation compounds. Test queries rank protocol-A test
compounds. Plate 14 is a separate audit file: seen conditions and unseen pairs are
not placed on the same panel.
"""

from __future__ import annotations

import polars as pl

from cellresponse.goals import VERSION, Goal, goals
from cellresponse.split import (
    ANCHOR_COMPOUND,
    HELD_OUT_PLATE,
    LINE_NAMES,
    MIN_QUERY_CANDIDATES,
    MIN_TREATED_CELLS,
)

FORBIDDEN_KEYS = frozenset(
    {
        "expression",
        "expressions",
        "counts",
        "genes",
        "score",
        "scores",
        "utility",
        "outcome",
        "outcomes",
        "response",
        "S_score",
        "G2M_score",
        "phase",
        "fold_change",
        "event_fraction",
        "threshold",
        "prediction",
    }
)

SCORING_TEXT = {
    "score_definition": (
        "mean log1p expression after normalizing the cell to 10,000 counts, "
        "over the goal's genes that are present in the vocabulary; a missing gene is zero"
    ),
    "utility_definition": (
        "treated event fraction minus the evaluation-vehicle event fraction "
        "on the same line and plate; the vehicle call has utility zero"
    ),
    "ranking": "descending utility; doing nothing is a candidate with utility zero",
}


def check_query_document(document: dict) -> None:
    """Raise if a query carries an outcome or a panel smaller than the document's minimum."""
    _walk(document)
    minimum = int(document["min_candidates"])
    for query in document["queries"]:
        panel = query["panel"]
        if len(panel) < minimum:
            raise RuntimeError(f"{query['query_id']} has {len(panel)} candidates")
        keys = [item["compound_key"] for item in panel]
        if ANCHOR_COMPOUND in keys:
            raise RuntimeError(f"{query['query_id']} contains the anchor compound")
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"{query['query_id']} repeats a compound")
        if "reference_cell_ids" not in document["contexts"][query["context_id"]]:
            raise RuntimeError(f"{query['query_id']} has no reference cells")


def _walk(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise RuntimeError(f"query document contains forbidden key {key!r}")
            _walk(child)
    elif isinstance(value, list):
        for child in value:
            _walk(child)


def build_documents(
    splits: pl.DataFrame,
    conditions: pl.DataFrame,
    seed: int,
    splits_sha256: str,
    min_treated: int = MIN_TREATED_CELLS,
    min_candidates: int = MIN_QUERY_CANDIDATES,
) -> dict[str, dict]:
    """Return dev, test, and plate-14 documents. Membership comes from splits, not responses."""
    roles = (
        splits.filter(~pl.col("is_control"))
        .select(
            "sample", "cell_line", "compound_key", "structure_group", "protocol_a", "protocol_b"
        )
        .unique()
    )
    support = conditions.filter(~pl.col("is_control")).select(
        "sample",
        "cell_line",
        "plate",
        "plate_num",
        "compound_key",
        "drug",
        "dose_uM",
        "retained_cells",
    )
    candidates = support.join(
        roles, on=["sample", "cell_line", "compound_key"], how="inner"
    ).filter(
        (pl.col("compound_key") != ANCHOR_COMPOUND) & (pl.col("retained_cells") >= min_treated)
    )
    contexts = _contexts(splits)
    documents = {
        "dev": _panel_document(
            "dev",
            "A",
            "validation",
            candidates.filter(
                (pl.col("protocol_a") == "validation") & (pl.col("plate_num") != HELD_OUT_PLATE)
            ),
            contexts,
            seed,
            splits_sha256,
            min_treated,
            min_candidates,
        ),
        "test": _panel_document(
            "test",
            "A",
            "test",
            candidates.filter(
                (pl.col("protocol_a") == "test") & (pl.col("plate_num") != HELD_OUT_PLATE)
            ),
            contexts,
            seed,
            splits_sha256,
            min_treated,
            min_candidates,
        ),
        "plate14": _plate14_document(
            candidates.filter(pl.col("plate_num") == HELD_OUT_PLATE),
            contexts,
            seed,
            splits_sha256,
            min_treated,
            min_candidates,
        ),
    }
    for document in documents.values():
        check_query_document(document)
    return documents


def _role_cells(controls: pl.DataFrame, role: str, name: str) -> pl.DataFrame:
    return (
        controls.filter(pl.col("control_role") == role)
        .group_by("plate_num", "cell_line")
        .agg(
            pl.col("sample").unique().sort().alias(f"{name}_wells"),
            pl.col("cell_id").sort().alias(f"{name}_cell_ids"),
        )
    )


def _contexts(splits: pl.DataFrame) -> dict[str, dict]:
    controls = splits.filter(pl.col("is_control")).select(
        "plate_num", "cell_line", "sample", "control_role", "cell_id"
    )
    grouped = _role_cells(controls, "reference", "reference").join(
        _role_cells(controls, "evaluation", "evaluation"),
        on=["plate_num", "cell_line"],
        how="inner",
    )
    contexts = {}
    for row in grouped.sort("plate_num", "cell_line").iter_rows(named=True):
        if len(row["reference_wells"]) != 1 or len(row["evaluation_wells"]) != 1:
            raise RuntimeError(
                f"plate {row['plate_num']} line {row['cell_line']} needs one well of each role"
            )
        context_id = f"{row['cell_line']}-plate{row['plate_num']}"
        contexts[context_id] = {
            "context_id": context_id,
            "cell_line": row["cell_line"],
            "cell_line_name": LINE_NAMES.get(row["cell_line"], row["cell_line"]),
            "plate": f"plate{row['plate_num']}",
            "plate_num": int(row["plate_num"]),
            "endpoint_hours": 24,
            "vehicle": "DMSO_TF",
            "vehicle_meaning": "contemporaneous vehicle endpoint, not a pretreatment measurement",
            "treatment_history": "unknown",
            "reference_well": row["reference_wells"][0],
            "evaluation_well": row["evaluation_wells"][0],
            "reference_cell_ids": [int(cell_id) for cell_id in row["reference_cell_ids"]],
            "evaluation_cell_ids": [int(cell_id) for cell_id in row["evaluation_cell_ids"]],
        }
    return contexts


def _panel_document(
    name: str,
    protocol: str,
    split: str,
    candidates: pl.DataFrame,
    contexts: dict[str, dict],
    seed: int,
    splits_sha256: str,
    min_treated: int,
    min_candidates: int,
) -> dict:
    queries = []
    dropped = []
    held_out = HELD_OUT_PLATE if protocol == "A" else None
    grid = [
        (context["cell_line"], context["plate_num"])
        for context in contexts.values()
        if held_out is None or context["plate_num"] != held_out
    ]
    for line, plate in sorted(grid):
        block = candidates.filter((pl.col("cell_line") == line) & (pl.col("plate_num") == plate))
        if block.height == 0:
            dropped.append(
                {
                    "cell_line": line,
                    "plate_num": plate,
                    "split": split,
                    "n_candidates": 0,
                    "reason": f"fewer than {min_candidates} candidates",
                }
            )
            continue
        _consume_block(
            queries,
            dropped,
            block,
            contexts,
            protocol,
            split,
            f"{protocol}-{split}",
            min_candidates,
        )
    return _document(
        name,
        protocol,
        split,
        queries,
        dropped,
        contexts,
        seed,
        splits_sha256,
        min_treated,
        min_candidates,
    )


def _plate14_document(
    candidates, contexts, seed, splits_sha256, min_treated, min_candidates
) -> dict:
    """Separate panels for conditions seen on another plate and pairs withheld everywhere."""
    queries = []
    dropped = []
    grid = sorted(
        (context["cell_line"], context["plate_num"])
        for context in contexts.values()
        if context["plate_num"] == HELD_OUT_PLATE
    )
    families = (
        (
            "seen_condition_new_plate",
            "seen_condition_new_plate",
            pl.col("protocol_b") == "seen_condition_new_plate",
        ),
        ("unseen_validation", "validation", pl.col("protocol_a") == "validation"),
        ("unseen_test", "test", pl.col("protocol_a") == "test"),
    )
    for family, split, predicate in families:
        subset = candidates.filter(predicate)
        for line, plate in grid:
            block = subset.filter((pl.col("cell_line") == line) & (pl.col("plate_num") == plate))
            if block.height == 0:
                dropped.append(
                    {
                        "cell_line": line,
                        "plate_num": plate,
                        "split": split,
                        "n_candidates": 0,
                        "reason": f"fewer than {min_candidates} candidates",
                    }
                )
                continue
            _consume_block(
                queries, dropped, block, contexts, "B", split, f"B-{family}", min_candidates
            )
    return _document(
        "plate14",
        "B",
        "audit",
        queries,
        dropped,
        contexts,
        seed,
        splits_sha256,
        min_treated,
        min_candidates,
    )


def _consume_block(
    queries, dropped, block, contexts, protocol, split, id_prefix, min_candidates
) -> None:
    line = block["cell_line"][0]
    plate = int(block["plate_num"][0])
    context_id = f"{line}-plate{plate}"
    doses = block["dose_uM"].unique().to_list()
    if len(doses) != 1:
        raise RuntimeError(f"{context_id} panel mixes doses {doses}")
    panel = [
        {
            "compound_key": row["compound_key"],
            "drug": row["drug"],
            "structure_group": row["structure_group"],
            "sample": row["sample"],
            "dose_uM": float(row["dose_uM"]),
            "retained_cells": int(row["retained_cells"]),
        }
        for row in block.sort("compound_key").iter_rows(named=True)
    ]
    if len(panel) < min_candidates:
        dropped.append(
            {
                "cell_line": line,
                "plate_num": plate,
                "split": split,
                "n_candidates": len(panel),
                "reason": f"fewer than {min_candidates} candidates",
            }
        )
        return
    context = dict(contexts[context_id])
    context["dose_uM"] = float(doses[0])
    contexts[context_id] = context
    for goal in goals():
        queries.append(
            {
                "query_id": f"{id_prefix}-{line}-plate{plate}-{goal.goal_id}",
                "protocol": protocol,
                "split": split,
                "context_id": context_id,
                "goal_id": goal.goal_id,
                "panel": panel,
                "scoring": _scoring(goal),
            }
        )


def _scoring(goal: Goal) -> dict:
    return {
        "goal_id": goal.goal_id,
        "goal_version": VERSION,
        "direction": goal.direction,
        "role": goal.role,
        "quantile": goal.quantile,
        "event": goal.event_rule(),
        **SCORING_TEXT,
    }


def _document(
    name,
    protocol,
    split,
    queries,
    dropped,
    contexts,
    seed,
    splits_sha256,
    min_treated,
    min_candidates,
) -> dict:
    used = {query["context_id"] for query in queries}
    return {
        "name": name,
        "protocol": protocol,
        "split": split,
        "seed": seed,
        "splits_sha256": splits_sha256,
        "min_treated_cells": min_treated,
        "min_candidates": min_candidates,
        "anchor_excluded": ANCHOR_COMPOUND,
        "contexts": {key: dict(contexts[key]) for key in sorted(used)},
        "queries": queries,
        "dropped_blocks": dropped,
    }
