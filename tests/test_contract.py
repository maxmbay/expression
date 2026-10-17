import numpy as np
import polars as pl
import pytest

from cellresponse.features.fit import choose_hvg
from cellresponse.goals import goals, passes_event
from cellresponse.queries import FORBIDDEN_KEYS, build_documents, check_query_document
from cellresponse.split import (
    ANCHOR_COMPOUND,
    assemble_splits,
    assign_protocol_a,
    repair_pair_roles,
    roles_from_units,
    unit_hash,
)


def test_unit_hash_is_stable_and_inside_the_unit_interval():
    assert unit_hash(0, "A", "L1", "G1") == unit_hash(0, "A", "L1", "G1")
    assert 0.0 <= unit_hash(0, "A", "L1", "G1") < 1.0
    assert unit_hash(0, "A", "L1", "G1") != unit_hash(1, "A", "L1", "G1")


def test_repair_gives_a_held_out_line_a_training_partner():
    pairs = [("L1", "G1"), ("L1", "G2"), ("L2", "G1"), ("L2", "G2")]
    units = {
        ("L1", "G1"): 0.95,
        ("L1", "G2"): 0.99,
        ("L2", "G1"): 0.10,
        ("L2", "G2"): 0.20,
    }
    roles = repair_pair_roles(pairs, units, roles_from_units(units))
    assert roles[("L1", "G1")] == "train"
    assert roles[("L1", "G2")] == "test"
    assert roles[("L2", "G2")] == "train"


def test_protocol_a_covers_every_held_out_partner():
    lines = [f"L{i}" for i in range(6)]
    groups = [f"G{i}" for i in range(40)]
    pairs = [(line, group) for line in lines for group in groups]
    roles = assign_protocol_a(pairs, seed=0)
    train_lines = {line for (line, _), role in roles.items() if role == "train"}
    train_groups = {group for (_, group), role in roles.items() if role == "train"}
    for (line, group), role in roles.items():
        if role in {"validation", "test"}:
            assert line in train_lines
            assert group in train_groups
    counts = {
        role: sum(value == role for value in roles.values())
        for role in ("train", "validation", "test")
    }
    assert counts["train"] > counts["validation"] > 0
    assert counts["train"] > counts["test"] > 0


def test_toy_split_has_no_leakage_and_queries_have_no_outcomes():
    cells, conditions, compounds = _toy_cohort()
    splits = assemble_splits(cells, conditions, compounds, seed=0, min_treated=1)
    treated = splits.filter(~pl.col("is_control") & (pl.col("compound_key") != ANCHOR_COMPOUND))
    roles = treated.group_by("cell_line", "structure_group").agg(n=pl.col("protocol_a").n_unique())
    assert roles.filter(pl.col("n") > 1).height == 0
    assert splits.filter(pl.col("fit_a") & (pl.col("plate_num") == 14)).height == 0
    assert splits.filter(pl.col("fit_c") & (pl.col("plate_num") == 14)).height == 0
    assert splits.filter((pl.col("protocol_a") == "anchor") & pl.col("fit_a")).height == 0
    wells = (
        splits.filter(pl.col("is_control")).select("plate_num", "sample", "control_role").unique()
    )
    assert wells.group_by("plate_num", "sample").len().filter(pl.col("len") > 1).height == 0
    reference = set(splits.filter(pl.col("control_role") == "reference")["cell_id"].to_list())
    evaluation = set(splits.filter(pl.col("control_role") == "evaluation")["cell_id"].to_list())
    assert reference.isdisjoint(evaluation)
    assert reference and evaluation

    documents = build_documents(
        splits, conditions, seed=0, splits_sha256="abc", min_treated=1, min_candidates=1
    )
    for name, document in documents.items():
        check_query_document(document)
        assert set(_keys(document)).isdisjoint(FORBIDDEN_KEYS)
        for query in document["queries"]:
            context = document["contexts"][query["context_id"]]
            assert set(context["reference_cell_ids"]).isdisjoint(context["evaluation_cell_ids"])
            for candidate in query["panel"]:
                assert candidate["compound_key"] != ANCHOR_COMPOUND
                role = splits.filter(
                    (pl.col("cell_line") == context["cell_line"])
                    & (pl.col("compound_key") == candidate["compound_key"])
                )["protocol_a"][0]
                if name == "plate14":
                    assert query["protocol"] == "B"
                else:
                    assert role == document["split"]
                    assert context["plate_num"] != 14


def test_query_document_rejects_an_outcome():
    document = {
        "min_candidates": 1,
        "contexts": {"c": {"reference_cell_ids": [1]}},
        "queries": [
            {"query_id": "q", "context_id": "c", "panel": [{"compound_key": "a"}], "utility": 0.4}
        ],
    }
    with pytest.raises(RuntimeError, match="forbidden key"):
        check_query_document(document)


def test_hvg_ranks_a_gene_above_the_mean_variance_trend():
    n_cells = 200
    n_genes = 400
    mean = np.linspace(0.05, 5.0, n_genes)
    var = mean.copy()
    var[7] = mean[7] * 40.0
    count_sum = mean * n_cells
    count_sumsq = (var + mean**2) * n_cells
    assert int(choose_hvg(count_sum, count_sumsq, n_cells, n_hvg=1)[0]) == 7


def test_goal_membership_is_the_hallmark_list():
    by_id = {goal.goal_id: goal for goal in goals()}
    assert set(by_id) == {"g2m_down", "p53_up", "apoptosis_constraint", "interferon_gamma_up"}
    assert "CDK1" in by_id["g2m_down"].symbols
    assert "CDKN1A" in by_id["p53_up"].symbols
    assert "CASP3" in by_id["apoptosis_constraint"].symbols
    assert "STAT1" in by_id["interferon_gamma_up"].symbols
    assert by_id["apoptosis_constraint"].role == "constraint"
    assert passes_event(np.array([0.1, 0.5]), 0.2, "down").tolist() == [True, False]
    assert passes_event(np.array([0.1, 0.5]), 0.2, "up").tolist() == [False, True]


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def _toy_cohort():
    lines = ["L1", "L2"]
    groups = [f"G{i}" for i in range(8)]
    cell_rows = []
    condition_rows = []
    cell_id = 1

    def add(n, **fields):
        nonlocal cell_id
        for _ in range(n):
            cell_rows.append({"cell_id": cell_id, **fields})
            cell_id += 1

    for plate in (6, 14):
        for suffix in ("a", "b"):
            sample = f"veh-{plate}-{suffix}"
            for line in lines:
                add(
                    3,
                    sample=sample,
                    cell_line=line,
                    plate_num=plate,
                    compound_key="dmso_tf",
                    dose_uM=0.0,
                    is_control=True,
                )
                condition_rows.append(
                    _condition(sample, line, plate, "dmso_tf", "DMSO_TF", 0.0, True, 3)
                )
        for group in groups:
            sample = f"p{plate}-{group}"
            for line in lines:
                add(
                    3,
                    sample=sample,
                    cell_line=line,
                    plate_num=plate,
                    compound_key=group,
                    dose_uM=5.0,
                    is_control=False,
                )
                condition_rows.append(_condition(sample, line, plate, group, group, 5.0, False, 3))
        for line in lines:
            sample = f"p{plate}-anchor"
            add(
                2,
                sample=sample,
                cell_line=line,
                plate_num=plate,
                compound_key=ANCHOR_COMPOUND,
                dose_uM=0.05,
                is_control=False,
            )
            condition_rows.append(
                _condition(sample, line, plate, ANCHOR_COMPOUND, "Adagrasib", 0.05, False, 2)
            )

    compounds = pl.DataFrame(
        {
            "compound_key": [*groups, ANCHOR_COMPOUND, "verteporfin"],
            "structure_group": [*groups, "ANCHOR-INCHI", "verteporfin"],
            "fingerprint_valid": [True] * len(groups) + [True, False],
        }
    )
    return pl.DataFrame(cell_rows), pl.DataFrame(condition_rows), compounds


def _condition(sample, line, plate, compound, drug, dose, is_control, retained):
    return {
        "sample": sample,
        "cell_line": line,
        "plate": f"plate{plate}",
        "plate_num": plate,
        "compound_key": compound,
        "drug": drug,
        "dose_uM": dose,
        "is_control": is_control,
        "retained_cells": retained,
    }
