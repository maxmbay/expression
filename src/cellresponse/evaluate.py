"""Lock test rankings before any test-treated score is read, then score them.

The selected model is the shared shift. Ridge and untreated go in the same ranking
file, so the comparison is fixed before the test outcomes are loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from cellresponse.baselines.run import (
    _curves,
    _fingerprints,
    _fit_ridge,
    _fit_shift,
    _gather,
    _ids_by_condition,
    _key,
    _matrix,
    _predict_ridge,
    _predict_shift,
    _predict_untreated,
    _utility,
)
from cellresponse.baselines.scores import ensure_scores, objective_goals, score_cells, score_names
from cellresponse.data.report import md_table
from cellresponse.data.source import sha256sum
from cellresponse.split import HELD_OUT_PLATE

GATE = 0.05
FORBIDDEN = {"observed", "observed_utility", "expression", "counts", "genes"}


def run(config: dict) -> None:
    prepared = Path(config["paths"]["prepared"]) / config["name"]
    seed = int(config["cohort"]["seed"])
    genes = pl.read_parquet(prepared / "genes.parquet")
    scores = ensure_scores(prepared, genes)
    splits = pl.read_parquet(prepared / "splits.parquet")
    joined = scores.join(splits, on="cell_id", how="inner")
    curves = _curves(joined)
    matrix, row_of = _matrix(joined)
    fingerprints = _fingerprints(pl.read_parquet(prepared / "compounds.parquet"))
    train = joined.filter(
        (pl.col("protocol_a") == "train")
        & ~pl.col("is_control")
        & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    reference = joined.filter(
        (pl.col("control_role") == "reference") & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    shift = _fit_shift(train, reference)
    ridge, bank, scaler = _fit_ridge(train, reference, fingerprints, seed)
    ids = {
        "test": _ids_by_condition(
            splits.filter((pl.col("protocol_a") == "test") & ~pl.col("is_control"))
        ),
        "validation": _ids_by_condition(
            splits.filter((pl.col("protocol_a") == "validation") & ~pl.col("is_control"))
        ),
        "train": _ids_by_condition(
            splits.filter((pl.col("protocol_a") == "train") & ~pl.col("is_control"))
        ),
    }
    documents = {
        "test": json.loads(Path("queries/test.json").read_text()),
        "plate14": json.loads(Path("queries/plate14.json").read_text()),
    }
    role = {"test": "test", "validation": "validation", "seen_condition_new_plate": "train"}
    contexts = {
        name: _contexts(document, matrix, row_of, fingerprints, ids, role)
        for name, document in documents.items()
    }
    predictions = {
        name: {
            "untreated": _predict_untreated(items),
            "shared_shift": _predict_shift(items, shift),
            "ridge": _predict_ridge(items, ridge, scaler, bank, seed),
        }
        for name, items in contexts.items()
    }
    ranking = {
        "selected_model": "shared_shift",
        "seed": seed,
        "sets": {
            name: _ranking_records(items, predictions[name], curves)
            for name, items in contexts.items()
        },
    }
    _reject_outcomes(ranking)
    path = Path("reports/test_rankings.json")
    path.write_text(json.dumps(ranking, indent=2) + "\n")
    digest = sha256sum(path)
    Path("reports/test_rankings.sha256").write_text(digest + "\n")
    if sha256sum(path) != digest:
        raise RuntimeError("ranking file changed after it was hashed")
    print(f"[evaluate] rankings hashed {digest}", flush=True)

    test_ids = splits.filter((pl.col("protocol_a") == "test") & ~pl.col("is_control"))["cell_id"]
    print(f"[evaluate] scoring {test_ids.len():,} test-treated cells", flush=True)
    extra = score_cells(prepared, genes, test_ids)
    matrix, row_of = _append_scores(matrix, row_of, extra)
    rows = []
    for name, items in contexts.items():
        _attach_observed(items, matrix, row_of)
        rows.extend(_observed_rows(name, items, predictions[name], curves))
    table = pl.DataFrame(rows)
    table.write_csv("reports/tables/test_queries.csv")
    Path("reports/benchmark.md").write_text(_report(table, digest))
    print(f"[evaluate] wrote reports/benchmark.md ({digest[:12]})", flush=True)


def _contexts(document, matrix, row_of, fingerprints, ids, role) -> list[dict]:
    objective = {goal.goal_id for goal in objective_goals()}
    contexts, seen = [], set()
    for query in document["queries"]:
        if query["goal_id"] not in objective:
            continue
        identity = (query["context_id"], query["split"])
        if identity in seen:
            continue
        seen.add(identity)
        source = ids[role[query["split"]]]
        meta = document["contexts"][query["context_id"]]
        line, plate = meta["cell_line"], int(meta["plate_num"])
        panel = []
        for candidate in query["panel"]:
            cell_ids = source.get((line, plate, candidate["compound_key"]), [])
            if not cell_ids:
                continue
            panel.append(
                {
                    **candidate,
                    "cell_ids": cell_ids,
                    "fingerprint": fingerprints.get(candidate["compound_key"]),
                }
            )
        if len(panel) < 10:
            continue
        contexts.append(
            {
                "cell_line": line,
                "plate_num": plate,
                "split": query["split"],
                "reference": _gather(matrix, row_of, meta["reference_cell_ids"]),
                "evaluation_ids": [int(cell_id) for cell_id in meta["evaluation_cell_ids"]],
                "panel": panel,
            }
        )
    return contexts


def _ranking_records(contexts, predictions, curves) -> list[dict]:
    records = []
    for context in contexts:
        for goal in objective_goals():
            for model, predicted in predictions.items():
                order = _bundle(context, predicted, curves, goal.goal_id)
                records.append(
                    {
                        "model": model,
                        "cell_line": context["cell_line"],
                        "plate_num": context["plate_num"],
                        "split": context["split"],
                        "goal_id": goal.goal_id,
                        "ranking": [
                            {
                                "compound_key": item["compound_key"],
                                "predicted_utility": round(float(item["predicted"]), 6),
                            }
                            for item in order
                        ],
                    }
                )
    return records


def _bundle(context, predicted, curves, goal_id) -> list[dict]:
    names = score_names()
    packed = _pack(curves, names)
    solvent = _utility(context["reference"], curves, *packed, goal_id)
    ranking = []
    for candidate in context["panel"]:
        sample = predicted[_key(context, candidate)]
        ranking.append(
            {
                "compound_key": candidate["compound_key"],
                "predicted": _utility(sample, curves, *packed, goal_id) - solvent,
            }
        )
    ranking.append({"compound_key": None, "predicted": 0.0})
    return sorted(
        ranking, key=lambda item: (item["predicted"], item["compound_key"] is None), reverse=True
    )


def _pack(curves, names):
    return (
        {name: index for index, name in enumerate(names)},
        {name: curves["goals"][name]["threshold"] for name in names},
        {name: curves["goals"][name]["scale"] for name in names},
        {name: curves["goals"][name]["direction"] for name in names},
    )


def _reject_outcomes(ranking: dict) -> None:
    for name, records in ranking["sets"].items():
        for record in records:
            if FORBIDDEN & set(record):
                raise RuntimeError(f"{name} ranking contains an outcome field")
            for item in record["ranking"]:
                if set(item) != {"compound_key", "predicted_utility"}:
                    raise RuntimeError(f"{name} ranking item has unexpected fields")


def _append_scores(matrix, row_of, extra: pl.DataFrame):
    block = extra.select(score_names()).to_numpy().astype(np.float32)
    start = matrix.shape[0]
    for offset, cell_id in enumerate(extra["cell_id"].to_list()):
        row_of[int(cell_id)] = start + offset
    return np.concatenate([matrix, block], axis=0), row_of


def _attach_observed(contexts, matrix, row_of) -> None:
    for context in contexts:
        context["evaluation"] = _gather(matrix, row_of, context["evaluation_ids"])
        for candidate in context["panel"]:
            candidate["observed"] = _gather(matrix, row_of, candidate["cell_ids"])


def _observed_rows(name, contexts, predictions, curves) -> list[dict]:
    names = score_names()
    packed = _pack(curves, names)
    rows = []
    for context in contexts:
        for goal in objective_goals():
            for model, predicted in predictions.items():
                solvent = _utility(context["reference"], curves, *packed, goal.goal_id)
                evaluation = _utility(context["evaluation"], curves, *packed, goal.goal_id)
                ranking = []
                for candidate in context["panel"]:
                    sample = predicted[_key(context, candidate)]
                    observed = _utility(candidate["observed"], curves, *packed, goal.goal_id)
                    ranking.append(
                        {
                            "compound_key": candidate["compound_key"],
                            "predicted": _utility(sample, curves, *packed, goal.goal_id) - solvent,
                            "observed": observed - evaluation,
                        }
                    )
                ranking.append({"compound_key": None, "predicted": 0.0, "observed": 0.0})
                row = _summarize(model, goal.goal_id, context, ranking)
                row["set"] = name
                rows.append(row)
    return rows


def _summarize(model, goal_id, context, ranking) -> dict:
    order = sorted(
        ranking, key=lambda item: (item["predicted"], item["compound_key"] is None), reverse=True
    )
    pick = order[0]
    best = max(item["observed"] for item in ranking)
    predicted = np.array(
        [item["predicted"] for item in ranking if item["compound_key"] is not None]
    )
    observed = np.array([item["observed"] for item in ranking if item["compound_key"] is not None])
    return {
        "model": model,
        "goal_id": goal_id,
        "split": context["split"],
        "cell_line": context["cell_line"],
        "plate_num": context["plate_num"],
        "top1_observed": pick["observed"],
        "regret": best - pick["observed"],
        "picked_nothing": pick["compound_key"] is None,
        "spearman": _spearman(predicted, observed),
    }


def _spearman(predicted: np.ndarray, observed: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if predicted.size < 3 or np.unique(predicted).size < 2 or np.unique(observed).size < 2:
        return float("nan")
    return float(spearmanr(predicted, observed).statistic)


def _report(table: pl.DataFrame, digest: str) -> str:
    headline = table.filter(pl.col("set") == "test")
    summary = (
        headline.group_by("model")
        .agg(
            top1=pl.col("top1_observed").mean(),
            regret=pl.col("regret").mean(),
            nothing=pl.col("picked_nothing").mean(),
            spearman=pl.col("spearman").drop_nans().mean(),
            queries=pl.len(),
        )
        .sort("top1", descending=True)
    )
    shift = float(summary.filter(pl.col("model") == "shared_shift")["top1"][0])
    others = summary.filter(pl.col("model") != "shared_shift")["top1"]
    margin = shift - float(others.max())
    passed = margin >= GATE
    plate = (
        table.filter(pl.col("set") == "plate14")
        .group_by("split", "model")
        .agg(top1=pl.col("top1_observed").mean(), queries=pl.len())
        .sort("split", "model")
    )
    return "\n".join(
        [
            "# Benchmark",
            "",
            f"Rankings were written to `reports/test_rankings.json` and hashed (`{digest}`)",
            "before any test-treated module score was computed. The selected model is the",
            "shared shift. It has no random seed. The flow was not a finalist.",
            "",
            "## Test queries",
            "",
            md_table(summary),
            "",
            f"Shared shift minus the next best model: {margin:.4f}.",
            (
                "The +5 percentage-point gate passes."
                if passed
                else "The +5 percentage-point gate does not pass."
            ),
            "",
            "## Plate 14",
            "",
            md_table(plate),
            "",
        ]
    )
