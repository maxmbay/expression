"""Fit the M2 baselines on protocol-A validation queries and write the report.

Test-treated module scores are not read. Observed utilities use validation wells only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cellresponse.baselines.gaussian import train_gaussian
from cellresponse.baselines.metrics import distributional, population_utility
from cellresponse.baselines.scores import (
    PENALTY_GOAL,
    PENALTY_WEIGHT,
    ensure_scores,
    objective_goals,
    score_names,
)
from cellresponse.data.report import md_table
from cellresponse.goals import fit_soft, goals
from cellresponse.split import HELD_OUT_PLATE

RESIDUALS = 256
FP_BITS = 2048
PROJ_DIM = 32


def run(config: dict) -> None:
    prepared = Path(config["paths"]["prepared"]) / config["name"]
    seed = int(config["cohort"]["seed"])
    genes = pl.read_parquet(prepared / "genes.parquet")
    scores = ensure_scores(prepared, genes)
    splits = pl.read_parquet(
        prepared / "splits.parquet",
        columns=[
            "cell_id",
            "sample",
            "cell_line",
            "plate_num",
            "compound_key",
            "structure_group",
            "dose_uM",
            "is_control",
            "control_role",
            "protocol_a",
            "protocol_c",
            "fit_a",
        ],
    )
    joined = scores.join(splits, on="cell_id", how="inner")
    curves = _curves(joined)
    (prepared / "utility.json").write_text(json.dumps(curves, indent=2) + "\n")
    matrix, row_of = _matrix(joined)
    dev = json.loads(Path("queries/dev.json").read_text())
    compounds = pl.read_parquet(prepared / "compounds.parquet")
    fingerprints = _fingerprints(compounds)
    projection = np.random.default_rng(seed).normal(size=(FP_BITS, PROJ_DIM)).astype(np.float32)
    projection /= np.sqrt(FP_BITS)
    directions = _directions(seed)

    train = joined.filter(
        (pl.col("protocol_a") == "train")
        & ~pl.col("is_control")
        & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    reference = joined.filter(
        (pl.col("control_role") == "reference") & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    shift = _fit_shift(train, reference)
    ridge, residual_bank, scaler = _fit_ridge(train, reference, fingerprints, seed)
    print("[baselines] training conditional gaussian", flush=True)
    gauss_h, gauss_y = _gaussian_rows(train, reference, fingerprints, projection, seed)
    import jax.numpy as jnp

    gaussian = train_gaussian(jnp.asarray(gauss_h), jnp.asarray(gauss_y), seed)
    treated = _ids_by_condition(
        splits.filter(
            (pl.col("protocol_a") == "validation")
            & ~pl.col("is_control")
            & (pl.col("plate_num") != HELD_OUT_PLATE)
        )
    )
    contexts = _contexts(dev, matrix, row_of, treated, fingerprints)
    predictions = {
        "untreated": _predict_untreated(contexts),
        "shared_shift": _predict_shift(contexts, shift),
        "ridge": _predict_ridge(contexts, ridge, scaler, residual_bank, seed),
        "gaussian": _predict_gaussian(contexts, gaussian, projection, seed),
    }
    rows, dist_rows = _score_predictions(contexts, predictions, curves, directions, seed)
    table = pl.DataFrame(rows)
    dist = pl.DataFrame(dist_rows) if dist_rows else pl.DataFrame()
    neighbor = _neighbor_report(
        joined, splits, fingerprints, curves, directions, seed, matrix, row_of
    )
    report = _report(table, dist, curves, neighbor)
    Path("reports/baselines.md").write_text(report)
    table.write_parquet(prepared / "baseline_queries.parquet")
    print("[baselines] wrote reports/baselines.md", flush=True)


def _curves(joined: pl.DataFrame) -> dict:
    solvent = joined.filter((pl.col("control_role") == "reference") & pl.col("fit_a"))
    fitted = {}
    for goal in goals():
        threshold, scale = fit_soft(solvent[goal.goal_id].to_numpy(), goal.quantile)
        fitted[goal.goal_id] = {
            "threshold": threshold,
            "scale": scale,
            "quantile": goal.quantile,
            "direction": goal.direction,
            "role": goal.role,
            "n_solvent_cells": solvent.height,
        }
    return {"penalty_goal": PENALTY_GOAL, "penalty_weight": PENALTY_WEIGHT, "goals": fitted}


def _matrix(joined: pl.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    matrix = joined.select(score_names()).to_numpy().astype(np.float32)
    row_of = {int(cell_id): index for index, cell_id in enumerate(joined["cell_id"].to_list())}
    return matrix, row_of


def _gather(matrix: np.ndarray, row_of: dict[int, int], cell_ids: list[int]) -> np.ndarray:
    rows = [row_of[int(cell_id)] for cell_id in cell_ids if int(cell_id) in row_of]
    if not rows:
        raise RuntimeError("a query population has no scored cells")
    return matrix[rows]


def _fingerprints(compounds: pl.DataFrame) -> dict[str, np.ndarray]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    found = {}
    for row in compounds.iter_rows(named=True):
        smiles = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            continue
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_BITS)
        found[row["compound_key"]] = np.asarray(bits, dtype=np.float32)
    return found


def _directions(seed: int) -> np.ndarray:
    draw = np.random.default_rng(seed).normal(size=(16, len(score_names())))
    return draw / np.linalg.norm(draw, axis=1, keepdims=True)


def _condition_means(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by(
        "cell_line", "plate_num", "compound_key", "structure_group", "dose_uM"
    ).agg([pl.col(name).mean() for name in score_names()])


def _fit_shift(train: pl.DataFrame, reference: pl.DataFrame) -> dict[tuple[str, float], np.ndarray]:
    treated = _condition_means(train)
    solvent = reference.group_by("cell_line", "plate_num").agg(
        [pl.col(name).mean().alias(f"ref_{name}") for name in score_names()]
    )
    joined = treated.join(solvent, on=["cell_line", "plate_num"], how="inner")
    deltas = joined.with_columns(
        [(pl.col(name) - pl.col(f"ref_{name}")).alias(f"d_{name}") for name in score_names()]
    )
    pooled = deltas.group_by("structure_group", "dose_uM").agg(
        [pl.col(f"d_{name}").mean().alias(name) for name in score_names()]
    )
    shift = {}
    for row in pooled.iter_rows(named=True):
        shift[(row["structure_group"], round(float(row["dose_uM"]), 4))] = np.array(
            [row[name] for name in score_names()], dtype=np.float32
        )
    return shift


def _fit_ridge(train, reference, fingerprints, seed):
    treated = _condition_means(train)
    solvent = reference.group_by("cell_line", "plate_num").agg(
        [pl.col(name).mean().alias(f"ref_{name}") for name in score_names()]
        + [pl.col(name).std().alias(f"sd_{name}") for name in score_names()]
    )
    joined = treated.join(solvent, on=["cell_line", "plate_num"], how="inner")
    features, targets = [], []
    for row in joined.iter_rows(named=True):
        features.append(_ridge_features(row, fingerprints.get(row["compound_key"])))
        targets.append([row[name] for name in score_names()])
    x = np.asarray(features, np.float32)
    y = np.asarray(targets, np.float32)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=10.0).fit(scaler.transform(x), y)
    bank = _residual_bank(train, joined, seed)
    return model, bank, scaler


def _ridge_features(row: dict, fingerprint: np.ndarray | None) -> np.ndarray:
    mean = np.array([row[f"ref_{name}"] for name in score_names()], dtype=np.float32)
    log_sd = np.log(np.maximum([row[f"sd_{name}"] for name in score_names()], 1e-3)).astype(
        np.float32
    )
    bits = fingerprint if fingerprint is not None else np.zeros(FP_BITS, np.float32)
    dose = np.array([np.log1p(float(row["dose_uM"]) / 0.05)], dtype=np.float32)
    return np.concatenate([mean, log_sd, bits, dose])


def _residual_bank(train: pl.DataFrame, condition_means: pl.DataFrame, seed: int) -> np.ndarray:
    means = condition_means.select(
        "cell_line",
        "plate_num",
        "compound_key",
        *[pl.col(name).alias(f"m_{name}") for name in score_names()],
    )
    cells = train.join(means, on=["cell_line", "plate_num", "compound_key"], how="inner")
    residual = cells.select(
        [(pl.col(name) - pl.col(f"m_{name}")).alias(name) for name in score_names()]
    )
    values = residual.to_numpy().astype(np.float32)
    rng = np.random.default_rng(seed)
    if values.shape[0] > 100_000:
        values = values[rng.choice(values.shape[0], 100_000, replace=False)]
    return values


def _gaussian_rows(train, reference, fingerprints, projection, seed):
    solvent = reference.group_by("cell_line", "plate_num").agg(
        [pl.col(name).mean().alias(f"ref_{name}") for name in score_names()]
        + [pl.col(name).std().alias(f"sd_{name}") for name in score_names()]
    )
    cells = train.join(solvent, on=["cell_line", "plate_num"], how="inner")
    rng = np.random.default_rng(seed)
    if cells.height > 60_000:
        keep = rng.choice(cells.height, 60_000, replace=False)
        cells = (
            cells.with_row_index("_row").filter(pl.col("_row").is_in(keep.tolist())).drop("_row")
        )
    features, targets = [], []
    for row in cells.iter_rows(named=True):
        features.append(_condition_vector(row, fingerprints.get(row["compound_key"]), projection))
        targets.append([row[name] for name in score_names()])
    return np.asarray(features, np.float32), np.asarray(targets, np.float32)


def _condition_vector(
    row: dict, fingerprint: np.ndarray | None, projection: np.ndarray
) -> np.ndarray:
    mean = np.array([row[f"ref_{name}"] for name in score_names()], dtype=np.float32)
    log_sd = np.log(np.maximum([row[f"sd_{name}"] for name in score_names()], 1e-3)).astype(
        np.float32
    )
    bits = fingerprint if fingerprint is not None else np.zeros(FP_BITS, np.float32)
    projected = bits @ projection
    dose = np.log1p(float(row["dose_uM"]) / 0.05)
    return np.concatenate([mean, log_sd, projected, np.array([dose, 1.0], dtype=np.float32)])


def _ids_by_condition(
    frame: pl.DataFrame, keys: tuple[str, ...] = ("cell_line", "plate_num", "compound_key")
) -> dict[tuple, list[int]]:
    grouped = {}
    for key, part in frame.group_by(list(keys)):
        grouped[key] = [int(cell_id) for cell_id in part["cell_id"].to_list()]
    return grouped


def _contexts(dev, matrix, row_of, treated, fingerprints) -> list[dict]:
    contexts = []
    seen = set()
    for query in dev["queries"]:
        if query["goal_id"] not in {goal.goal_id for goal in objective_goals()}:
            continue
        if query["context_id"] in seen:
            continue
        seen.add(query["context_id"])
        context = dev["contexts"][query["context_id"]]
        line, plate = context["cell_line"], int(context["plate_num"])
        panel = []
        for candidate in query["panel"]:
            key = (line, plate, candidate["compound_key"])
            cell_ids = treated.get(key, [])
            if not cell_ids:
                continue
            panel.append(
                {
                    **candidate,
                    "cell_ids": cell_ids,
                    "observed": _gather(matrix, row_of, cell_ids),
                    "fingerprint": fingerprints.get(candidate["compound_key"]),
                }
            )
        if len(panel) < 10:
            continue
        contexts.append(
            {
                "cell_line": line,
                "plate_num": plate,
                "reference": _gather(matrix, row_of, context["reference_cell_ids"]),
                "evaluation": _gather(matrix, row_of, context["evaluation_cell_ids"]),
                "panel": panel,
            }
        )
    return contexts


def _predict_untreated(contexts) -> dict:
    predicted = {}
    for context in contexts:
        for candidate in context["panel"]:
            predicted[_key(context, candidate)] = context["reference"]
    return predicted


def _predict_shift(contexts, shift) -> dict:
    predicted = {}
    for context in contexts:
        for candidate in context["panel"]:
            delta = shift.get((candidate["structure_group"], round(float(candidate["dose_uM"]), 4)))
            base = context["reference"]
            predicted[_key(context, candidate)] = base if delta is None else base + delta
    return predicted


def _predict_ridge(contexts, model, scaler, bank, seed) -> dict:
    rng = np.random.default_rng(seed)
    predicted = {}
    for context in contexts:
        ref_mean = context["reference"].mean(axis=0)
        ref_sd = context["reference"].std(axis=0)
        row = {f"ref_{name}": ref_mean[i] for i, name in enumerate(score_names())}
        row.update({f"sd_{name}": ref_sd[i] for i, name in enumerate(score_names())})
        for candidate in context["panel"]:
            row["dose_uM"] = candidate["dose_uM"]
            features = _ridge_features(row, candidate["fingerprint"]).reshape(1, -1)
            mean = model.predict(scaler.transform(features))[0]
            draw = bank[rng.integers(0, bank.shape[0], size=RESIDUALS)]
            predicted[_key(context, candidate)] = mean + draw
    return predicted


def _predict_gaussian(contexts, model, projection, seed) -> dict:
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    conditions, keys = [], []
    for context in contexts:
        ref_mean = context["reference"].mean(axis=0)
        ref_sd = context["reference"].std(axis=0)
        row = {f"ref_{name}": ref_mean[i] for i, name in enumerate(score_names())}
        row.update({f"sd_{name}": ref_sd[i] for i, name in enumerate(score_names())})
        for candidate in context["panel"]:
            row["dose_uM"] = candidate["dose_uM"]
            conditions.append(_condition_vector(row, candidate["fingerprint"], projection))
            keys.append(_key(context, candidate))
    stacked = jnp.asarray(np.asarray(conditions, np.float32))
    samples = jax.vmap(lambda condition, key: model.sample(condition, key, RESIDUALS))(
        stacked, jr.split(jr.PRNGKey(seed), stacked.shape[0])
    )
    return {key: np.asarray(sample) for key, sample in zip(keys, samples, strict=True)}


def _key(context: dict, candidate: dict) -> tuple:
    return (context["cell_line"], context["plate_num"], candidate["compound_key"])


def _score_predictions(
    contexts, predictions, curves, directions, seed
) -> tuple[list[dict], list[dict]]:
    names = score_names()
    columns = {name: index for index, name in enumerate(names)}
    center = {name: curves["goals"][name]["threshold"] for name in names}
    scale = {name: curves["goals"][name]["scale"] for name in names}
    direction = {name: curves["goals"][name]["direction"] for name in names}
    rng = np.random.default_rng(seed)
    rows, dist_rows = [], []
    for context in contexts:
        for goal in objective_goals():
            for model, predicted in predictions.items():
                ranking = []
                for candidate in context["panel"]:
                    key = _key(context, candidate)
                    pred_u = _utility(
                        predicted[key], curves, columns, center, scale, direction, goal.goal_id
                    )
                    solvent_u = _utility(
                        context["reference"],
                        curves,
                        columns,
                        center,
                        scale,
                        direction,
                        goal.goal_id,
                    )
                    obs_u = _utility(
                        candidate["observed"],
                        curves,
                        columns,
                        center,
                        scale,
                        direction,
                        goal.goal_id,
                    )
                    eval_u = _utility(
                        context["evaluation"],
                        curves,
                        columns,
                        center,
                        scale,
                        direction,
                        goal.goal_id,
                    )
                    ranking.append(
                        {
                            "compound_key": candidate["compound_key"],
                            "predicted": pred_u - solvent_u,
                            "observed": obs_u - eval_u,
                            "predicted_sample": predicted[key],
                            "observed_sample": candidate["observed"],
                        }
                    )
                ranking.append(
                    {
                        "compound_key": None,
                        "predicted": 0.0,
                        "observed": 0.0,
                        "predicted_sample": None,
                        "observed_sample": None,
                    }
                )
                rows.append(_rank_row(model, goal.goal_id, context, ranking))
                if goal.goal_id != objective_goals()[0].goal_id:
                    continue
                for item in ranking:
                    if item["predicted_sample"] is None:
                        continue
                    metrics = distributional(
                        _subsample(item["predicted_sample"], rng),
                        _subsample(item["observed_sample"], rng),
                        directions,
                    )
                    dist_rows.append(
                        {
                            "model": model,
                            "cell_line": context["cell_line"],
                            "plate_num": context["plate_num"],
                            **metrics,
                        }
                    )
    return rows, dist_rows


def _utility(samples, curves, columns, center, scale, direction, goal_id) -> float:
    return population_utility(
        samples,
        columns,
        center,
        scale,
        direction,
        goal_id,
        curves["penalty_goal"],
        curves["penalty_weight"],
    )


def _rank_row(model, goal_id, context, ranking) -> dict:
    # On a tie, doing nothing wins. Otherwise every zero-utility model picks the first drug.
    order = sorted(
        ranking,
        key=lambda item: (item["predicted"], item["compound_key"] is None),
        reverse=True,
    )
    pick = order[0]
    top5 = order[:5]
    best = max(item["observed"] for item in ranking)
    predicted = np.array(
        [item["predicted"] for item in ranking if item["compound_key"] is not None]
    )
    observed = np.array([item["observed"] for item in ranking if item["compound_key"] is not None])
    spearman = _spearman(predicted, observed)
    return {
        "model": model,
        "goal_id": goal_id,
        "cell_line": context["cell_line"],
        "plate_num": context["plate_num"],
        "n_candidates": len(ranking) - 1,
        "top1_observed": pick["observed"],
        "top5_observed": float(np.mean([item["observed"] for item in top5])),
        "regret": best - pick["observed"],
        "picked_nothing": pick["compound_key"] is None,
        "spearman": spearman,
    }


def _spearman(predicted: np.ndarray, observed: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if predicted.size < 3 or np.unique(predicted).size < 2 or np.unique(observed).size < 2:
        return float("nan")
    return float(spearmanr(predicted, observed).statistic)


def _subsample(values: np.ndarray, rng: np.random.Generator, n: int = 64) -> np.ndarray:
    if values.shape[0] <= n:
        return values
    return values[rng.choice(values.shape[0], n, replace=False)]


def _neighbor_report(
    joined, splits, fingerprints, curves, directions, seed, matrix, row_of
) -> dict:
    """Protocol C transfers the shift of the nearest training structure."""
    names = score_names()
    train_groups = set(
        splits.filter(
            (pl.col("protocol_c") == "train")
            & ~pl.col("is_control")
            & (pl.col("plate_num") != HELD_OUT_PLATE)
        )["structure_group"].to_list()
    )
    val = splits.filter(
        (pl.col("protocol_c") == "validation")
        & ~pl.col("is_control")
        & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    group_fp = _group_fingerprints(splits, fingerprints)
    train_keys = sorted(group for group in train_groups if group_fp.get(group) is not None)
    neighbors = {}
    for group, bits in group_fp.items():
        if group in train_keys or bits is None:
            continue
        neighbors[group] = max(train_keys, key=lambda other: _tanimoto(bits, group_fp[other]))
    train = joined.filter(
        (pl.col("protocol_c") == "train")
        & ~pl.col("is_control")
        & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    reference = joined.filter(
        (pl.col("control_role") == "reference") & (pl.col("plate_num") != HELD_OUT_PLATE)
    )
    shift = _fit_shift(train, reference)
    treated = _ids_by_condition(val)
    solvent_ids = {
        role: _ids_by_condition(
            splits.filter(pl.col("control_role") == role), ("cell_line", "plate_num")
        )
        for role in ("reference", "evaluation")
    }
    contexts, predicted = [], {}
    blocks = val.select(
        "cell_line", "plate_num", "compound_key", "structure_group", "dose_uM"
    ).unique()
    for key, part in blocks.group_by("cell_line", "plate_num"):
        line, plate = key
        ref_ids = solvent_ids["reference"].get((line, plate), [])
        eval_ids = solvent_ids["evaluation"].get((line, plate), [])
        if not ref_ids or not eval_ids:
            continue
        reference_scores = _gather(matrix, row_of, ref_ids)
        panel = []
        for row in part.iter_rows(named=True):
            neighbor = neighbors.get(row["structure_group"])
            delta = shift.get((neighbor, round(float(row["dose_uM"]), 4))) if neighbor else None
            cell_ids = [
                cell_id
                for cell_id in treated.get((line, plate, row["compound_key"]), [])
                if cell_id in row_of
            ]
            if delta is None or not cell_ids:
                continue
            panel.append(
                {
                    "compound_key": row["compound_key"],
                    "structure_group": row["structure_group"],
                    "dose_uM": row["dose_uM"],
                    "cell_ids": cell_ids,
                    "observed": _gather(matrix, row_of, cell_ids),
                    "fingerprint": fingerprints.get(row["compound_key"]),
                }
            )
            predicted[(line, plate, row["compound_key"])] = reference_scores + delta
        if len(panel) < 10:
            continue
        contexts.append(
            {
                "cell_line": line,
                "plate_num": int(plate),
                "reference": reference_scores,
                "evaluation": _gather(matrix, row_of, eval_ids),
                "panel": panel,
            }
        )
    rows, _dist = _score_predictions(contexts, {"neighbor": predicted}, curves, directions, seed)
    summary = pl.DataFrame(rows) if rows else pl.DataFrame()
    return {
        "queries": summary.height if summary.height else 0,
        "held_out_groups": len(neighbors),
        "train_groups": len(train_keys),
        "summary": summary,
        "score_names": names,
    }


def _group_fingerprints(splits, fingerprints) -> dict[str, np.ndarray | None]:
    pairs = splits.filter(~pl.col("is_control")).select("structure_group", "compound_key").unique()
    grouped: dict[str, list[np.ndarray]] = {}
    for group, compound in pairs.iter_rows():
        if compound in fingerprints:
            grouped.setdefault(group, []).append(fingerprints[compound])
    return {group: (np.mean(bits, axis=0) if bits else None) for group, bits in grouped.items()}


def _tanimoto(left: np.ndarray, right: np.ndarray) -> float:
    inter = float(np.minimum(left, right).sum())
    union = float(np.maximum(left, right).sum())
    return inter / union if union else 0.0


def _report(table: pl.DataFrame, dist: pl.DataFrame, curves: dict, neighbor: dict) -> str:
    summary = (
        table.group_by("model")
        .agg(
            top1=pl.col("top1_observed").mean(),
            top5=pl.col("top5_observed").mean(),
            regret=pl.col("regret").mean(),
            nothing=pl.col("picked_nothing").mean(),
            spearman=pl.col("spearman").drop_nans().mean(),
            queries=pl.len(),
        )
        .sort("top1", descending=True)
    )
    by_goal = (
        table.group_by("model", "goal_id")
        .agg(top1=pl.col("top1_observed").mean(), spearman=pl.col("spearman").drop_nans().mean())
        .sort("goal_id", "model")
    )
    lines = [
        "# Baselines",
        "",
        "Validation queries only. Utility is the soft membership gain of the objective minus",
        "the apoptosis membership gain. Doing nothing has utility zero. Test-treated cells",
        "were not scored and do not enter these rankings.",
        "",
        "## Solvent curves",
        "",
    ]
    curve_rows = [
        {
            "goal": name,
            "threshold": round(item["threshold"], 4),
            "scale": round(item["scale"], 4),
            "n": item["n_solvent_cells"],
        }
        for name, item in curves["goals"].items()
    ]
    lines.append(md_table(pl.DataFrame(curve_rows)))
    lines += ["", "## Search", "", md_table(summary), "", "## By goal", "", md_table(by_goal), ""]
    if dist.height:
        dist_summary = (
            dist.group_by("model")
            .agg(
                [
                    pl.col(name).mean()
                    for name in ("energy", "mean_shift", "marginal_w1", "sliced_w1", "covariance")
                ]
            )
            .sort("energy")
        )
        lines += ["## Distributional distance to validation cells", "", md_table(dist_summary), ""]
    lines += [
        "## Protocol C neighbor transfer",
        "",
        "A held-out structure borrows the training shift of its nearest Morgan neighbor.",
        "These blocks are not the protocol-A queries.",
        "",
        f"Held-out groups: {neighbor.get('held_out_groups', 0)}. "
        f"Training groups: {neighbor.get('train_groups', 0)}.",
        "",
    ]
    summary = neighbor.get("summary")
    if summary is not None and summary.height:
        collapsed = summary.group_by("goal_id").agg(
            top1=pl.col("top1_observed").mean(),
            spearman=pl.col("spearman").drop_nans().mean(),
            queries=pl.len(),
        )
        lines += [md_table(collapsed.sort("goal_id")), ""]
    return "\n".join(lines)
