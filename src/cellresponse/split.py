"""Deterministic splits. Published algorithm for protocol A:

1. An eligible pair is a (cell line, structure group) with at least `min_treated`
   retained cells in one non-anchor well. Structure group is the largest-fragment
   InChIKey, so salts and stereo-free duplicates stay together.
2. Draw ``u = blake2s(seed, "A", line, structure_group)`` in ``[0, 1)``.
   ``u < 0.70`` is train, ``u < 0.85`` is validation, and the rest is test.
   The hash payload is the UTF-8 bytes of those fields joined by NUL.
3. Repair: while a line or a structure group has a validation or test pair and no
   training pair, move its lowest-``u`` held-out pair to train.
4. Every dose, well, and plate of a pair, including plate 14, inherits that role.

Protocol C hashes ``blake2s(seed, "C", structure_group)`` with the same cutoffs and
no repair, because the held-out unit is the whole group. Groups that fail
fingerprinting, and the Adagrasib anchor, are excluded from C.

Each plate has two vehicle wells. The well with the smaller ``blake2s(seed,
"control", plate, sample)`` is the reference; the other is evaluation. Reference
cells may condition a model. They enter HVG and PCA fitting only on plates other
than 14. Evaluation cells are the vehicle endpoint and never enter a fit mask.
Plate 14 never enters a fit mask.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import polars as pl

ANCHOR_COMPOUND = "adagrasib"
HELD_OUT_PLATE = 14
TRAIN_CUT = 0.70
VAL_CUT = 0.85
MIN_TREATED_CELLS = 128
MIN_QUERY_CANDIDATES = 10

LINE_NAMES = {
    "CVCL_0480": "PANC-1",
    "CVCL_0546": "SW480",
    "CVCL_1056": "A498",
    "CVCL_1285": "HOP62",
    "CVCL_0359": "J82",
    "CVCL_0459": "NCI-H460",
    "CVCL_0399": "LoVo",
    "CVCL_0023": "A549",
    "CVCL_1119": "CFPAC-1",
    "CVCL_1381": "LOX-IMVI",
    "CVCL_1550": "NCI-H2347",
    "CVCL_0320": "HT-29",
}


def unit_hash(seed: int, *parts: str) -> float:
    payload = "\0".join((str(seed), *parts)).encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def roles_from_units(units: dict, train_cut: float = TRAIN_CUT, val_cut: float = VAL_CUT) -> dict:
    roles = {}
    for key, unit in units.items():
        if unit < train_cut:
            roles[key] = "train"
        elif unit < val_cut:
            roles[key] = "validation"
        else:
            roles[key] = "test"
    return roles


def repair_pair_roles(
    pairs: list[tuple[str, str]], units: dict[tuple[str, str], float], roles: dict
) -> dict:
    """Give every held-out line and structure group a training partner."""
    roles = dict(roles)
    for _ in range(len(pairs) + 1):
        by_line: dict[str, list[str]] = defaultdict(list)
        by_group: dict[str, list[str]] = defaultdict(list)
        for line, group in pairs:
            by_line[line].append(group)
            by_group[group].append(line)
        moved = False
        for line, groups in by_line.items():
            moved |= _move_to_train([(line, group) for group in groups], units, roles)
        for group, lines in by_group.items():
            moved |= _move_to_train([(line, group) for line in lines], units, roles)
        if not moved:
            return roles
    raise RuntimeError("protocol A repair did not terminate")


def _move_to_train(keys: list[tuple[str, str]], units: dict, roles: dict) -> bool:
    present = {roles[key] for key in keys}
    if "train" in present or not ({"validation", "test"} & present):
        return False
    held = [key for key in keys if roles[key] != "train"]
    roles[min(held, key=lambda key: (units[key], key))] = "train"
    return True


def assign_protocol_a(pairs: list[tuple[str, str]], seed: int) -> dict[tuple[str, str], str]:
    units = {(line, group): unit_hash(seed, "A", line, group) for line, group in pairs}
    return repair_pair_roles(pairs, units, roles_from_units(units))


def assign_protocol_c(groups: list[str], seed: int) -> dict[str, str]:
    units = {group: unit_hash(seed, "C", group) for group in groups}
    return roles_from_units(units)


def assign_control_wells(
    plate_wells: dict[int, list[str]], seed: int
) -> dict[tuple[int, str], str]:
    """One reference well and one evaluation well per plate. The well is shared by every line."""
    roles: dict[tuple[int, str], str] = {}
    for plate, wells in plate_wells.items():
        ordered = sorted(
            set(wells), key=lambda well: (unit_hash(seed, "control", str(plate), well), well)
        )
        if len(ordered) != 2:
            raise RuntimeError(f"plate {plate} has {len(ordered)} vehicle wells; expected 2")
        roles[(plate, ordered[0])] = "reference"
        roles[(plate, ordered[1])] = "evaluation"
    return roles


def assemble_splits(
    cells: pl.DataFrame,
    conditions: pl.DataFrame,
    compounds: pl.DataFrame,
    seed: int,
    min_treated: int = MIN_TREATED_CELLS,
) -> pl.DataFrame:
    """One row per cell: pair role, replicate-plate label, control role, and fit masks."""
    chemistry = compounds.select("compound_key", "structure_group", "fingerprint_valid")
    if chemistry["compound_key"].is_duplicated().any():
        raise RuntimeError("compound_key is not unique")
    treated = conditions.filter(
        ~pl.col("is_control") & (pl.col("compound_key") != ANCHOR_COMPOUND)
    ).join(chemistry, on="compound_key", how="left")
    if treated["structure_group"].null_count():
        raise RuntimeError("a treated compound has no structure group")
    eligible = (
        treated.group_by("cell_line", "structure_group")
        .agg(max_cells=pl.col("retained_cells").max())
        .filter(pl.col("max_cells") >= min_treated)
        .sort("cell_line", "structure_group")
    )
    pairs = list(eligible.select("cell_line", "structure_group").iter_rows())
    roles_a = assign_protocol_a(pairs, seed)

    anchor_groups = set(
        chemistry.filter(pl.col("compound_key") == ANCHOR_COMPOUND)["structure_group"].to_list()
    )
    invalid_groups = set(
        chemistry.group_by("structure_group")
        .agg(ok=pl.col("fingerprint_valid").all())
        .filter(~pl.col("ok"))["structure_group"]
        .to_list()
    )
    c_groups = sorted(
        group
        for group in chemistry["structure_group"].unique().to_list()
        if group not in anchor_groups and group not in invalid_groups
    )
    roles_c = assign_protocol_c(c_groups, seed)

    vehicles = (
        conditions.filter(pl.col("is_control"))
        .select("plate_num", "sample")
        .unique()
        .sort("plate_num", "sample")
    )
    plate_wells: dict[int, list[str]] = defaultdict(list)
    for plate, sample in vehicles.iter_rows():
        plate_wells[int(plate)].append(sample)
    control_roles = assign_control_wells(dict(plate_wells), seed)

    meta = cells.select(
        "cell_id", "sample", "cell_line", "plate_num", "compound_key", "dose_uM", "is_control"
    ).join(chemistry, on="compound_key", how="left")
    pair_roles = pl.DataFrame(
        {
            "cell_line": [line for line, _ in roles_a],
            "structure_group": [group for _, group in roles_a],
            "pair_role": [roles_a[key] for key in roles_a],
        }
    )
    group_roles = pl.DataFrame(
        {"structure_group": list(roles_c), "group_role": [roles_c[group] for group in roles_c]}
    )
    well_roles = pl.DataFrame(
        {
            "plate_num": [plate for plate, _ in control_roles],
            "sample": [sample for _, sample in control_roles],
            "well_role": [control_roles[key] for key in control_roles],
        }
    )
    meta = (
        meta.join(pair_roles, on=["cell_line", "structure_group"], how="left")
        .join(group_roles, on="structure_group", how="left")
        .join(well_roles, on=["plate_num", "sample"], how="left")
    )
    anchor = pl.col("compound_key") == ANCHOR_COMPOUND
    held_out = pl.col("plate_num") == HELD_OUT_PLATE
    meta = meta.with_columns(
        protocol_a=(
            pl.when(anchor)
            .then(pl.lit("anchor"))
            .when(pl.col("is_control"))
            .then(pl.lit("control"))
            .when(pl.col("pair_role").is_null())
            .then(pl.lit("ineligible"))
            .otherwise(pl.col("pair_role"))
        ),
        protocol_c=(
            pl.when(pl.col("is_control"))
            .then(pl.lit("control"))
            .when(anchor | pl.col("structure_group").is_in(list(invalid_groups)))
            .then(pl.lit("excluded"))
            .when(pl.col("group_role").is_null())
            .then(pl.lit("excluded"))
            .otherwise(pl.col("group_role"))
        ),
        control_role=pl.when(pl.col("is_control"))
        .then(pl.col("well_role"))
        .otherwise(pl.lit(None, dtype=pl.Utf8)),
    )
    meta = meta.with_columns(
        protocol_b=(
            pl.when(anchor)
            .then(pl.lit("anchor"))
            .when(pl.col("is_control"))
            .then(pl.lit("control"))
            .when(held_out & (pl.col("protocol_a") == "train"))
            .then(pl.lit("seen_condition_new_plate"))
            .when(held_out & pl.col("protocol_a").is_in(["validation", "test"]))
            .then(pl.lit("unseen_pair_new_plate"))
            .when(held_out)
            .then(pl.lit("ineligible"))
            .otherwise(pl.col("protocol_a"))
        ),
        fit_a=(
            ((pl.col("protocol_a") == "train") & ~pl.col("is_control") & ~held_out)
            | ((pl.col("control_role") == "reference") & ~held_out)
        ),
        fit_c=(
            ((pl.col("protocol_c") == "train") & ~pl.col("is_control") & ~held_out)
            | ((pl.col("control_role") == "reference") & ~held_out)
        ),
        seed=pl.lit(seed, dtype=pl.Int64),
    )
    splits = meta.select(
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
        "protocol_b",
        "protocol_c",
        "fit_a",
        "fit_c",
        "seed",
    ).sort("cell_id")
    check_leakage(splits)
    return splits


def check_leakage(splits: pl.DataFrame) -> None:
    """Raise if a fit mask or a pair assignment violates the contract."""
    treated = splits.filter(~pl.col("is_control") & (pl.col("protocol_a") != "anchor"))
    split_pairs = treated.group_by("cell_line", "structure_group").agg(
        n=pl.col("protocol_a").n_unique()
    )
    if split_pairs.filter(pl.col("n") > 1).height:
        raise RuntimeError("protocol A gives one line-structure pair more than one role")
    grouped = (
        treated.filter(pl.col("protocol_c") != "excluded")
        .group_by("structure_group")
        .agg(n=pl.col("protocol_c").n_unique())
    )
    if grouped.filter(pl.col("n") > 1).height:
        raise RuntimeError("protocol C splits a structure group across roles")
    if splits.filter(pl.col("fit_a") & (pl.col("plate_num") == HELD_OUT_PLATE)).height:
        raise RuntimeError("plate 14 is in the protocol A fit mask")
    if splits.filter(pl.col("fit_c") & (pl.col("plate_num") == HELD_OUT_PLATE)).height:
        raise RuntimeError("plate 14 is in the protocol C fit mask")
    if splits.filter(pl.col("fit_a") & (pl.col("control_role") == "evaluation")).height:
        raise RuntimeError("evaluation controls are in the protocol A fit mask")
    if splits.filter(
        (pl.col("protocol_a") == "anchor") & (pl.col("fit_a") | pl.col("fit_c"))
    ).height:
        raise RuntimeError("anchor cells are in a fit mask")
    illegal_fit = splits.filter(
        pl.col("fit_a")
        & ~(
            ((pl.col("protocol_a") == "train") & ~pl.col("is_control"))
            | (pl.col("control_role") == "reference")
        )
    )
    if illegal_fit.height:
        raise RuntimeError(
            "protocol A fit mask is not training treated cells plus reference controls"
        )

    wells = (
        splits.filter(pl.col("is_control")).select("plate_num", "sample", "control_role").unique()
    )
    if (
        wells.group_by("plate_num", "sample")
        .agg(n=pl.col("control_role").n_unique())
        .filter(pl.col("n") > 1)
        .height
    ):
        raise RuntimeError("a vehicle well is both reference and evaluation")
    if (
        wells.group_by("plate_num")
        .agg(n=pl.col("control_role").n_unique())
        .filter(pl.col("n") < 2)
        .height
    ):
        raise RuntimeError("a plate is missing a reference or evaluation well")

    pairs = (
        treated.filter(pl.col("protocol_a").is_in(["train", "validation", "test"]))
        .select("cell_line", "structure_group", "protocol_a")
        .unique()
    )
    train_lines = set(pairs.filter(pl.col("protocol_a") == "train")["cell_line"].to_list())
    train_groups = set(pairs.filter(pl.col("protocol_a") == "train")["structure_group"].to_list())
    for held in ("validation", "test"):
        held_pairs = pairs.filter(pl.col("protocol_a") == held)
        missing_lines = set(held_pairs["cell_line"].to_list()) - train_lines
        missing_groups = set(held_pairs["structure_group"].to_list()) - train_groups
        if missing_lines or missing_groups:
            raise RuntimeError(
                f"protocol A {held} lacks a training partner "
                f"(lines {sorted(missing_lines)[:3]}, groups {sorted(missing_groups)[:3]})"
            )
