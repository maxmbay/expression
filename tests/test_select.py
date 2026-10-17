import numpy as np
import polars as pl
import pytest

from cellresponse.data.locate import sublibrary_order
from cellresponse.data.select import CohortSpec, plan_row_groups, select_cells


def synthetic_cells(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    obs_row = 0
    wells = [("smp_1", False, 0), ("smp_2", False, 100), ("ctl_1", True, 200)]
    for sample, is_control, rg_offset in wells:
        for rg in range(40):
            for _ in range(60):
                rows.append(
                    {
                        "obs_row": obs_row,
                        "sample": sample,
                        "is_control": is_control,
                        "cell_line": f"L{rng.integers(0, 4)}",
                        "rg_id": rg + rg_offset,
                        "BARCODE_SUB_LIB_ID": f"bc{obs_row:07d}",
                    }
                )
                obs_row += 1
    return pl.DataFrame(rows)


def test_caps_treated_cells_and_keeps_all_controls():
    cells = synthetic_cells()
    spec = CohortSpec(plates=(1,), lines=("L0", "L1", "L2", "L3"), treated_cap=50, control_cap=None)
    visit = plan_row_groups(cells, spec)
    chosen = select_cells(cells, visit, spec)
    counts = chosen.group_by("sample", "cell_line", "is_control").len()
    assert counts.filter(~pl.col("is_control"))["len"].to_list() == [50] * 8
    assert chosen.filter(pl.col("is_control")).height == cells.filter(pl.col("is_control")).height


def test_visits_only_part_of_each_treated_well():
    cells = synthetic_cells()
    spec = CohortSpec(plates=(1,), lines=("L0", "L1", "L2", "L3"), treated_cap=50)
    visit = plan_row_groups(cells, spec)
    per_sample = visit.group_by("sample").len()
    treated = per_sample.filter(pl.col("sample") != "ctl_1")["len"]
    assert (treated < 40).all()
    assert per_sample.filter(pl.col("sample") == "ctl_1")["len"].item() == 40


def test_selection_is_deterministic_and_seed_dependent():
    cells = synthetic_cells()
    spec = CohortSpec(plates=(1,), lines=("L0", "L1", "L2", "L3"), treated_cap=30, seed=7)
    first = select_cells(cells, plan_row_groups(cells, spec), spec)["obs_row"].sort()
    again = select_cells(cells, plan_row_groups(cells, spec), spec)["obs_row"].sort()
    other_spec = CohortSpec(plates=(1,), lines=("L0", "L1", "L2", "L3"), treated_cap=30, seed=8)
    other = select_cells(cells, plan_row_groups(cells, other_spec), other_spec)["obs_row"].sort()
    assert first.equals(again)
    assert not first.equals(other)


def test_control_cap_applies_to_vehicle_wells():
    cells = synthetic_cells()
    spec = CohortSpec(plates=(1,), lines=("L0", "L1", "L2", "L3"), treated_cap=30, control_cap=100)
    chosen = select_cells(cells, plan_row_groups(cells, spec), spec)
    assert chosen.filter(pl.col("is_control")).group_by("cell_line").len()["len"].max() == 100


def _index(barcode_ranges):
    return pl.DataFrame(
        {
            "shard": [f"data/train-{i:05d}-of-03388.parquet" for i in range(len(barcode_ranges))],
            "row_group": [0] * len(barcode_ranges),
            "BARCODE_SUB_LIB_ID_min": [lo for lo, _ in barcode_ranges],
            "BARCODE_SUB_LIB_ID_max": [hi for _, hi in barcode_ranges],
        }
    )


def test_sublibrary_order_skips_boundary_groups():
    index = _index([("01-lib_2", "09-lib_2"), ("01-lib_2", "09-lib_5"), ("01-lib_5", "05-lib_5")])
    assert sublibrary_order(index) == ["lib_2", "lib_5"]


def test_sublibrary_order_rejects_non_contiguous_blocks():
    index = _index([("01-lib_2", "09-lib_2"), ("01-lib_5", "05-lib_5"), ("01-lib_2", "02-lib_2")])
    with pytest.raises(ValueError, match="non-contiguous"):
        sublibrary_order(index)
