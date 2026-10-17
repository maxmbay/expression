"""Safe parsing of Tahoe treatment strings and sparse expression rows."""

from __future__ import annotations

import ast
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

VEHICLE = "DMSO_TF"
MARKER_TOKEN = 1

_UNIT_TO_UM = {"um": 1.0, "µm": 1.0, "μm": 1.0, "nm": 1e-3, "mm": 1e3, "m": 1e6}


@dataclass(frozen=True)
class Treatment:
    name: str
    dose: float
    unit: str
    dose_uM: float


def parse_drugname_drugconc(raw: str) -> list[Treatment]:
    """Parse strings such as "[('Erdafitinib ', 0.05, 'uM')]" without evaluating code."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"unparseable treatment string: {raw!r}") from error
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected a non-empty list of treatments: {raw!r}")
    treatments = []
    for entry in value:
        if not (isinstance(entry, tuple) and len(entry) == 3):
            raise ValueError(f"expected (name, dose, unit) tuples: {raw!r}")
        name, dose, unit = entry
        if not isinstance(name, str) or not isinstance(unit, str):
            raise ValueError(f"name and unit must be strings: {raw!r}")
        if isinstance(dose, bool) or not isinstance(dose, int | float) or not np.isfinite(dose):
            raise ValueError(f"dose must be a finite number: {raw!r}")
        treatments.append(Treatment(name, float(dose), unit, to_micromolar(float(dose), unit)))
    return treatments


def to_micromolar(dose: float, unit: str) -> float:
    key = unit.strip().lower()
    if key not in _UNIT_TO_UM:
        raise ValueError(f"unknown concentration unit: {unit!r}")
    return dose * _UNIT_TO_UM[key]


def normalize_compound_name(name: str) -> str:
    """Case- and whitespace-insensitive key for detecting aliases; not a structure identity."""
    text = unicodedata.normalize("NFKC", name).strip().lower()
    return re.sub(r"\s+", " ", text)


def strip_marker(genes: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the leading marker entry by position, then validate the remaining counts.

    Raises instead of silently dropping entries, so parsing errors cannot hide as zeros.
    """
    if genes.shape != counts.shape or genes.ndim != 1:
        raise ValueError("genes and expressions must be aligned 1-D arrays")
    if genes.size == 0 or genes[0] != MARKER_TOKEN or not counts[0] < 0:
        raise ValueError(
            f"expected leading marker token {MARKER_TOKEN} with a negative value, "
            f"got token={genes[:1].tolist()} value={counts[:1].tolist()}"
        )
    genes, counts = genes[1:], counts[1:]
    if not np.all(np.isfinite(counts)):
        raise ValueError("non-finite counts after marker removal")
    if np.any(counts < 0):
        raise ValueError("negative counts after marker removal")
    if np.any(counts != np.round(counts)):
        raise ValueError("non-integer counts after marker removal")
    if np.unique(genes).size != genes.size:
        raise ValueError("duplicate gene tokens within a cell")
    return genes, counts


def strip_marker_batch(
    offsets: np.ndarray, genes: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized `strip_marker` over list-column buffers (Arrow offsets + flat values).

    Returns new offsets, int32 gene tokens, and int32 counts with each row's marker removed.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    if genes.shape != counts.shape or offsets[-1] != genes.size:
        raise ValueError("genes and expressions buffers are misaligned")
    starts = offsets[:-1]
    if np.any(offsets[1:] <= starts):
        raise ValueError("empty row: missing marker entry")
    if np.any(genes[starts] != MARKER_TOKEN) or not np.all(counts[starts] < 0):
        bad = int(np.flatnonzero((genes[starts] != MARKER_TOKEN) | ~(counts[starts] < 0))[0])
        raise ValueError(
            f"row {bad}: expected leading marker token {MARKER_TOKEN} with negative value"
        )
    keep = np.ones(genes.size, dtype=bool)
    keep[starts] = False
    genes, counts = genes[keep], counts[keep]
    new_offsets = offsets - np.arange(offsets.size, dtype=np.int64)
    if not np.all(np.isfinite(counts)):
        raise ValueError("non-finite counts after marker removal")
    if np.any(counts < 0):
        raise ValueError("negative counts after marker removal")
    if np.any(counts != np.round(counts)):
        raise ValueError("non-integer counts after marker removal")
    rows = np.repeat(np.arange(offsets.size - 1), np.diff(new_offsets))
    order = np.lexsort((genes, rows))
    if np.any((np.diff(rows[order]) == 0) & (np.diff(genes[order]) == 0)):
        raise ValueError("duplicate gene tokens within a cell")
    return new_offsets, genes.astype(np.int32), counts.astype(np.int32)
