"""Frozen gene-set membership. Scoring code lives in `cellresponse.goals`."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_PATH = Path(__file__).with_name("hallmark_2020.json")


@cache
def hallmark_2020() -> dict:
    """MSigDB Hallmark 2020 symbols, retrieved from Enrichr and committed as-is."""
    return json.loads(_PATH.read_text())
