"""Render audit tables into markdown reports."""

from __future__ import annotations

import polars as pl


def md_table(frame: pl.DataFrame, max_rows: int = 30, float_digits: int = 3) -> str:
    frame = frame.head(max_rows)
    def fmt(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.{float_digits}g}" if abs(value) < 1e3 else f"{value:,.0f}"
        if isinstance(value, list):
            shown = ", ".join(fmt(v) for v in value[:8])
            return shown + (", …" if len(value) > 8 else "")
        if isinstance(value, int) and abs(value) >= 10_000:
            return f"{value:,}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.iter_rows()]
    return "\n".join([header, rule, *rows])


def dict_table(values: dict, key: str = "quantity", value: str = "value") -> str:
    return md_table(pl.DataFrame({key: list(values), value: [str(v) for v in values.values()]}))
