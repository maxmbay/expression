"""Render the README figures into docs/figures/.

Measured inputs: the smoke cohort's plate-6 solvent cells (`cellresponse prepare --config
configs/smoke.yaml`) and the audited plate design (`reports/tables/plate_design.csv`). Every drug
response is synthetic, because no model is fit yet and treated cells stay unread until the splits
and goals are frozen.

    uv run python scripts/readme_figures.py [same-mean inference plate-search scope]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from flowjax.bijections import RationalQuadraticSpline  # noqa: E402
from flowjax.distributions import Normal, Transformed  # noqa: E402
from flowjax.flows import coupling_flow  # noqa: E402
from flowjax.train import fit_to_data  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
SMOKE = ROOT / "data/prepared/smoke"
DESIGN = ROOT / "reports/tables/plate_design.csv"
DPI = 200

BG = "#0f141a"
CARD = "#151b23"
WELL = "#1b222c"
INK = "#e8edf2"
MUTED = "#9aa4b2"
DIM = "#667284"
FAINT = "#3a4453"
RULE = "#29313d"
SOLVENT = "#8391a5"
CORAL = "#ff7a6b"
TEAL = "#2ec4b6"
VIOLET = "#a78bfa"
AMBER = "#f5b942"
BLUE = "#6aa9ff"
DIVERGE = LinearSegmentedColormap.from_list("diverge", [BLUE, "#1f2732", CORAL])

WANTED_Q, UNWANTED_Q = 0.95, 0.90
MARGIN = 0.35
DRAWS = 512
LINE = "A498"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.labelsize": 9,
            "axes.edgecolor": RULE,
            "axes.facecolor": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": FAINT,
            "ytick.color": FAINT,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


class Sheet:
    """A figure laid out in inches from its top-left corner."""

    def __init__(self, width: float, height: float):
        self.fig = plt.figure(figsize=(width, height))
        self.w, self.h = width, height

    def rect(self, x: float, y: float, w: float, h: float) -> list[float]:
        return [x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h]

    def axes(self, x: float, y: float, w: float, h: float) -> plt.Axes:
        return self.fig.add_axes(self.rect(x, y, w, h))

    def point(self, x: float, y: float) -> tuple[float, float]:
        return x / self.w, 1 - y / self.h

    def text(self, x: float, y: float, s: str, **kw) -> None:
        kw.setdefault("va", "top")
        self.fig.text(*self.point(x, y), s, **kw)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        face: str = CARD,
        edge: str = RULE,
        lw: float = 0.8,
        ls: str = "-",
        radius: float = 0.1,
        alpha: float = 1.0,
        z: float = -10,
    ) -> None:
        left, bottom, width, height = self.rect(x, y, w, h)
        self.fig.patches.append(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={radius / self.h}",
                transform=self.fig.transFigure,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                zorder=z,
            )
        )

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = DIM,
        lw: float = 1.2,
        style: str = "-|>",
        ls: str = "-",
        rad: float = 0.0,
    ) -> None:
        self.fig.patches.append(
            FancyArrowPatch(
                self.point(*start),
                self.point(*end),
                transform=self.fig.transFigure,
                arrowstyle=f"{style},head_length=4,head_width=2.6",
                connectionstyle=f"arc3,rad={rad}",
                color=color,
                lw=lw,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
            )
        )

    def headline(self, title: str, subtitle: str) -> None:
        self.text(MARGIN, 0.26, title, size=15.5, weight=600)
        self.text(MARGIN, 0.62, subtitle, size=10, color=MUTED, linespacing=1.45)

    def step(
        self,
        x: float,
        y: float,
        number: str,
        title: str,
        tag: str | None = None,
        tag_color: str = DIM,
    ) -> None:
        self.text(
            x + 0.2,
            y + 0.2,
            number,
            size=9.5,
            weight=600,
            color=BG,
            ha="center",
            bbox=dict(boxstyle="circle,pad=0.22", fc=INK, ec="none"),
        )
        self.text(x + 0.42, y + 0.19, title, size=11, weight=600)
        if tag:
            self.text(x + 0.42, y + 0.47, tag, size=8.5, color=tag_color)

    def footer(self, text: str) -> None:
        self.text(MARGIN, self.h - 0.14, text, size=8.5, color=DIM, va="bottom", linespacing=1.5)

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=DPI)
        plt.close(self.fig)
        print(f"wrote {path.relative_to(ROOT)}")


def signed(points: float) -> str:
    return f"{points:+.0f}".replace("-", "−")


def bare(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def solvent_cells(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.multivariate_normal([0, 0], [[1, 0.15], [0.15, 1]], n)


def goal_thresholds(solvent: np.ndarray) -> tuple[float, float]:
    wanted = np.quantile(solvent[:, 0], WANTED_Q)
    return float(wanted), float(np.quantile(solvent[:, 1], UNWANTED_Q))


def in_goal(z: np.ndarray, wanted: float, unwanted: float) -> np.ndarray:
    return (z[:, 0] >= wanted) & (z[:, 1] <= unwanted)


def draw_goal(
    ax: plt.Axes, wanted: float, unwanted: float, annotate: bool = False, label: bool = True
) -> None:
    _, x1 = ax.get_xlim()
    y0, _ = ax.get_ylim()
    ax.add_patch(
        Rectangle(
            (wanted, y0), x1 - wanted, unwanted - y0, facecolor=AMBER, alpha=0.11, lw=0, zorder=0
        )
    )
    ax.plot([wanted, wanted], [y0, unwanted], color=AMBER, lw=1.1, zorder=1)
    ax.plot([wanted, x1], [unwanted, unwanted], color=AMBER, lw=1.1, zorder=1)
    if label:
        ax.text(
            x1 - 0.1, y0 + 0.12, "goal", color=AMBER, size=9, ha="right", va="bottom", weight=600
        )
    if annotate:
        ax.text(
            wanted - 0.08,
            y0 + 0.12,
            f"{WANTED_Q * 100:.0f}th pct",
            color=AMBER,
            size=8,
            rotation=90,
            ha="right",
            va="bottom",
            alpha=0.85,
        )
        ax.text(
            x1 - 0.1,
            unwanted + 0.1,
            f"{UNWANTED_Q * 100:.0f}th pct",
            color=AMBER,
            size=8,
            ha="right",
            va="bottom",
            alpha=0.85,
        )


def program_axes(ax: plt.Axes, xlim=(-2.7, 4.3), ylim=(-2.9, 3.9), labels: bool = True) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if labels:
        ax.set_xlabel("wanted program  →", loc="right", labelpad=4)
        ax.set_ylabel("unwanted program  →", loc="top", labelpad=4)


# Measured solvent cells -------------------------------------------------------------------------


@dataclass
class Projection:
    z: np.ndarray
    line: np.ndarray
    sample: np.ndarray


def solvent_projection(components: int = 64, genes_kept: int = 2000) -> Projection:
    cells = (
        pl.read_parquet(SMOKE / "cells.parquet")
        .filter(pl.col("is_control"))
        .sort("fragment", "fragment_row")
    )
    genes = pl.read_parquet(SMOKE / "genes.parquet")
    column = np.full(int(genes["token_id"].max()) + 1, -1, dtype=np.int64)
    column[genes["token_id"].to_numpy()] = genes["column"].to_numpy()
    parts = []
    for (fragment,), rows in cells.group_by("fragment", maintain_order=True):
        shard = pl.read_parquet(SMOKE / "fragments" / fragment).with_row_index("fragment_row")
        parts.append(shard.join(rows.select("fragment_row"), on="fragment_row", how="semi"))
    counts = pl.concat(parts)
    meta = counts.select("cell_id").join(cells, on="cell_id", how="left")
    lengths = counts["genes"].list.len().to_numpy()
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    x = sparse.csr_matrix(
        (
            counts["counts"].explode().to_numpy().astype(np.float32),
            column[counts["genes"].explode().to_numpy()],
            indptr,
        ),
        shape=(counts.height, genes.height),
    )
    library = np.asarray(x.sum(1)).ravel()
    logged = (sparse.diags(1e4 / library) @ x).log1p().tocsr()
    mean = np.asarray(logged.mean(0)).ravel()
    var = np.asarray(logged.multiply(logged).mean(0)).ravel() - mean**2
    dispersion = np.where(mean > 0.01, var / np.maximum(mean, 1e-9), 0)
    dense = logged[:, np.argsort(-dispersion)[:genes_kept]].toarray()
    dense -= dense.mean(0)
    z = PCA(components, random_state=0).fit_transform(dense)
    z /= z.std(0)
    return Projection(z, meta["cell_name"].to_numpy(), meta["sample"].to_numpy())


# Display flow ------------------------------------------------------------------------------------


@dataclass
class DisplayFlows:
    flows: list[Transformed]
    wanted: float
    unwanted: float


def synthetic_response(rng: np.random.Generator, n: int) -> np.ndarray:
    responder = rng.random(n) < 0.33
    bulk = rng.multivariate_normal([0.55, 0.25], [[0.45**2, 0.05], [0.05, 0.75**2]], n)
    tail = rng.multivariate_normal([2.25, 0.05], [[0.5**2, -0.06], [-0.06, 0.45**2]], n)
    return np.where(responder[:, None], tail, bulk)


def fit_display_flows(seeds: int = 3, n: int = 3000) -> DisplayFlows:
    rng = np.random.default_rng(3)
    solvent = solvent_cells(rng, n)
    drug = synthetic_response(rng, n)
    x = np.concatenate([solvent, drug]).astype(np.float32)
    flag = np.repeat([0.0, 1.0], n)[:, None].astype(np.float32)
    flows = []
    for seed in range(seeds):
        flow = coupling_flow(
            jr.key(10 * seed),
            base_dist=Normal(jnp.zeros(2)),
            transformer=RationalQuadraticSpline(knots=8, interval=4),
            cond_dim=1,
            flow_layers=6,
            nn_width=64,
            nn_depth=2,
        )
        flow, _ = fit_to_data(
            jr.key(10 * seed + 1),
            flow,
            (x, flag),
            learning_rate=2e-3,
            max_epochs=200,
            max_patience=15,
            batch_size=256,
            show_progress=False,
        )
        flows.append(flow)
    return DisplayFlows(flows, *goal_thresholds(solvent))


def push(flow: Transformed, eps: np.ndarray, flag: float) -> np.ndarray:
    condition = jnp.full((1,), flag, dtype=jnp.float32)
    move = jax.vmap(lambda e: flow.bijection.transform(e, condition))
    return np.asarray(move(jnp.asarray(eps, dtype=jnp.float32)))


def noise(seed: int, n: int = DRAWS) -> np.ndarray:
    return np.asarray(jr.normal(jr.key(seed), (n, 2)))


# Figure 1 ----------------------------------------------------------------------------------------


def draw_same_mean(path: Path) -> None:
    rng = np.random.default_rng(7)
    n, n_share = 2400, 200_000
    population = solvent_cells(rng, n_share)
    wanted, unwanted = goal_thresholds(population)
    solvent_share = in_goal(population, wanted, unwanted).mean()
    solvent = population[:n]
    mean = np.array([1.0, 0.6])
    spread = np.diag([0.95, 1.0])
    drugs = [
        ("Tight", "spread 0.3 SD", CORAL, np.diag([0.33, 0.33]) ** 2),
        ("Broad", "spread 1 SD, independent", TEAL, spread**2),
        (
            "Broad, tied to unwanted",
            "spread 1 SD, correlated",
            VIOLET,
            spread @ np.array([[1, 0.9], [0.9, 1]]) @ spread,
        ),
    ]

    sheet = Sheet(10, 5.05)
    sheet.headline(
        "Same average, different share",
        "Three predicted responses of one cell line with the same mean (◎). A score built on the "
        "mean ties them;\nthe share of cells that land inside the goal separates them.",
    )
    gap, top, height = 0.18, 1.32, 3.12
    width = (sheet.w - 2 * MARGIN - 2 * gap) / 3
    for i, (name, note, color, cov) in enumerate(drugs):
        x = MARGIN + i * (width + gap)
        sheet.box(x, top, width, height)
        z = rng.multivariate_normal(mean, cov, n)
        hit = in_goal(z, wanted, unwanted)
        share = in_goal(rng.multivariate_normal(mean, cov, n_share), wanted, unwanted).mean()
        sheet.text(x + 0.16, top + 0.17, name, size=11, weight=600)
        sheet.text(x + 0.16, top + 0.43, note, size=9, color=MUTED)
        sheet.text(
            x + width - 0.16,
            top + 0.1,
            f"{share:.0%}",
            size=19,
            weight=600,
            color=AMBER,
            ha="right",
        )
        delta = (share - solvent_share) * 100
        versus = "same as solvent" if abs(delta) < 0.5 else f"{signed(delta)} pts vs solvent"
        sheet.text(x + width - 0.16, top + 0.47, versus, size=9, color=MUTED, ha="right")
        ax = sheet.axes(x + 0.3, top + 0.8, width - 0.42, height - 1.08)
        program_axes(ax, labels=i == 0)
        ax.scatter(*solvent.T, s=2.2, color=SOLVENT, alpha=0.26, lw=0, rasterized=True)
        ax.scatter(*z[~hit].T, s=3, color=color, alpha=0.5, lw=0, rasterized=True)
        ax.scatter(*z[hit].T, s=4.5, color=AMBER, alpha=0.95, lw=0, rasterized=True)
        draw_goal(ax, wanted, unwanted, annotate=i == 0)
        ax.scatter(*mean, s=95, facecolor="none", edgecolor=INK, lw=1.4, zorder=5)
        ax.scatter(*mean, s=9, color=INK, zorder=5)
    sheet.footer(
        f"Synthetic populations; {n:,} cells drawn per panel. Axes are in standard deviations of "
        f"the line's solvent cells (gray), {solvent_share:.1%} of which fall in the goal.\nGoal: "
        f"wanted program above the solvent {WANTED_Q * 100:.0f}th percentile and unwanted program "
        f"below its {UNWANTED_Q * 100:.0f}th. Amber cells are inside it."
    )
    sheet.save(path)


# Figure 2 ----------------------------------------------------------------------------------------


def draw_inference(path: Path, projection: Projection, display: DisplayFlows) -> None:
    sheet = Sheet(10, 9.7)
    sheet.headline(
        "Inference: one line, one drug, one dose",
        "A call reads the line's own solvent cells, the drug's structure, and the dose, and "
        "draws a population of cells.\nIt returns the share of those cells in the goal and, "
        "separately, how far to trust it. Nothing in the call names the line or the drug.",
    )
    top, height, gap = 1.3, 3.6, 0.2
    solvent_panel(sheet, projection, MARGIN, top, 4.05, height)
    condition_panel(sheet, projection, MARGIN + 4.05 + gap, top, 9.3 - 4.05 - gap, height)
    second, low = top + height + gap, 3.95
    flow_panel(sheet, display, MARGIN, second, 5.95, low)
    readout_panel(sheet, display, MARGIN + 5.95 + gap, second, 9.3 - 5.95 - gap, low)
    sheet.footer(
        "Panel 1 is measured: the smoke cohort's plate-6 solvent cells, on a PCA fit to those "
        "cells for display (the real map T is fit on training cells).\nPanels 3–4 use a 2-D "
        "spline flow (6 couplings, 8 bins) fit three times to a synthetic response; the real "
        "flow is 64-dimensional. Panel 2's structure bits are illustrative."
    )
    sheet.save(path)


def reference_cells(p: Projection) -> np.ndarray:
    mine = p.line == LINE
    return mine & (p.sample == sorted(set(p.sample[mine]))[0])


def solvent_panel(sheet: Sheet, p: Projection, x: float, y: float, w: float, h: float) -> None:
    sheet.box(x, y, w, h)
    sheet.step(x, y, "1", "Read the line's solvent well", "measured · 4 lines on plate 6", TEAL)
    ax = sheet.axes(x + 0.3, y + 0.75, w - 0.45, h - 1.75)
    for name in dict.fromkeys(p.line):
        if name == LINE:
            continue
        cloud = p.z[p.line == name, :2]
        ax.scatter(*cloud.T, s=1.6, color="#465163", alpha=0.6, lw=0, rasterized=True)
        ax.text(
            *np.median(cloud, 0),
            name,
            size=8.5,
            color=MUTED,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=CARD, ec="none", alpha=0.85),
        )
    reference = reference_cells(p)
    held = (p.line == LINE) & ~reference
    ax.scatter(*p.z[held, :2].T, s=1.6, color="#465163", alpha=0.6, lw=0, rasterized=True)
    ax.scatter(*p.z[reference, :2].T, s=2.6, color="#b7c3d3", alpha=0.75, lw=0, rasterized=True)
    m0, s0 = p.z[reference].mean(0), p.z[reference].std(0)
    ax.errorbar(m0[0], m0[1], xerr=s0[0], yerr=s0[1], color=INK, lw=1.3, capsize=3, zorder=5)
    ax.scatter(*m0[:2], s=80, facecolor=CARD, edgecolor=INK, lw=1.5, zorder=6)
    label_x = np.quantile(p.z[reference, 0], 0.99) + 0.25
    ax.plot([m0[0] + s0[0] + 0.06, label_x - 0.06], [m0[1], m0[1]], color=MUTED, lw=0.8, zorder=4)
    ax.text(label_x, m0[1] + 0.06, LINE, size=9.5, color=INK, va="bottom", weight=600)
    ax.text(label_x, m0[1] - 0.06, "m0 ± σ0", size=9, color=INK, va="top")
    ax.set_xlabel("PC 1", labelpad=2)
    ax.set_ylabel("PC 2", labelpad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    sheet.text(
        x + 0.3,
        y + h - 0.78,
        f"m0 and σ0: mean and spread of {reference.sum()} cells from one\nsolvent well, in "
        "each of 64 coordinates (2 shown). The line's\nsecond solvent well stays out; it "
        "measures the real no-drug share.",
        size=8.5,
        color=MUTED,
        linespacing=1.4,
    )


def condition_panel(sheet: Sheet, p: Projection, x: float, y: float, w: float, h: float) -> None:
    sheet.box(x, y, w, h)
    sheet.step(
        x,
        y,
        "2",
        "Build the condition h",
        "258 numbers, read by every coupling. No line, drug, or plate ID.",
    )
    reference = reference_cells(p)
    m0 = p.z[reference].mean(0)
    log_s0 = np.log(p.z[reference].std(0))
    rng = np.random.default_rng(11)
    left, label_w = x + 0.22, 1.62
    vis_w = w - 0.22 - label_w - 0.55
    rows = [
        ("m0", "reference-cell mean", "64"),
        ("log σ0", "reference-cell spread", "64"),
        ("structure", "Morgan bits, radius 2", "128"),
        ("dose", "log1p(dose / 0.05 µM)", "1"),
        ("flag", "drug or solvent", "1"),
    ]
    row_top, row_h = y + 0.86, 0.53
    for i, (name, note, dims) in enumerate(rows):
        ry = row_top + i * row_h
        sheet.text(left, ry, name, size=9.5, weight=600)
        sheet.text(left, ry + 0.2, note, size=8.5, color=MUTED)
        sheet.text(x + w - 0.2, ry + 0.06, dims, size=9, color=MUTED, ha="right")
        vx, vy, vh = left + label_w, ry + 0.02, 0.24
        if name in ("m0", "log σ0"):
            ax = sheet.axes(vx, vy, vis_w, vh)
            ax.imshow(
                (m0 if name == "m0" else log_s0)[None, :],
                aspect="auto",
                cmap=DIVERGE,
                vmin=-1.6,
                vmax=1.6,
            )
            bare(ax)
        elif name == "structure":
            bits_w = vis_w * 0.48
            ax = sheet.axes(vx, vy, bits_w, vh)
            ax.set_facecolor(WELL)
            ax.vlines(rng.choice(2048, 46, replace=False), 0, 1, color=INK, lw=0.7, alpha=0.85)
            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 1)
            bare(ax)
            sheet.text(vx, vy + vh + 0.04, "2,048 bits", size=7.5, color=DIM)
            sheet.arrow(
                (vx + bits_w + 0.07, vy + vh / 2), (vx + bits_w + 0.33, vy + vh / 2), color=MUTED
            )
            proj_x = vx + bits_w + 0.4
            ax = sheet.axes(proj_x, vy, vx + vis_w - proj_x, vh)
            ax.imshow(
                rng.normal(0, 0.8, (1, 128)), aspect="auto", cmap=DIVERGE, vmin=-1.6, vmax=1.6
            )
            bare(ax)
            sheet.text(proj_x, vy + vh + 0.04, "learned projection", size=7.5, color=DIM)
        elif name == "dose":
            ax = sheet.axes(vx, vy - 0.1, vis_w, vh + 0.24)
            bare(ax)
            ax.set_xlim(0.3, 5.0)
            ax.set_ylim(-1, 1)
            ax.plot([0.4, 4.9], [0, 0], color=FAINT, lw=1)
            for dose in (0.05, 0.5, 5.0):
                value, plate = np.log1p(dose / 0.05), dose == 5.0
                color = AMBER if plate else MUTED
                ax.scatter(value, 0, s=36 if plate else 16, color=color, zorder=3)
                ax.text(
                    value,
                    0.3,
                    f"{dose:g} µM" + (", plate 6" if plate else ""),
                    size=8,
                    ha="center",
                    va="bottom",
                    color=color,
                )
                ax.text(value, -0.3, f"{value:.2f}", size=8, ha="center", va="top", color=color)
        else:
            cx = vx
            for text, color, active in (("drug  1", TEAL, True), ("solvent  0", SOLVENT, False)):
                sheet.text(
                    cx + 0.06,
                    vy + vh / 2,
                    text,
                    size=8.5,
                    va="center",
                    weight=600,
                    color=BG if active else color,
                    bbox=dict(
                        boxstyle="round,pad=0.3,rounding_size=0.5",
                        fc=color if active else "none",
                        ec=color,
                        lw=1,
                    ),
                )
                cx += 0.95


def flow_panel(sheet: Sheet, d: DisplayFlows, x: float, y: float, w: float, h: float) -> None:
    sheet.box(x, y, w, h)
    sheet.step(
        x,
        y,
        "3",
        "Draw cells: noise through the flow, then add m0",
        "a 2-D flow fit to a synthetic response stands in for the 64-D one",
    )
    flow = d.flows[0]
    side, ay = 2.2, y + 0.82
    left_x, right_x = x + 0.22, x + w - side - 0.2
    eps = noise(100)
    drug = push(flow, eps, 1.0)
    solvent = push(flow, noise(200), 0.0)
    hit = in_goal(drug, d.wanted, d.unwanted)

    ax = sheet.axes(left_x, ay, side, side)
    lim = 3.3
    grid = np.linspace(-lim, lim, 260)
    gx, gy = np.meshgrid(grid, grid)
    mapped = push(flow, np.column_stack([gx.ravel(), gy.ravel()]), 1.0)
    region = in_goal(mapped, d.wanted, d.unwanted).reshape(gx.shape)
    ax.contourf(gx, gy, region, levels=[0.5, 1.5], colors=[AMBER], alpha=0.16)
    ax.contour(gx, gy, region, levels=[0.5], colors=[AMBER], linewidths=1.0)
    for ring, angle in polar_grid():
        ax.plot(*ring.T, color=FAINT, lw=0.7, zorder=1)
        ax.plot(*angle.T, color=FAINT, lw=0.7, zorder=1)
    ax.scatter(*eps[~hit].T, s=4, color=TEAL, alpha=0.55, lw=0, zorder=2)
    ax.scatter(*eps[hit].T, s=6, color=AMBER, lw=0, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    bare(ax)
    caption_y = ay + side + 0.12
    sheet.text(left_x, caption_y, f"noise ε ~ N(0, I), {DRAWS} draws", size=9, weight=600)
    sheet.text(
        left_x,
        caption_y + 0.22,
        "Amber area: the noise this flow\nsends into the goal.",
        size=8.5,
        color=MUTED,
        linespacing=1.35,
    )

    gap_left, gap_right = left_x + side + 0.1, right_x - 0.1
    lane = ay + side / 2
    size, step = 0.11, (gap_right - gap_left - 0.2) / 6
    sheet.arrow((gap_left, lane), (gap_right, lane), color=MUTED, lw=1.1)
    centres = [gap_left + 0.1 + step * (i + 0.5) for i in range(6)]
    hub = ((centres[0] + centres[-1]) / 2, lane - 0.62)
    sheet.text(hub[0], hub[1] - 0.06, "h", size=10.5, weight=600, ha="center", va="bottom")
    for cx in centres:
        sheet.box(
            cx - size / 2,
            lane - size / 2,
            size,
            size,
            face=WELL,
            edge=INK,
            lw=0.9,
            radius=0.025,
            z=5,
        )
        sheet.arrow(hub, (cx, lane - size / 2 - 0.03), color=FAINT, lw=0.7)
    sheet.text(
        hub[0],
        lane + 0.18,
        "6 spline couplings\nthen + m0",
        size=8.5,
        color=MUTED,
        ha="center",
        linespacing=1.35,
    )

    ax = sheet.axes(right_x, ay, side, side)
    program_axes(ax, xlim=(-2.9, 4.4), ylim=(-3.0, 3.6), labels=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for ring, angle in polar_grid():
        ax.plot(*push(flow, ring, 1.0).T, color=FAINT, lw=0.7, zorder=1)
        ax.plot(*push(flow, angle, 1.0).T, color=FAINT, lw=0.7, zorder=1)
    ax.scatter(*solvent.T, s=3, color=SOLVENT, alpha=0.35, lw=0, zorder=2)
    ax.scatter(*drug[~hit].T, s=4, color=TEAL, alpha=0.6, lw=0, zorder=3)
    ax.scatter(*drug[hit].T, s=6, color=AMBER, lw=0, zorder=4)
    draw_goal(ax, d.wanted, d.unwanted)
    ax.text(0.02, 0.02, "wanted →", transform=ax.transAxes, size=8, color=DIM)
    ax.text(0.02, 0.98, "unwanted ↑", transform=ax.transAxes, size=8, color=DIM, va="top")
    sheet.text(right_x, caption_y, "cells Z", size=9, weight=600)
    sheet.text(
        right_x,
        caption_y + 0.22,
        f"Drug call (teal), solvent call (gray),\n{DRAWS} cells each. Grid: the circles, warped.",
        size=8.5,
        color=MUTED,
        linespacing=1.35,
    )


def polar_grid() -> list[tuple[np.ndarray, np.ndarray]]:
    t = np.linspace(0, 2 * np.pi, 240)
    r = np.linspace(0, 3.2, 120)
    pairs = []
    for i, radius in enumerate((1.0, 2.0, 3.0)):
        theta = i * np.pi / 3
        ring = np.column_stack([radius * np.cos(t), radius * np.sin(t)])
        angle = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        pairs.append((ring, angle))
    for theta in np.arange(3, 6) * np.pi / 3:
        angle = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        pairs.append((angle[:1], angle))
    return pairs


@dataclass
class Readout:
    drug: np.ndarray
    solvent: np.ndarray

    @property
    def utility(self) -> np.ndarray:
        return self.drug - self.solvent

    @property
    def monte_carlo(self) -> float:
        var = self.drug * (1 - self.drug) / DRAWS + self.solvent * (1 - self.solvent) / DRAWS
        return float(np.sqrt(var.sum()) / len(var))


def readout(d: DisplayFlows) -> Readout:
    drug, solvent = [], []
    for seed, flow in enumerate(d.flows):
        drug.append(in_goal(push(flow, noise(100 + seed), 1.0), d.wanted, d.unwanted).mean())
        solvent.append(in_goal(push(flow, noise(200 + seed), 0.0), d.wanted, d.unwanted).mean())
    return Readout(np.array(drug), np.array(solvent))


def readout_panel(sheet: Sheet, d: DisplayFlows, x: float, y: float, w: float, h: float) -> None:
    sheet.box(x, y, w, h)
    sheet.step(x, y, "4", "Score it", "the same goal for every call")
    r = readout(d)
    ax = sheet.axes(x + 0.82, y + 0.9, w - 1.05, 1.0)
    for row, values, color, name in ((0, r.drug, TEAL, "drug"), (1, r.solvent, SOLVENT, "no drug")):
        ax.plot(
            [values.min(), values.max()],
            [row, row],
            color=color,
            lw=2.2,
            alpha=0.5,
            solid_capstyle="round",
        )
        ax.scatter(values, [row] * len(values), s=16, color=color, zorder=3)
        ax.scatter(values.mean(), row, s=60, facecolor=CARD, edgecolor=color, lw=1.6, zorder=4)
        ax.text(-0.012, row, name, size=8.5, color=MUTED, ha="right", va="center")
    ax.annotate(
        "",
        xy=(r.drug.mean(), 0.5),
        xytext=(r.solvent.mean(), 0.5),
        arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.22", color=AMBER, lw=1.3),
    )
    ax.set_xlim(0, 0.4)
    ax.set_ylim(1.6, -0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4], ["0", "10", "20", "30", "40%"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=2)
    rows = [
        ("utility", f"{signed(r.utility.mean() * 100)} pts", AMBER),
        ("share in goal", f"{r.drug.mean():.0%} drug, {r.solvent.mean():.0%} no drug", INK),
        ("seed spread", f"± {r.utility.std(ddof=1) * 100:.1f} pts over 3 flows", INK),
        ("Monte Carlo", f"± {r.monte_carlo * 100:.1f} pts, {DRAWS} draws each", INK),
        ("support", "fingerprint ✓   measured dose ✓", INK),
        ("refused", "no fingerprint", CORAL),
        ("flagged", "dose off {0.05, 0.5, 5} µM", AMBER),
    ]
    sheet.fig.lines.append(
        plt.Line2D(
            [sheet.point(x + 0.25, 0)[0], sheet.point(x + w - 0.25, 0)[0]],
            [sheet.point(0, y + 2.22)[1]] * 2,
            transform=sheet.fig.transFigure,
            color=RULE,
            lw=0.8,
        )
    )
    for i, (key, value, color) in enumerate(rows):
        ry = y + 2.36 + i * 0.22
        sheet.text(x + 0.25, ry, key, size=8.5, color=MUTED)
        sheet.text(x + 1.2, ry, value, size=8.5, color=color, weight=600 if i == 0 else 400)


# Figure 3 ----------------------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    mean: tuple[float, float]
    sd: tuple[float, float]
    rho: float = 0.0
    responders: float = 0.0
    tail: tuple[float, float] = (2.4, 0.1)

    def draw(self, rng: np.random.Generator, n: int, jitter: float = 0.0) -> np.ndarray:
        mean = np.array(self.mean) + rng.normal(0, jitter, 2)
        sd = np.array(self.sd) * np.exp(rng.normal(0, jitter, 2))
        cov = np.diag(sd) @ np.array([[1, self.rho], [self.rho, 1]]) @ np.diag(sd)
        z = rng.multivariate_normal(mean, cov, n)
        if self.responders:
            tail = rng.multivariate_normal(
                np.array(self.tail) + rng.normal(0, jitter, 2), np.diag([0.45, 0.5]) ** 2, n
            )
            z = np.where((rng.random(n) < self.responders)[:, None], tail, z)
        return z


CANDIDATES = [
    Candidate("drug 01", (0.3, 0.2), (0.6, 0.8), responders=0.3),
    Candidate("drug 02", (0.9, 0.2), (1.0, 0.8)),
    Candidate("drug 03", (1.38, 0.3), (0.2, 0.3)),
    Candidate("drug 04", (1.2, 1.3), (0.9, 0.9), rho=0.8),
    Candidate("drug 05", (0.6, 0.0), (0.8, 0.9)),
    Candidate("drug 06", (0.05, 0.05), (1.0, 1.0)),
    Candidate("drug 07", (-0.1, 0.2), (0.9, 1.0)),
    Candidate("drug 08", (0.3, 1.6), (0.8, 0.7)),
    Candidate("drug 09", (-0.9, -0.2), (0.7, 0.8)),
    Candidate("drug 10", (0.2, 0.0), (0.4, 0.5)),
    Candidate("drug 11", (0.3, 0.3), (1.4, 1.3), rho=0.3),
    Candidate("drug 12", (-1.6, 0.1), (0.6, 0.7)),
]
NO_DRUG = Candidate("no drug", (0.0, 0.0), (1.0, 1.0), rho=0.15)


def draw_plate_search(path: Path) -> None:
    rng = np.random.default_rng(5)
    wanted, unwanted = goal_thresholds(solvent_cells(np.random.default_rng(7), 200_000))
    rows = []
    for c in [*CANDIDATES, NO_DRUG]:
        seeds = [in_goal(c.draw(rng, DRAWS, 0.05), wanted, unwanted).mean() for _ in range(3)]
        rows.append((c, np.array(seeds), c.draw(rng, 260)))
    base = next(seeds for c, seeds, _ in rows if c is NO_DRUG).mean()
    rows.sort(key=lambda row: -(row[1].mean() - base))
    by_mean = max((row for row in rows if row[0] is not NO_DRUG), key=lambda r: r[2][:, 0].mean())

    sheet = Sheet(10, 7.75)
    sheet.headline(
        "Optimization: every drug on the plate, one reference, keep the best",
        "A query is one line on one plate, so every candidate shares the plate's dose and the "
        "same solvent reference.\nEach row is one inference call. The no-drug row is the same "
        "call with the solvent flag, and its share is subtracted from every drug's.",
    )
    x, y, w, h = MARGIN, 1.3, 9.3, 5.75
    sheet.box(x, y, w, h)
    head = y + 0.22
    strip_x, strip_w = x + 1.55, 4.6
    share_x, share_w = x + 6.45, 1.95
    xlim = (-3.2, 4.6)
    sheet.text(x + 0.25, head, "candidate", size=9, color=MUTED)
    sheet.text(strip_x, head, "predicted cells along the wanted program", size=9, color=MUTED)
    threshold_x = strip_x + strip_w * (wanted - xlim[0]) / (xlim[1] - xlim[0])
    sheet.text(threshold_x + 0.06, head, "threshold", size=9, color=AMBER)
    sheet.text(share_x, head, "share in the goal", size=9, color=MUTED)
    sheet.text(x + w - 0.25, head, "vs no drug, pts", size=9, color=MUTED, ha="right")
    body_y, body_h = y + 0.62, h - 1.25
    n = len(rows)
    strip = sheet.axes(strip_x, body_y, strip_w, body_h)
    share = sheet.axes(share_x, body_y, share_w, body_h)
    for ax in (strip, share):
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    strip.set_xlim(*xlim)
    strip.set_xticks([])
    strip.axvline(wanted, color=AMBER, lw=1, ls=(0, (3, 3)), zorder=1)
    row_h = body_h / n
    for i, (c, seeds, cells) in enumerate(rows):
        ry = body_y + i * row_h
        pick = i == 0
        is_base = c is NO_DRUG
        if pick:
            sheet.box(
                x + 0.1,
                ry + 0.02,
                w - 0.2,
                row_h - 0.04,
                face=AMBER,
                edge="none",
                alpha=0.07,
                radius=0.05,
                z=-5,
            )
        name_color = AMBER if pick else (SOLVENT if is_base else INK)
        sheet.text(
            x + 0.25,
            ry + row_h / 2,
            c.name,
            size=9.5,
            va="center",
            color=name_color,
            weight=600 if pick or is_base else 400,
        )
        if pick:
            sheet.text(
                x + 1.0,
                ry + row_h / 2,
                "pick",
                size=8,
                va="center",
                color=BG,
                weight=600,
                bbox=dict(boxstyle="round,pad=0.25", fc=AMBER, ec="none"),
            )
        if c is by_mean[0]:
            sheet.text(
                x + 0.92,
                ry + row_h / 2,
                "top mean",
                size=8,
                va="center",
                color=CORAL,
                weight=600,
                bbox=dict(boxstyle="round,pad=0.25", fc="none", ec=CORAL),
            )
        jitter = rng.uniform(-0.3, 0.3, len(cells))
        goal = in_goal(cells, wanted, unwanted)
        blocked = (cells[:, 0] >= wanted) & ~goal
        rest = ~goal & ~blocked
        tone = SOLVENT if is_base else "#55637a"
        strip.scatter(
            cells[rest, 0], i + jitter[rest], s=3, color=tone, alpha=0.6, lw=0, rasterized=True
        )
        strip.scatter(
            cells[blocked, 0],
            i + jitter[blocked],
            s=5,
            color=CORAL,
            alpha=0.9,
            lw=0,
            rasterized=True,
        )
        strip.scatter(cells[goal, 0], i + jitter[goal], s=5, color=AMBER, lw=0, rasterized=True)
        strip.plot(
            [cells[:, 0].mean()] * 2,
            [i - 0.36, i + 0.36],
            color=CORAL if c is by_mean[0] else INK,
            lw=2,
            solid_capstyle="round",
        )
        color = AMBER if pick else (SOLVENT if is_base else INK)
        share.plot(
            [seeds.min(), seeds.max()],
            [i, i],
            color=color,
            lw=2,
            alpha=0.45,
            solid_capstyle="round",
        )
        share.scatter(seeds, [i] * 3, s=9, color=color, zorder=3)
        share.scatter(seeds.mean(), i, s=42, facecolor=CARD, edgecolor=color, lw=1.4, zorder=4)
        u = (seeds.mean() - base) * 100
        sheet.text(
            x + w - 0.25,
            ry + row_h / 2,
            "0" if is_base else signed(u),
            size=9.5,
            va="center",
            ha="right",
            color=color if u > 0.5 or is_base else DIM,
            weight=600 if pick else 400,
        )
        if is_base and i < n - 1:
            line_y = ry + row_h
            sheet.fig.lines.append(
                plt.Line2D(
                    [sheet.point(x + 0.1, 0)[0], sheet.point(x + w - 0.1, 0)[0]],
                    [sheet.point(0, line_y)[1]] * 2,
                    transform=sheet.fig.transFigure,
                    color=SOLVENT,
                    lw=0.8,
                    ls=(0, (2, 3)),
                )
            )
            sheet.text(
                x + 0.25,
                line_y,
                "below: worse than leaving the cells in solvent",
                size=8,
                color=SOLVENT,
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=CARD, ec="none"),
            )
            sheet.box(
                x + 0.1,
                line_y + 0.02,
                w - 0.2,
                body_y + body_h - line_y - 0.02,
                face="#0c1015",
                edge="none",
                alpha=0.7,
                radius=0.05,
                z=-5,
            )
    share.axvline(base, color=SOLVENT, lw=1, ls=(0, (2, 3)), zorder=1)
    share.set_xlim(-0.012, 0.36)
    share.spines["bottom"].set_bounds(0, 0.36)
    share.set_xticks([0, 0.1, 0.2, 0.3], ["0", "10", "20", "30%"])
    share.tick_params(length=2)
    legend_y = y + h - 0.36
    items = [
        (AMBER, "●", "in the goal", 1.05),
        (CORAL, "●", "wanted on, unwanted over limit", 2.25),
        ("#55637a", "●", "neither", 0.75),
        (INK, "|", "mean", 0.6),
    ]
    lx = strip_x
    for color, mark, label, width in items:
        sheet.text(lx, legend_y, mark, size=9, color=color, va="center", weight=600)
        sheet.text(lx + 0.15, legend_y, label, size=8.5, color=MUTED, va="center")
        lx += width
    sheet.text(
        share_x, legend_y, "dots: 3 seeds, ring: their mean", size=8.5, color=MUTED, va="center"
    )
    sheet.footer(
        "Synthetic predictions, 3 seeds × 512 draws per candidate. A test query ranks only the "
        "drugs held out for that line on that plate (at least 10).\nThe ranking is stored and "
        "hashed before treated wells are opened; the pick is then scored on its measured cells "
        "against the held-out solvent well."
    )
    sheet.save(path)


# Figure 4 ----------------------------------------------------------------------------------------


def draw_scope(path: Path) -> None:
    design = pl.read_csv(DESIGN)
    sheet = Sheet(10, 6.6)
    sheet.headline(
        "What the search may range over, now and later",
        "Each plate ran one compound group at one dose, next to its own solvent wells, so today "
        "the argmax stays on one plate.\nCrossing plates needs evidence that one solvent "
        "reference means the same thing on all of them.",
    )
    x, y, w, h = MARGIN, 1.3, 5.85, 4.62
    sheet.box(x, y, w, h)
    sheet.text(x + 0.22, y + 0.19, "Where the argmax may look", size=11, weight=600)
    sheet.text(
        x + 0.22,
        y + 0.47,
        "the audited design: each plate is one compound group at one dose",
        size=8.5,
        color=TEAL,
    )
    cell_w, cell_h, gx, gy, repeat_gap = 1.02, 0.52, 0.1, 0.12, 0.42
    origin_x, origin_y = x + 0.92, y + 1.2
    columns = [origin_x + j * (cell_w + gx) for j in range(3)]
    columns.append(columns[-1] + cell_w + repeat_gap)
    cohort = {4, 5, 6, 7, 8, 9, 14}
    for cx, label in zip(columns, ["0.05 µM", "0.5 µM", "5 µM", "repeat"], strict=True):
        sheet.text(
            cx + cell_w / 2, origin_y - 0.1, label, size=8.5, color=MUTED, ha="center", va="bottom"
        )
    for i, group in enumerate("ABCD"):
        cy = origin_y + i * (cell_h + gy)
        sheet.text(
            origin_x - 0.14,
            cy + cell_h / 2,
            f"group {group}",
            size=8.5,
            color=MUTED,
            ha="right",
            va="center",
        )
        plates = design.filter(pl.col("plate_group") == group).sort("plate_num")
        for row in plates.iter_rows(named=True):
            column = 3 if row["plate_num"] == 14 else (row["plate_num"] - 1) % 3
            plate_cell(sheet, columns[column], cy, cell_w, cell_h, row, row["plate_num"] in cohort)
    mixed = design.filter(pl.col("plate_num") == 13).row(0, named=True)
    my = origin_y + 4 * (cell_h + gy)
    plate_cell(
        sheet,
        origin_x,
        my,
        3 * cell_w + 2 * gx,
        cell_h,
        mixed,
        False,
        note=f"{mixed['panel_compounds']} drugs at all three doses, mostly repeats",
    )
    sheet.text(
        origin_x - 0.14, my + cell_h / 2, "mixed", size=8.5, color=MUTED, ha="right", va="center"
    )

    b_y = origin_y + cell_h + gy
    sheet.box(
        origin_x - 0.05,
        b_y - 0.05,
        3 * cell_w + 2 * gx + 0.1,
        cell_h + 0.1,
        face="none",
        edge=VIOLET,
        lw=1.3,
        ls=(0, (4, 3)),
        radius=0.08,
        z=6,
    )
    sheet.text(
        origin_x + 0.04,
        b_y - 0.05,
        "later",
        size=8,
        color=BG,
        weight=600,
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc=VIOLET, ec="none"),
        zorder=8,
    )
    six_x = columns[2]
    sheet.box(
        six_x - 0.025,
        b_y - 0.025,
        cell_w + 0.05,
        cell_h + 0.05,
        face="none",
        edge=AMBER,
        lw=1.8,
        radius=0.07,
        z=7,
    )
    sheet.text(
        six_x + cell_w - 0.04,
        b_y - 0.025,
        "today",
        size=8,
        color=BG,
        weight=600,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc=AMBER, ec="none"),
        zorder=8,
    )
    sheet.arrow(
        (six_x + cell_w + 0.06, b_y + cell_h / 2),
        (columns[3] - 0.06, b_y + cell_h / 2),
        color=INK,
        lw=1.1,
        style="<|-|>",
    )
    sheet.text(
        columns[3] + 0.02,
        b_y + cell_h + 0.14,
        "same drugs, same\ndose: does one \nreference transfer?",
        size=8,
        color=MUTED,
        linespacing=1.35,
    )
    legend_y = y + h - 0.24
    sheet.text(x + 0.22, legend_y, "●", size=8, color=TEAL, va="center")
    sheet.text(
        x + 0.38,
        legend_y,
        "Adagrasib 0.05 µM, on every plate: the cross-plate anchor",
        size=8.5,
        color=MUTED,
        va="center",
    )
    sheet.text(
        x + w - 0.22,
        legend_y,
        "dimmed: not in the v0.1 cohort",
        size=8.5,
        color=DIM,
        ha="right",
        va="center",
    )
    next_well_panel(sheet, x + w + 0.2, y, 9.3 - w - 0.2, h)
    sheet.footer(
        "Left: the audited design (reports/tables/plate_design.csv). Every plate also has two "
        "solvent wells. Right: synthetic scores for illustration.\nThe later search opens only "
        "if plate 14 matches plate 6 and the Adagrasib anchor holds across plates; plate 13 is "
        "the check across groups."
    )
    sheet.save(path)


def plate_cell(
    sheet: Sheet,
    x: float,
    y: float,
    w: float,
    h: float,
    row: dict,
    active: bool,
    note: str | None = None,
) -> None:
    alpha = 1.0 if active else 0.42
    sheet.box(x, y, w, h, face=WELL, edge=RULE if active else "#20262f", radius=0.06, z=1)
    sheet.text(
        x + 0.1, y + 0.1, f"plate {row['plate_num']}", size=9, weight=600, color=to_rgba(INK, alpha)
    )
    sheet.text(
        x + 0.1,
        y + 0.3,
        note or f"{row['panel_compounds']} drugs",
        size=8,
        color=to_rgba(MUTED, alpha),
    )
    if row["anchor_wells"]:
        sheet.text(x + w - 0.1, y + 0.1, "●", size=7, color=to_rgba(TEAL, alpha), ha="right")


def next_well_panel(sheet: Sheet, x: float, y: float, w: float, h: float) -> None:
    sheet.box(x, y, w, h)
    sheet.text(x + 0.22, y + 0.19, "Later: the next well to measure", size=11, weight=600)
    sheet.text(x + 0.22, y + 0.47, "score = utility − λ × seed spread", size=8.5, color=MUTED)
    rng = np.random.default_rng(21)
    lam, n = 1.5, 46
    spread = rng.gamma(2.4, 1.0, n)
    utility = rng.normal(2.5, 3.5, n) + 0.8 * spread
    spread[:2], utility[:2] = (8.6, 1.7), (21.0, 15.5)
    score = utility - lam * spread
    over = (score > score[1] - 1.5) & (np.arange(n) > 1)
    utility[over] -= score[over] - (score[1] - 1.5)
    top, best = 0, 1
    ax = sheet.axes(x + 0.62, y + 0.85, w - 0.82, h - 1.95)
    xs = np.array([0, 10.5])
    for level in score[best] - 4 * np.arange(1, 8):
        ax.plot(xs, level + lam * xs, color=FAINT, lw=0.7, zorder=0)
    ax.plot(xs, score[best] + lam * xs, color=AMBER, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.scatter(spread[2:], utility[2:], s=14, color="#55637a", lw=0, zorder=2)
    ax.scatter(spread[top], utility[top], s=60, facecolor=CARD, edgecolor=CORAL, lw=1.6, zorder=3)
    ax.scatter(spread[best], utility[best], s=60, color=AMBER, zorder=3)
    box = dict(boxstyle="round,pad=0.2", fc=CARD, ec="none")
    ax.annotate(
        "highest utility,\nseeds disagree",
        (spread[top], utility[top]),
        xytext=(10.2, 15.2),
        textcoords="data",
        ha="right",
        va="top",
        size=8,
        color=CORAL,
        bbox=box,
        arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.8, shrinkB=5),
    )
    ax.annotate(
        "best score:\nmeasure next",
        (spread[best], utility[best]),
        xytext=(0.2, 19.6),
        textcoords="data",
        ha="left",
        va="bottom",
        size=8,
        color=AMBER,
        bbox=box,
        arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.8, shrinkB=5),
    )
    ax.set_xlim(*xs)
    ax.set_ylim(-6, 25)
    ax.set_xlabel("seed spread (pts)")
    ax.set_ylabel("predicted utility (pts)")
    ax.tick_params(length=2)
    sheet.text(
        x + 0.22,
        y + h - 0.5,
        f"Dots: drug × dose pairs, one reference.\nLines: equal "
        f"score at λ = {lam:g}, fixed on validation.",
        size=8.5,
        color=MUTED,
        linespacing=1.4,
    )


# -------------------------------------------------------------------------------------------------


def main() -> None:
    names = ["same-mean", "inference", "plate-search", "scope"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figures", nargs="*", choices=names, default=None)
    chosen = parser.parse_args().figures or names
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    if "same-mean" in chosen:
        draw_same_mean(OUT / "same-mean.png")
    if "inference" in chosen:
        draw_inference(OUT / "inference.png", solvent_projection(), fit_display_flows())
    if "plate-search" in chosen:
        draw_plate_search(OUT / "plate-search.png")
    if "scope" in chosen:
        draw_scope(OUT / "scope.png")


if __name__ == "__main__":
    main()
