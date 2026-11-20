#!/usr/bin/env python3
"""
ChartTrap generator: synthetic "tricky" business charts + verifiable QA pairs.

Pipeline per chart
  1. Sample a data table and axis configuration from a seeded RNG (the ONLY source of truth).
  2. Enforce a minimum *pixel* separation for every answer (so the answer is visually resolvable).
  3. Render with Matplotlib at a fixed layout (1000x600 px, fixed axes rectangle).
  4. Independently re-measure the rendered PNG (pixel scan) and check it agrees with the data.
  5. Build questions whose reference answers are computed from the data table, then re-derive
     every answer from the *pixel measurements* and drop any question that disagrees.

Usage
  python generate.py --out data --n 60 --seed 7
  python generate.py --revalidate data          # re-run the pixel audit on an existing dataset
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from PIL import Image  # noqa: E402

# --------------------------------------------------------------------------- layout constants
DPI = 100
FIG_W, FIG_H = 10.0, 6.0                       # -> 1000 x 600 px
AX_RECT = (0.09, 0.23, 0.80, 0.64)             # left, bottom, width, height (figure fractions)
AX_H_PX = AX_RECT[3] * FIG_H * DPI             # plot-area height in pixels (432)
AX_W_PX = AX_RECT[2] * FIG_W * DPI
BAR_COLORS = ["#1f77b4", "#ff7f0e"]
LINE_COLOR = "#d62728"
BAR_PX_TOL = 2.0                               # max |pixel-measured - true| for bars
LINE_PX_TOL = 2.5                              # same for line markers

# difficulty tiers: n categories, min pixel gap between the answer and its closest rival,
# accepted read-off tolerance (px), and probabilities of each "trap" feature.
TIERS = {
    "easy":   dict(n=(5, 7),   gap=30, tol=16, p_log=0.00, p_trunc=0.00, p_dual=0.20, p_clutter=0.00, p_grid=1.0,  p_hint=1.0),
    "medium": dict(n=(7, 10),  gap=15, tol=12, p_log=0.15, p_trunc=0.35, p_dual=0.30, p_clutter=0.35, p_grid=0.7,  p_hint=0.7),
    "hard":   dict(n=(9, 12),  gap=8,  tol=9,  p_log=0.30, p_trunc=0.50, p_dual=0.40, p_clutter=0.70, p_grid=0.35, p_hint=0.3),
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
UNITS_BIZ = ["Cloud", "Retail", "Payments", "Ads", "Devices", "Logistics", "Media", "Health",
             "Energy", "Security", "Gaming", "Travel"]
REGIONS = ["North America", "EMEA", "APAC", "LATAM", "India", "China", "Japan", "Germany",
           "UK", "Brazil", "Canada", "Australia"]
FUNDS = ["Atlas Growth", "Borealis Income", "Cedar Value", "Delta Tech", "Echo Global", "Fjord Bond",
         "Granite Core", "Harbor Small-Cap", "Ivy Balanced", "Juniper ESG", "Keystone REIT", "Lumen Macro"]

THEMES = [
    dict(name="revenue", title="Quarterly Revenue by Business Unit", measure="Revenue", ylabel="Revenue ($M)",
         unit="$M", cats=UNITS_BIZ, ordered=False, series=("FY2024", "FY2025"), lo=40, hi=420,
         right=("Operating margin", "Operating margin (%)", "%", 4, 36)),
    dict(name="traffic", title="Monthly Website Visits", measure="Visits", ylabel="Visits (thousands)",
         unit="thousand visits", cats=MONTHS, ordered=True, series=("2024", "2025"), lo=120, hi=900,
         right=("Conversion rate", "Conversion rate (%)", "%", 1.2, 6.5)),
    dict(name="sales", title="Regional Unit Sales", measure="Units sold", ylabel="Units sold (thousands)",
         unit="thousand units", cats=REGIONS, ordered=False, series=("H1", "H2"), lo=15, hi=240,
         right=("Return rate", "Return rate (%)", "%", 1, 9)),
    dict(name="aum", title="Assets Under Management by Fund", measure="AUM", ylabel="AUM ($M)",
         unit="$M", cats=FUNDS, ordered=False, series=("Dec 2024", "Dec 2025"), lo=25, hi=700,
         right=("1-yr return", "1-year return (%)", "%", 2, 24)),
]


# --------------------------------------------------------------------------- data model
@dataclass
class Axis:
    ymin: float
    ymax: float
    scale: str = "linear"  # "linear" | "log"

    def frac(self, v: float) -> float:
        """Data value -> fractional height inside the plot area (0 = bottom, 1 = top)."""
        if self.scale == "log":
            lo, hi = math.log10(self.ymin), math.log10(self.ymax)
            return (math.log10(v) - lo) / (hi - lo)
        return (v - self.ymin) / (self.ymax - self.ymin)

    def value(self, f: float) -> float:
        """Fractional height -> data value (inverse of frac)."""
        if self.scale == "log":
            lo, hi = math.log10(self.ymin), math.log10(self.ymax)
            return 10 ** (lo + f * (hi - lo))
        return self.ymin + f * (self.ymax - self.ymin)


@dataclass
class ChartSpec:
    chart_id: str
    tier: str
    theme: str
    title: str
    kind: str                       # "bar" | "grouped" | "dual"
    categories: list
    series: dict                    # name -> list[float]
    roles: dict                     # name -> "bar" | "line"
    axis_of: dict                   # name -> "left" | "right"
    units: dict                     # name -> unit string
    left_axis: Axis
    right_axis: Axis | None
    left_label: str
    right_label: str | None
    threshold: float | None
    threshold_series: str | None
    gridlines: bool
    minor_grid: bool
    clutter: bool
    axis_color_hint: bool
    focus: str

    def axis_for(self, name: str) -> Axis:
        return self.left_axis if self.axis_of[name] == "left" else self.right_axis

    def bar_names(self) -> list:
        return [n for n, r in self.roles.items() if r == "bar"]

    @staticmethod
    def from_dict(d: dict) -> "ChartSpec":
        d = dict(d)
        d["left_axis"] = Axis(**d["left_axis"])
        d["right_axis"] = Axis(**d["right_axis"]) if d["right_axis"] else None
        return ChartSpec(**d)


# --------------------------------------------------------------------------- small helpers
def nice_ceil(x: float) -> float:
    if x <= 0:
        return 1.0
    base = 10 ** math.floor(math.log10(x))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * base >= x - 1e-12:
            return m * base
    return 10 * base


def round_sig(v: float, n: int = 3) -> float:
    return float(f"{v:.{n - 1}e}")


def round_series(vals, axis: Axis):
    if axis.scale == "log":
        return [round_sig(v, 3) for v in vals]
    d = 1 if (axis.ymax - axis.ymin) >= 20 else 2
    return [round(v, d) for v in vals]


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])


def color_of(spec: ChartSpec, name: str) -> str:
    if spec.roles[name] == "line":
        return LINE_COLOR
    return BAR_COLORS[spec.bar_names().index(name)]


def gap_px(axis: Axis, a: float, b: float) -> float:
    return abs(axis.frac(a) - axis.frac(b)) * AX_H_PX


# --------------------------------------------------------------------------- chart construction
def pick_categories(rng, theme, n):
    pool = theme["cats"]
    if theme["ordered"]:
        return pool[:n]
    return rng.sample(pool, n)


def make_left_axis(values, use_log: bool, trunc: bool):
    lo, hi = min(values), max(values)
    if use_log:
        ymin = 10 ** math.floor(math.log10(lo))
        ymax = 10 ** math.ceil(math.log10(hi * 1.05))
        if math.log10(ymax / ymin) < 2:
            ymax = ymin * 100
        return Axis(ymin, ymax, "log")
    if trunc:
        r = max(hi - lo, 1e-9)
        step = nice_ceil(r / 5)
        ymin = math.floor((lo - 0.4 * r) / step) * step
        ymax = math.ceil((hi + 0.15 * r) / step) * step
        if ymin > 0:
            return Axis(ymin, ymax, "linear")
    return Axis(0.0, nice_ceil(hi * 1.12), "linear")


def enforce_top_gap(rng, vals, axis: Axis, min_gap: float):
    """Keep the maximum fixed; force the runner-up to sit a controlled number of px below it."""
    n = len(vals)
    f = [axis.frac(v) for v in vals]
    i = max(range(n), key=lambda k: f[k])
    j = rng.choice([k for k in range(n) if k != i])
    fj = f[i] - min_gap * rng.uniform(1.05, 1.9) / AX_H_PX
    f[j] = fj
    for k in range(n):
        if k not in (i, j):
            if f[k] > fj - 0.01:
                f[k] = fj - rng.uniform(0.01, 0.12)
            f[k] = max(f[k], 0.04)
    return [axis.value(x) for x in f]


def enforce_largest_diff(rng, A, B, axis: Axis, min_gap: float):
    """Make exactly one category have the largest (B - A) by a controlled pixel margin."""
    n, span = len(A), axis.ymax - axis.ymin
    m = rng.randrange(n)
    d2 = max(B[k] - A[k] for k in range(n) if k != m)
    target = max(d2, 0.0) + min_gap * rng.uniform(1.05, 1.8) / AX_H_PX * span
    a = A[m]
    if a + target > axis.ymin + 0.95 * span:
        a = axis.ymin + 0.95 * span - target
    if a <= axis.ymin + 0.03 * span:
        return False
    A[m], B[m] = a, a + target
    return True


def choose_threshold(rng, axis: Axis, vals, min_gap: float):
    """Pick a horizontal line value leaving >= min_gap px on each side of every bar."""
    fs = sorted(axis.frac(v) for v in vals)
    options = []
    for k in range(1, len(fs)):
        if (fs[k] - fs[k - 1]) * AX_H_PX >= 2 * min_gap:
            options.append((fs[k] + fs[k - 1]) / 2)
    if not options:
        return None
    return axis.value(rng.choice(options))


def build_chart(rng: random.Random, tier: str, idx: int) -> ChartSpec | None:
    cfg = TIERS[tier]
    th = rng.choice(THEMES)
    kind = "dual" if rng.random() < cfg["p_dual"] else rng.choice(["bar", "grouped"])
    n = rng.randint(*cfg["n"])
    cats = pick_categories(rng, th, n)
    use_log = rng.random() < cfg["p_log"]
    trunc = (not use_log) and rng.random() < cfg["p_trunc"]
    bar_names = list(th["series"]) if kind == "grouped" else [th["measure"]]

    # ---- sample raw bar values
    def sample_first():
        if use_log:
            return [th["lo"] * 10 ** rng.uniform(0, 2.6) for _ in range(n)]
        if trunc:
            base = rng.uniform(th["lo"] + 0.35 * (th["hi"] - th["lo"]), th["lo"] + 0.6 * (th["hi"] - th["lo"]))
            return [base * rng.uniform(1.0, 1.35) for _ in range(n)]
        return [rng.uniform(th["lo"], th["hi"]) for _ in range(n)]

    raw = {bar_names[0]: sample_first()}
    if kind == "grouped":
        fac = (lambda: 10 ** rng.uniform(-0.3, 0.3)) if use_log else (lambda: rng.uniform(0.75, 1.3))
        raw[bar_names[1]] = [v * fac() for v in raw[bar_names[0]]]

    axis = make_left_axis([v for s in raw.values() for v in s], use_log, trunc)
    trunc = axis.scale == "linear" and axis.ymin > 0
    series, roles, axis_of, units = {}, {}, {}, {}
    for nm in bar_names:
        roles[nm], axis_of[nm], units[nm] = "bar", "left", th["unit"]

    right_axis, right_label = None, None
    if kind == "dual":
        rname, right_label, runit, rlo, rhi = th["right"]
        line_vals = [rng.uniform(rlo, rhi) for _ in range(n)]
        right_axis = Axis(0.0, nice_ceil(max(line_vals) * 1.15), "linear")
        raw[rname] = line_vals
        roles[rname], axis_of[rname], units[rname] = "line", "right", runit

    def ax_of(nm):
        return axis if axis_of.get(nm, "left") == "left" else right_axis

    # ---- choose what challenge this chart is built around
    focus = "argmax:" + rng.choice(list(raw))
    if kind == "grouped" and not use_log and rng.random() < 0.4:
        if enforce_largest_diff(rng, raw[bar_names[0]], raw[bar_names[1]], axis, cfg["gap"]):
            focus = "largest_gap"
    if focus.startswith("argmax:"):
        nm = focus.split(":", 1)[1]
        raw[nm] = enforce_top_gap(rng, raw[nm], ax_of(nm), cfg["gap"])

    for nm, vals in raw.items():
        series[nm] = round_series(vals, ax_of(nm))

    # ---- optional threshold line (for "how many bars above the dashed line")
    thr, thr_series = None, None
    if rng.random() < 0.65:
        thr_series = rng.choice(bar_names)
        thr = choose_threshold(rng, axis, series[thr_series], cfg["gap"])
        if thr is None:
            thr_series = None
        else:
            thr = round_series([thr], axis)[0]
            # rounding must not move the line into a bar's tolerance band
            if min(gap_px(axis, thr, v) for v in series[thr_series]) < cfg["gap"] * 0.9:
                thr, thr_series = None, None

    return ChartSpec(
        chart_id=f"chart_{idx:04d}", tier=tier, theme=th["name"], title=th["title"], kind=kind,
        categories=cats, series=series, roles=roles, axis_of=axis_of, units=units,
        left_axis=axis, right_axis=right_axis, left_label=th["ylabel"], right_label=right_label,
        threshold=thr, threshold_series=thr_series,
        gridlines=rng.random() < cfg["p_grid"], minor_grid=rng.random() < cfg["p_clutter"],
        clutter=rng.random() < cfg["p_clutter"], axis_color_hint=rng.random() < cfg["p_hint"], focus=focus,
    )


# --------------------------------------------------------------------------- rendering
def render_chart(spec: ChartSpec, path: Path) -> dict:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes(AX_RECT)
    n = len(spec.categories)
    x = np.arange(n)
    la = spec.left_axis
    ax.set_yscale(la.scale)

    bars = spec.bar_names()
    width = 0.62 if len(bars) == 1 else 0.38
    offsets = [0.0] if len(bars) == 1 else [-width / 2, width / 2]
    geom = {}
    unit_px = AX_W_PX / (n + 0.2)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(la.ymin, la.ymax)

    if spec.clutter:
        ax.axvspan(n - 2.5, n - 0.5, color="#efefef", zorder=0)
        ax.text(n - 1.5, 0.965, "Forecast", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9, color="#666666", zorder=1)
        fig.text(0.01, 0.008, "Source: synthetic data | unaudited | figures rounded", fontsize=8, color="#777777")

    for nm, off in zip(bars, offsets):
        vals = spec.series[nm]
        ax.bar(x + off, [v - la.ymin for v in vals], width, bottom=la.ymin, color=color_of(spec, nm),
               label=nm, zorder=3, linewidth=0)
        geom[nm] = {"x_px": [float(ax.transData.transform((xi + off, la.ymin))[0]) for xi in x],
                    "w_px": width * unit_px}

    ax.set_axisbelow(True)
    if spec.gridlines:
        ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.8)
    if spec.minor_grid:
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.grid(axis="y", which="minor", color="#e6e6e6", linewidth=0.5)
    if la.scale == "log":
        ax.yaxis.set_minor_locator(LogLocator(subs=range(2, 10), numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))

    ax.set_xticks(x)
    rot = 45 if (n > 8 or max(len(c) for c in spec.categories) > 9) else 0
    ax.set_xticklabels(spec.categories, rotation=rot, ha="right" if rot else "center", fontsize=10)
    hint_l = BAR_COLORS[0] if (spec.axis_color_hint and spec.kind == "dual") else "black"
    ax.set_ylabel(spec.left_label, color=hint_l, fontsize=11)

    if spec.threshold is not None:
        ax.axhline(spec.threshold, color="black", linestyle="--", linewidth=1.6, label="Target", zorder=4)

    handles, labels = ax.get_legend_handles_labels()
    if spec.right_axis is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(spec.right_axis.ymin, spec.right_axis.ymax)
        ax2.set_xlim(-0.6, n - 0.4)
        lname = [k for k, r in spec.roles.items() if r == "line"][0]
        ax2.plot(x, spec.series[lname], color=LINE_COLOR, marker="o", markersize=8, linewidth=2.2,
                 label=lname, zorder=5)
        ax2.set_ylabel(spec.right_label, color=LINE_COLOR if spec.axis_color_hint else "black", fontsize=11)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
        geom[lname] = {"x_px": [float(ax2.transData.transform((xi, spec.right_axis.ymin))[0]) for xi in x],
                       "w_px": 0.0}
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=10)
    fig.suptitle(spec.title, y=0.985, va="top", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return geom


# --------------------------------------------------------------------------- independent pixel audit
def measure_pixels(path: Path, spec: ChartSpec, geom: dict) -> dict:
    """Read the PNG back and recover each mark's fractional height WITHOUT using the data table."""
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    H = img.shape[0]
    top_row = int(round(H * (1 - (AX_RECT[1] + AX_RECT[3]))))
    bot_row = int(round(H * (1 - AX_RECT[1])))
    ax_bottom = AX_RECT[1] * H
    out = {}
    for name, role in spec.roles.items():
        col = hex_rgb(color_of(spec, name))
        fs = []
        for xpx in geom[name]["x_px"]:
            if role == "bar":
                rows = []
                for dx in (-0.35, -0.17, 0, 0.17, 0.35):
                    c = int(round(xpx + dx * geom[name]["w_px"]))
                    hit = np.flatnonzero(np.abs(img[top_row:bot_row + 1, c, :] - col).max(axis=1) <= 10)
                    if hit.size:
                        rows.append(top_row + int(hit[0]))   # overlays can hide a top edge, never fake one
                fs.append(((H - min(rows)) - ax_bottom) / AX_H_PX if rows else None)
            else:
                ys = []
                for dx in (-1, 0, 1):
                    c = int(round(xpx)) + dx
                    hit = np.flatnonzero(np.abs(img[top_row:bot_row + 1, c, :] - col).max(axis=1) <= 14)
                    ys.extend((top_row + hit).tolist())
                fs.append(((H - (float(np.mean(ys)) + 0.5)) - ax_bottom) / AX_H_PX if ys else None)
        out[name] = fs
    return out


def check_measured(spec: ChartSpec, measured: dict):
    errors, max_err = [], 0.0
    for name, fs in measured.items():
        tol = LINE_PX_TOL if spec.roles[name] == "line" else BAR_PX_TOL
        axis = spec.axis_for(name)
        for i, (f, v) in enumerate(zip(fs, spec.series[name])):
            if f is None:
                errors.append(f"{name}[{i}] not found in image")
                continue
            err = abs(f - axis.frac(v)) * AX_H_PX
            max_err = max(max_err, err)
            if err > tol:
                errors.append(f"{name}[{i}] pixel error {err:.1f}px")
    return errors, max_err


def pixel_validate_question(q: dict, spec: ChartSpec, measured: dict) -> bool:
    """Re-derive the reference answer from pixel measurements only."""
    p, t = q["params"], q["qtype"]

    def val(name, i):
        return spec.axis_for(name).value(min(max(measured[name][i], 0.0), 1.0))

    cats = spec.categories
    if t == "read_value":
        return q["accept_lo"] <= val(p["series"], p["index"]) <= q["accept_hi"]
    if t == "argmax":
        fs = measured[p["series"]]
        return cats[max(range(len(fs)), key=lambda k: fs[k])] == q["answer"]
    if t == "count_above":
        thr_f = spec.left_axis.frac(spec.threshold)
        return sum(1 for f in measured[p["series"]] if f > thr_f) == q["answer"]
    if t == "largest_gap":
        a, b = p["series_a"], p["series_b"]
        d = [val(b, k) - val(a, k) for k in range(len(cats))]
        return cats[max(range(len(d)), key=lambda k: d[k])] == q["answer"]
    if t == "ratio":
        s = p["series"]
        r = val(s, p["index_a"]) / val(s, p["index_b"])
        return q["accept_lo"] <= r <= q["accept_hi"]
    return False


# --------------------------------------------------------------------------- question construction
def features(spec: ChartSpec) -> dict:
    la = spec.left_axis
    return dict(kind=spec.kind, scale=la.scale, truncated_axis=(la.scale == "linear" and la.ymin > 0),
                dual_axis=spec.right_axis is not None, clutter=spec.clutter, gridlines=spec.gridlines,
                axis_color_hint=spec.axis_color_hint, n_categories=len(spec.categories),
                has_threshold=spec.threshold is not None)


def ref(spec: ChartSpec, name: str) -> str:
    return f"{name} line" if spec.roles[name] == "line" else f"{name} bar"


def base_q(spec, n, qtype, text, atype, answer, **kw):
    q = dict(id=f"{spec.chart_id}_q{n}", chart_id=spec.chart_id, image=f"images/{spec.chart_id}.png",
             qtype=qtype, question=text, answer_type=atype, answer=answer, tier=spec.tier,
             features=features(spec))
    q.update(kw)
    return q


def q_read_value(spec, rng, cfg, n):
    name = rng.choice(list(spec.series))
    axis, i = spec.axis_for(name), rng.randrange(len(spec.categories))
    v = spec.series[name][i]
    f, d = axis.frac(v), cfg["tol"] / AX_H_PX
    lo, hi = axis.value(max(f - d, 0.0)), axis.value(min(f + d, 1.0))
    dis = {"neighbors": {c: spec.series[name][k] for k, c in enumerate(spec.categories) if k != i}}
    if axis.scale == "log":
        dis["linear_reading"] = axis.ymin + f * (axis.ymax - axis.ymin)
    if spec.right_axis is not None:
        other = spec.right_axis if axis is spec.left_axis else spec.left_axis
        dis["other_axis_reading"] = other.value(f)
    others = [s for s in spec.bar_names() if s != name]
    if spec.kind == "grouped" and others and spec.roles[name] == "bar":
        dis["other_series_same_category"] = spec.series[others[0]][i]
    cat = spec.categories[i]
    text = f"What is the value of the {ref(spec, name)} for {cat}? Give your best numerical estimate in {spec.units[name]}."
    return base_q(spec, n, "read_value", text, "numeric", v, accept_lo=lo, accept_hi=hi, distractors=dis,
                  params={"series": name, "index": i})


def q_argmax(spec, rng, cfg, n):
    name = spec.focus.split(":", 1)[1] if spec.focus.startswith("argmax:") else rng.choice(list(spec.series))
    vals, axis = spec.series[name], spec.axis_for(name)
    order = sorted(range(len(vals)), key=lambda k: -axis.frac(vals[k]))
    g = gap_px(axis, vals[order[0]], vals[order[1]])
    if g < cfg["gap"] * 0.8:
        return None
    what = "highest point on the" if spec.roles[name] == "line" else "tallest"
    text = f"Which category has the {what} {ref(spec, name)}? Answer with the category label exactly as written on the x-axis."
    return base_q(spec, n, "argmax", text, "category", spec.categories[order[0]], choices=list(spec.categories),
                  distractors={"runner_up": spec.categories[order[1]]},
                  params={"series": name, "gap_px": round(g, 1)})


def q_count_above(spec, rng, cfg, n):
    if spec.threshold is None:
        return None
    name = spec.threshold_series
    cnt = sum(1 for v in spec.series[name] if spec.left_axis.frac(v) > spec.left_axis.frac(spec.threshold))
    text = f"How many of the {name} bars rise above the dashed 'Target' line? Answer with a whole number."
    return base_q(spec, n, "count_above", text, "integer", cnt, params={"series": name})


def q_largest_gap(spec, rng, cfg, n):
    if spec.focus != "largest_gap":
        return None
    a, b = spec.bar_names()
    d = [y - x for x, y in zip(spec.series[a], spec.series[b])]
    order = sorted(range(len(d)), key=lambda k: -d[k])
    span = spec.left_axis.ymax - spec.left_axis.ymin
    g = (d[order[0]] - d[order[1]]) / span * AX_H_PX
    if g < cfg["gap"] * 0.8:
        return None
    tallest_b = max(range(len(d)), key=lambda k: spec.series[b][k])
    text = (f"In which category does the {b} bar exceed the {a} bar by the greatest amount? "
            f"Answer with the category label exactly as written on the x-axis.")
    return base_q(spec, n, "largest_gap", text, "category", spec.categories[order[0]], choices=list(spec.categories),
                  distractors={"runner_up": spec.categories[order[1]], "tallest_second_series": spec.categories[tallest_b]},
                  params={"series_a": a, "series_b": b, "gap_px": round(g, 1)})


def q_ratio(spec, rng, cfg, n):
    if spec.kind != "bar" or spec.left_axis.scale != "linear":
        return None
    name = spec.bar_names()[0]
    vals, axis = spec.series[name], spec.left_axis
    trunc = axis.ymin > 0
    if not trunc and rng.random() > 0.3:           # keep a few non-truncated charts as controls
        return None
    pairs = [(a, b) for a in range(len(vals)) for b in range(len(vals))
             if vals[a] / vals[b] >= 1.2 and gap_px(axis, vals[a], vals[b]) >= 40]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    r = vals[a] / vals[b]
    vis = axis.frac(vals[a]) / axis.frac(vals[b])
    if trunc and abs(vis / r - 1) < 0.2:
        return None
    ca, cb = spec.categories[a], spec.categories[b]
    text = (f"Approximately how many times larger is the {name} for {ca} than for {cb}? "
            f"Answer as a decimal ratio, for example 1.5.")
    return base_q(spec, n, "ratio", text, "numeric", r, accept_lo=r * 0.9, accept_hi=r * 1.1,
                  distractors={"visual_ratio": vis, "inverse_ratio": 1 / r},
                  params={"series": name, "index_a": a, "index_b": b})


def make_questions(spec, rng, cfg):
    qs = []
    for builder in (q_argmax, q_largest_gap, q_count_above, q_ratio, q_read_value, q_read_value):
        q = builder(spec, rng, cfg, len(qs) + 1)
        if q:
            qs.append(q)
    return qs[:4]


# --------------------------------------------------------------------------- driver
def generate(out: Path, n_charts: int, seed: int, tiers: list):
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats = Counter()
    chart_rows, q_rows = [], []
    for idx in range(1, n_charts + 1):
        tier = tiers[(idx - 1) % len(tiers)]
        cfg = TIERS[tier]
        for _attempt in range(30):
            spec = build_chart(rng, tier, idx)
            if spec is None:
                continue
            png = out / "images" / f"{spec.chart_id}.png"
            geom = render_chart(spec, png)
            measured = measure_pixels(png, spec, geom)
            errs, max_err = check_measured(spec, measured)
            if errs:
                stats["chart_rejected_pixel_audit"] += 1
                continue
            qs = make_questions(spec, rng, cfg)
            good = []
            for q in qs:
                if pixel_validate_question(q, spec, measured):
                    q["verification"] = {"pixel_audit": "passed", "max_pixel_error": round(max_err, 2)}
                    good.append(q)
                else:
                    stats["question_rejected_pixel_audit"] += 1
            if len(good) >= 2:
                chart_rows.append({"spec": asdict(spec), "geometry": geom, "measured": measured})
                q_rows.extend(good)
                stats["charts_ok"] += 1
                stats["questions_ok"] += len(good)
                break
        else:
            stats["chart_gave_up"] += 1

    with open(out / "dataset.jsonl", "w") as f:
        for q in q_rows:
            f.write(json.dumps(q) + "\n")
    with open(out / "charts.jsonl", "w") as f:
        for c in chart_rows:
            f.write(json.dumps(c) + "\n")
    manifest = dict(seed=seed, tiers=tiers, requested_charts=n_charts, **stats,
                    qtypes=dict(Counter(q["qtype"] for q in q_rows)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def revalidate(out: Path):
    """Re-measure every stored PNG and re-check every question against pixels only."""
    charts = {}
    for line in open(out / "charts.jsonl"):
        row = json.loads(line)
        spec = ChartSpec.from_dict(row["spec"])
        charts[spec.chart_id] = (spec, row["geometry"])
    bad, total = 0, 0
    for line in open(out / "dataset.jsonl"):
        q = json.loads(line)
        spec, geom = charts[q["chart_id"]]
        measured = measure_pixels(out / q["image"], spec, geom)
        errs, _ = check_measured(spec, measured)
        ok = not errs and pixel_validate_question(q, spec, measured)
        total += 1
        if not ok:
            bad += 1
            print("FAILED:", q["id"], errs[:2])
    print(f"revalidated {total} questions, {bad} failures")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--n", type=int, default=60, help="number of charts (about 3 questions each)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tiers", default="easy,medium,hard")
    ap.add_argument("--revalidate", type=Path, help="audit an existing dataset directory and exit")
    a = ap.parse_args()
    if a.revalidate:
        revalidate(a.revalidate)
    else:
        generate(a.out, a.n, a.seed, a.tiers.split(","))