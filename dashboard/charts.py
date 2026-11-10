"""Theme-aware Plotly charts for the dashboard.

Colors come from the reference data-viz palette and were checked with its
validator in both modes (lightness band, chroma floor, colorblind separation,
normal-vision separation, contrast). Every check passes:

* Output A / Output B: categorical slots 1-2 (blue, orange). The smallest
  colorblind-simulated difference is 24.7 in light mode and 26.8 in dark (8 is
  the target).
* Improvement / regression: the diverging blue / red pair around a neutral zero.

Mark specs follow the same guidance: bars at most about 24px thick with 4px
rounded data ends, hairline solid gridlines, text in text colors (never the
series color), a legend for two or more series, and values at bar tips.
Every chart has a table view next to it in the UI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Import pandas fully before Plotly can look for it. Plotly finds pandas through
# sys.modules without taking the import lock, and newer Streamlit versions import
# pandas lazily in another thread. If a chart is built during that import, Plotly
# sees a half-initialised module and fails with "partially initialized module
# 'pandas' has no attribute 'Series'". A regular import waits for it to finish.
import pandas  # noqa: F401
import plotly.graph_objects as go


@dataclass(frozen=True)
class Theme:
    text_primary: str
    text_secondary: str
    grid: str
    series_a: str
    series_b: str
    positive: str
    negative: str
    neutral: str
    hover_bg: str


LIGHT = Theme(
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e6e5e0",
    series_a="#2a78d6",
    series_b="#eb6834",
    positive="#2a78d6",
    negative="#e34948",
    neutral="#a3a29b",
    hover_bg="#fcfcfb",
)
DARK = Theme(
    text_primary="#f0efec",
    text_secondary="#c3c2b7",
    grid="#2f2f2d",
    series_a="#3987e5",
    series_b="#d95926",
    positive="#3987e5",
    negative="#e66767",
    neutral="#6c6b66",
    hover_bg="#1a1a19",
)

_ROW_PX = 34  # with bargap 0.35 this keeps single bars near 22px thick
_FONT = "Source Sans Pro, Source Sans 3, system-ui, sans-serif"
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def theme_for(mode: str | None) -> Theme:
    return DARK if mode == "dark" else LIGHT


def _layout(
    fig: go.Figure,
    theme: Theme,
    *,
    height: int,
    x_title: str = "",
    legend: bool,
    labels: Sequence[str] = (),
) -> None:
    axis = {
        "gridcolor": theme.grid,
        "gridwidth": 1,
        "zeroline": False,
        "showline": False,
        "tickfont": {"color": theme.text_secondary},
        "title": {"font": {"color": theme.text_secondary}},
    }
    fig.update_layout(
        height=height,
        # Reserve room for the longest category label: Plotly's automargin underestimates
        # some fonts and clips the first characters otherwise.
        margin={
            "l": 16 + 8 * max((len(str(x)) for x in labels), default=0),
            "r": 48,
            "t": 36 if legend else 8,
            "b": 40,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": _FONT, "color": theme.text_secondary, "size": 13},
        barcornerradius=4,
        bargap=0.35,
        showlegend=legend,
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.02,
            "yanchor": "bottom",
            "font": {"color": theme.text_primary},
            "traceorder": "normal",
            # Fixed entry widths: measured widths can clip the last letter when the
            # web font loads after Plotly measures the text.
            "entrywidth": 104,
            "entrywidthmode": "pixels",
        },
        hoverlabel={
            "bgcolor": theme.hover_bg,
            "bordercolor": theme.grid,
            "font": {"color": theme.text_primary, "family": _FONT},
        },
    )
    fig.update_xaxes(**axis, title_text=x_title)
    # The tick suffix keeps category labels clear of the plot edge (and never clipped).
    fig.update_yaxes(**{**axis, "showgrid": False}, automargin=False, ticksuffix="  ")


def _categorical_y(fig: go.Figure) -> None:
    """Keep category order even for ids like "1", "2" (Plotly would make them numbers)."""
    fig.update_yaxes(type="category")


def criterion_comparison(
    criteria: Sequence[str],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    scale: tuple[float, float] | None,
    theme: Theme,
) -> go.Figure:
    """Grouped horizontal bars: Output A vs Output B on each criterion.

    ``scale`` is the shared score range. Pass ``None`` when criteria use
    different scales; the values must then already be normalised to 0-1.
    """
    names = list(reversed(criteria))  # first criterion at the top
    a, b = list(reversed(scores_a)), list(reversed(scores_b))
    fmt = "%{x:.2f}" if scale is None else "%{x:g}"
    fig = go.Figure()
    # Horizontal groups stack traces bottom-up, so add B first to put A on top of each
    # pair; the legend is reversed to read "Output A, Output B" in the same order.
    for label, values, color in (("Output B", b, theme.series_b), ("Output A", a, theme.series_a)):
        fig.add_bar(
            name=label,
            y=names,
            x=values,
            orientation="h",
            marker={"color": color},
            text=[f"{v:.2f}" if scale is None else f"{v:g}" for v in values],
            textposition="outside",
            textfont={"color": theme.text_primary},
            cliponaxis=False,
            hovertemplate=f"{label} · %{{y}}: {fmt}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargroupgap=0.08)
    _layout(
        fig,
        theme,
        height=max(180, int(len(names) * _ROW_PX * 1.7) + 80),
        x_title="Score (share of scale, 0-1)" if scale is None else "Score",
        legend=True,
        labels=names,
    )
    _categorical_y(fig)
    fig.update_layout(legend_traceorder="reversed")
    low, high = scale if scale is not None else (0.0, 1.0)
    fig.update_xaxes(range=[low, high + (high - low) * 0.08])
    return fig


def ranked_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    theme: Theme,
    x_title: str,
    value_range: tuple[float, float] = (0.0, 1.0),
    hover_suffix: Sequence[str] | None = None,
) -> go.Figure:
    """Single-series horizontal bars, best at the top (leaderboards, model stats)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    names = [labels[i] for i in order]
    vals = [values[i] for i in order]
    extra = [hover_suffix[i] for i in order] if hover_suffix else [""] * len(order)
    fig = go.Figure(
        go.Bar(
            y=names,
            x=vals,
            orientation="h",
            marker={"color": theme.series_a},
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont={"color": theme.text_primary},
            cliponaxis=False,
            customdata=extra,
            hovertemplate="%{y}: %{x:.3f}%{customdata}<extra></extra>",
        )
    )
    _layout(
        fig,
        theme,
        height=max(140, len(names) * _ROW_PX + 70),
        x_title=x_title,
        legend=False,
        labels=names,
    )
    _categorical_y(fig)
    low, high = value_range
    fig.update_xaxes(range=[low, high + (high - low) * 0.1])
    return fig


def delta_bars(
    row_ids: Sequence[str], deltas: Sequence[float], *, theme: Theme, limit: int = 30
) -> go.Figure:
    """Per-row change between two runs: improvements and regressions around zero.

    Only the ``limit`` largest absolute changes are drawn (the table holds all).
    """
    pairs = sorted(zip(row_ids, deltas, strict=True), key=lambda p: abs(p[1]), reverse=True)[:limit]
    pairs.sort(key=lambda p: p[1])
    fig = go.Figure()
    groups = (
        ("Improved", [p for p in pairs if p[1] > 0], theme.positive),
        ("Unchanged", [p for p in pairs if p[1] == 0], theme.neutral),
        ("Regressed", [p for p in pairs if p[1] < 0], theme.negative),
    )
    for label, items, color in groups:
        if not items:
            continue
        if label == "Unchanged":  # a zero-length bar is invisible: mark it with a dot
            fig.add_scatter(
                name=label,
                y=[p[0] for p in items],
                x=[0.0] * len(items),
                mode="markers",
                marker={"color": color, "size": 9},
                hovertemplate=f"{label} · %{{y}}<extra></extra>",
            )
            continue
        fig.add_bar(
            name=label,
            y=[p[0] for p in items],
            x=[p[1] for p in items],
            orientation="h",
            marker={"color": color},
            hovertemplate=f"{label} · %{{y}}: %{{x:+.3f}}<extra></extra>",
        )
    fig.update_layout(barmode="relative")
    _layout(
        fig,
        theme,
        height=max(160, len(pairs) * _ROW_PX + 80),
        x_title="Change in score (candidate - baseline)",
        legend=True,
        labels=[p[0] for p in pairs],
    )
    _categorical_y(fig)
    fig.update_yaxes(categoryorder="array", categoryarray=[p[0] for p in pairs])
    fig.add_vline(x=0, line_width=1, line_color=theme.text_secondary)
    return fig


def score_histogram(values: Sequence[float], *, theme: Theme, x_title: str) -> go.Figure:
    """Distribution of normalised scores (0-1) in ten equal bins."""
    fig = go.Figure(
        go.Histogram(
            x=list(values),
            xbins={"start": 0, "end": 1.0001, "size": 0.1},
            marker={"color": theme.series_a},
            hovertemplate="%{x}: %{y} row(s)<extra></extra>",
        )
    )
    _layout(fig, theme, height=240, x_title=x_title, legend=False)
    fig.update_layout(bargap=0.08)
    # Whole-number row counts only; the x origin label is dropped so the two 0s don't collide.
    counts = [0] * 10
    for v in values:
        counts[min(9, max(0, int(v * 10)))] += 1
    step = max(1, math.ceil(max(counts, default=1) / 4))
    fig.update_xaxes(range=[0, 1], tickvals=[0.2, 0.4, 0.6, 0.8, 1.0])
    fig.update_yaxes(
        title_text="Rows", showgrid=True, dtick=step, tickformat="d", ticksuffix="", automargin=True
    )
    return fig
