"""
Plotting utilities for qPCR analysis results.

Overview
--------
This module provides reusable plotting functions for visualizing normalized
qPCR expression data, time-course experiments, single-gene comparisons, and
inter-plate calibration offsets.

The plotting functions are designed to work with tidy pandas DataFrames
produced by the upstream qPCR pipeline, especially after:
- preprocessing
- normalization
- statistical testing
- optional inter-plate calibration

Main goals
----------
- create publication-ready exploratory plots with minimal repeated code
- display both raw biological replicate values and summarized group means
- support flexible grouping via user-specified dataframe columns
- keep plotting logic independent of hard-coded biological factor names
- support optional significance annotation from pairwise comparison tables
- provide consistent figure-saving utilities across formats

What this module plots
----------------------
1. Expression grid plots
   - genes as subplot rows
   - conditions as subplot columns
   - one categorical x-axis variable per panel

2. Time-course grid plots
   - genes as subplot rows
   - conditions as subplot columns
   - timepoints on the x-axis
   - separate lines for user-defined groups

3. Single-gene detail plots
   - one target gene at a time
   - raw points plus mean/error overlays
   - optional significance annotations

4. Calibrator offset plots
   - per-target inter-plate offset visualization
   - useful for quality control of plate calibration

Data expectations
-----------------
These functions expect tidy DataFrames where each row typically represents one
biological replicate (or one summarized observation), and columns define:
- target gene identity
- experimental grouping variables
- plotted values such as:
    - normalized_expression
    - log2_normalized_expression
    - fold_change
    - offset_ct

The plotting layer does not assume specific biological meanings for columns
like genotype, condition, or timepoint. Instead, the caller explicitly maps
dataframe columns to plotting roles such as:
- gene_col
- condition_col
- x_col
- hue_col
- time_col
- value_col

Design principles
-----------------
- plotting should be transparent and data-driven
- raw data points should remain visible whenever practical
- summary statistics are overlaid, not substituted for raw values
- ordering, colors, markers, and scales should be customizable
- shared helper functions should reduce repeated plotting logic
- significance annotation should be compatible with pairwise comparison tables
  without forcing one rigid biological naming scheme

Significance annotations
------------------------
Single-gene plots can optionally annotate pairwise statistical comparisons.
To keep plotting flexible, annotation keys are rebuilt from raw plotting factor
columns rather than relying entirely on one hard-coded prejoined label format.

For annotations to work correctly, the group labels in the supplied statistics
table must match the keys reconstructed from the plotting factors used in the
figure.

Typical usage
-------------
>>> from qpcr.plotting import plot_expression_grid, plot_single_gene
>>>
>>> fig, axes = plot_expression_grid(
...     analysis_df,
...     gene_col="target",
...     condition_col="group",
...     x_col="sample_id",
...     value_col="normalized_expression",
... )
>>>
>>> fig_gene, ax_gene = plot_single_gene(
...     analysis_df,
...     gene="LHCSR3",
...     x_col="sample_id",
...     hue_col="group",
...     value_col="log2_normalized_expression",
...     stats_df=pairwise_df,
...     annotate_stats=True,
...     annotation_factor_cols=["sample_id", "group"],
... )

Notes
-----
- This module focuses on matplotlib-based plotting.
- It does not perform statistical testing itself.
- Summary statistics drawn in plots are intended for visualization, not as a
  replacement for the formal statistics module.
- Many functions return both the figure and axes so the caller can further
  customize the plot before saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ============================================================================
# Generic helpers
# ============================================================================

def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    """
    Raise a clear error if required columns are missing.

    Parameters
    ----------
    df
        Dataframe to validate.
    required
        Column names that must be present.
    context
        Short description of where the check is happening.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def significance_stars(p) -> str:
    """
    Convert a p-value into a standard significance label.

    Parameters
    ----------
    p
        P-value or missing value.

    Returns
    -------
    str
        One of:
        - '***' for p < 0.001
        - '**'  for p < 0.01
        - '*'   for p < 0.05
        - 'ns'  otherwise
        - ''    for missing p-values
    """
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def has_real_values(df: pd.DataFrame, col: str) -> bool:
    """
    Return True if a column exists and contains at least one non-missing value.

    Parameters
    ----------
    df
        Input dataframe.
    col
        Column name to inspect.

    Returns
    -------
    bool
        True if the column exists and has at least one non-null value.
    """
    return col in df.columns and df[col].notna().any()


def _ordered_levels(df: pd.DataFrame, col: str, order: Sequence[str] | None = None) -> list:
    """
    Return ordered levels for a column.

    Parameters
    ----------
    df
        Input dataframe.
    col
        Column whose levels should be extracted.
    order
        Optional explicit order. If provided, this is returned as-is.

    Returns
    -------
    list
        Ordered unique non-missing levels.
    """
    if order is not None:
        return list(order)
    return [x for x in df[col].dropna().unique().tolist()]


def _prepare_plot_df(
    df: pd.DataFrame,
    *,
    required_cols: Sequence[str],
    context: str,
) -> pd.DataFrame:
    """
    Validate plotting inputs and drop rows missing required plotting values.

    Parameters
    ----------
    df
        Input dataframe.
    required_cols
        Columns that must exist and must be non-missing for a row to be plotted.
    context
        Description used in error messages.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe containing only plottable rows.

    Raises
    ------
    ValueError
        If required columns are missing or if no plottable rows remain.
    """
    _require_columns(df, required_cols, context)
    plot_df = df.copy().dropna(subset=list(required_cols))
    if plot_df.empty:
        raise ValueError(f"No plottable rows remain in {context} after dropping missing required values.")
    return plot_df


def _make_annotation_label(values: Sequence[object], sep: str = " | ") -> str:
    """
    Join multiple factor values into one annotation label.

    Parameters
    ----------
    values
        Sequence of values defining one plotted group.
    sep
        Separator used between values.

    Returns
    -------
    str
        Combined label such as 'WT | BL' or 'WT | BL | 0h'.
    """
    return sep.join("NA" if pd.isna(v) else str(v) for v in values)


def _build_annotation_key(
    row_or_mapping,
    factor_cols: Sequence[str],
    *,
    sep: str = " | ",
) -> str:
    """
    Build one annotation key from raw factor columns.

    Parameters
    ----------
    row_or_mapping
        A pandas row or mapping-like object containing the requested columns.
    factor_cols
        Columns whose values define the plotted group identity.
    sep
        Separator used between factor values.

    Returns
    -------
    str
        Combined annotation key.

    Example
    -------
    factor_cols = ['sample_id', 'group']
    -> 'CC-5415 | BL'
    """
    return _make_annotation_label([row_or_mapping[col] for col in factor_cols], sep=sep)


def _add_interaction_annotation(
    ax,
    *,
    interaction_df: pd.DataFrame | None,
    target: str,
    target_col: str = "target",
    p_col: str = "p_adj",
    term_col: str = "interaction_term",
    x: float = 0.02,
    y: float = 1.14,
):
    """
    Add interaction-effect annotation to a plot.

    Displays the interaction term (e.g. sample_id × group) together with
    its significance level.

    Parameters
    ----------
    ax
        Matplotlib axis.
    interaction_df
        Dataframe with interaction statistics per target.
    target
        Target gene currently plotted.
    target_col
        Column identifying targets.
    p_col
        Column containing p-values.
    term_col
        Column containing the interaction term string.
    x, y
        Position in axis coordinates.
    """
    if interaction_df is None or interaction_df.empty:
        return

    if target_col not in interaction_df.columns:
        return

    sub = interaction_df.loc[interaction_df[target_col] == target]
    if sub.empty:
        return

    p = sub[p_col].iloc[0] if p_col in sub.columns else np.nan
    stars = significance_stars(p)
    if stars == "":
        return

    # Convert "C(sample_id):C(group)" → "sample_id × group"
    if term_col in sub.columns:
        raw_term = sub[term_col].iloc[0]
        cleaned = (
            raw_term
            .replace("C(", "")
            .replace(")", "")
            .replace(":", " × ")
        )
    else:
        cleaned = "interaction"

    ax.text(
        x,
        y,
        f"{cleaned}: {stars}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )


def save_figure(fig, save_path: str | Path, *, dpi: int = 300, bbox_inches: str = "tight") -> None:
    """
    Save a figure to a single file.

    Parameters
    ----------
    fig
        Matplotlib figure object.
    save_path
        Full output path including suffix, e.g. 'plot.png'.
    dpi
        Output resolution.
    bbox_inches
        Passed to matplotlib savefig.
    """
    save_path = Path(save_path)
    base = save_path.with_suffix("")
    suffix = save_path.suffix.lstrip(".")
    if not suffix:
        raise ValueError("save_path must include a file suffix, e.g. '.png' or '.pdf'.")
    save_figure_multiple_formats(
        fig,
        base,
        formats=(suffix,),
        dpi=dpi,
        bbox_inches=bbox_inches,
    )


def save_figure_multiple_formats(
    fig,
    base_path: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """
    Save a figure in multiple formats.

    Parameters
    ----------
    fig
        Matplotlib figure object.
    base_path
        Base output path without suffix.
    formats
        File formats to write, e.g. ('png', 'pdf').
    dpi
        Output resolution.
    bbox_inches
        Passed to matplotlib savefig.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(base_path.with_suffix(f".{ext}"), dpi=dpi, bbox_inches=bbox_inches)


def _compute_summary_stats(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    error: str = "sd",
) -> pd.DataFrame:
    """
    Compute mean and error statistics for grouped plotting.

    Parameters
    ----------
    df
        Input dataframe.
    group_cols
        Columns defining the summary groups.
    value_col
        Numeric column to summarize.
    error
        Error metric to compute:
        - 'sd'  : standard deviation
        - 'sem' : standard error of the mean

    Returns
    -------
    pd.DataFrame
        Grouped dataframe with:
        - mean_value
        - sd_value
        - n
        - error_value
    """
    out = (
        df.groupby(list(group_cols), dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "std": "sd_value", "count": "n"})
    )

    if error == "sd":
        out["error_value"] = out["sd_value"]
    elif error == "sem":
        out["error_value"] = out["sd_value"] / np.sqrt(out["n"])
    else:
        raise ValueError("error must be 'sd' or 'sem'.")

    return out


def _coerce_sortable_timepoint(values: Sequence) -> list:
    """
    Sort timepoints numerically if possible, otherwise lexicographically.

    Parameters
    ----------
    values
        Sequence of timepoint labels.

    Returns
    -------
    list
        Ordered timepoint labels.
    """
    series = pd.Series(list(values), dtype="string")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        order = series.iloc[np.argsort(numeric.values)].tolist()
        return list(dict.fromkeys(order))
    return sorted(series.tolist())


def _build_color_lookup(levels: Sequence, colors: Mapping | Sequence | None = None) -> dict:
    """
    Create a mapping level -> color.

    Parameters
    ----------
    levels
        Distinct plotted levels.
    colors
        Either:
        - mapping level -> color
        - sequence of colors
        - None, in which case matplotlib defaults are used

    Returns
    -------
    dict
        Mapping from level to color.
    """
    levels = list(levels)

    if isinstance(colors, Mapping):
        return {level: colors.get(level, None) for level in levels}

    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not prop_cycle:
            prop_cycle = [None] * len(levels)
        return {level: prop_cycle[i % len(prop_cycle)] for i, level in enumerate(levels)}

    colors = list(colors)
    return {level: colors[i % len(colors)] for i, level in enumerate(levels)}


def _build_marker_lookup(levels: Sequence, markers: Mapping | Sequence | None = None) -> dict:
    """
    Create a mapping level -> marker.

    Parameters
    ----------
    levels
        Distinct plotted levels.
    markers
        Either:
        - mapping level -> marker
        - sequence of markers
        - None, in which case defaults are cycled

    Returns
    -------
    dict
        Mapping from level to marker.
    """
    levels = list(levels)

    if isinstance(markers, Mapping):
        return {level: markers.get(level, "o") for level in levels}

    if markers is None:
        default_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        return {level: default_markers[i % len(default_markers)] for i, level in enumerate(levels)}

    markers = list(markers)
    return {level: markers[i % len(markers)] for i, level in enumerate(levels)}


def _compute_y_limits(
    plot_df: pd.DataFrame,
    *,
    value_col: str,
    y_scale: str = "linear",
    y_limits: tuple[float, float] | None = None,
    pad_fraction: float = 0.08,
    min_pad: float = 0.5,
    upper_pad_multiplier: float = 1.0,
) -> tuple[float, float] | None:
    """
    Compute y-axis limits once for consistent subplot scaling.

    Parameters
    ----------
    plot_df
        Input plotting dataframe.
    value_col
        Numeric column plotted on the y-axis.
    y_scale
        'linear' or 'log'.
    y_limits
        Optional manually supplied limits.
    pad_fraction
        Fractional padding added above and below the observed range.
    min_pad
        Minimum absolute padding if the range is very small.
    upper_pad_multiplier
        Extra multiplier for top padding, useful for significance annotations.

    Returns
    -------
    tuple[float, float] | None
        Computed y-axis limits, or None if no valid values exist.
    """
    if y_limits is not None:
        return y_limits

    y_vals = pd.to_numeric(plot_df[value_col], errors="coerce").dropna()
    if y_vals.empty:
        return None

    if y_scale == "log":
        positive_vals = y_vals[y_vals > 0]
        if positive_vals.empty:
            raise ValueError("Cannot use log y-scale because no positive values were found.")
        return (positive_vals.min() * 0.8, positive_vals.max() * 1.2)

    y_min = y_vals.min()
    y_max = y_vals.max()
    pad = pad_fraction * (y_max - y_min) if y_max > y_min else min_pad
    return (y_min - pad, y_max + upper_pad_multiplier * pad)


def _style_y_axis(ax, *, y_scale: str = "linear", y_limits: tuple[float, float] | None = None) -> None:
    """
    Apply consistent y-axis styling to one axis.

    Parameters
    ----------
    ax
        Matplotlib axis.
    y_scale
        'linear' or 'log'.
    y_limits
        Optional y-axis limits.
    """
    ax.set_yscale(y_scale)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    if y_scale == "linear":
        ax.yaxis.set_major_locator(mticker.AutoLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    else:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    ax.grid(axis="y", which="major", linewidth=0.8, alpha=0.4)
    ax.grid(axis="y", which="minor", linewidth=0.5, alpha=0.2)


def _scatter_points(ax, xs, ys, *, color, marker, point_size: float) -> None:
    """
    Draw hollow scatter points.

    Parameters
    ----------
    ax
        Matplotlib axis.
    xs, ys
        Coordinates to plot.
    color
        Edge color.
    marker
        Marker symbol.
    point_size
        Marker size.
    """
    ax.scatter(
        xs,
        ys,
        facecolors="none",
        edgecolors=color,
        s=point_size,
        marker=marker,
        linewidths=1.0,
    )


def _draw_mean_error(
    ax,
    x,
    mean_val,
    err_val,
    *,
    color,
    marker,
    mean_marker_size: float,
    errorbar_capsize: float,
) -> None:
    """
    Draw a summary point and error bar.

    Parameters
    ----------
    ax
        Matplotlib axis.
    x
        X-position.
    mean_val
        Mean value.
    err_val
        Error value (SD or SEM).
    color
        Marker and line color.
    marker
        Marker symbol.
    mean_marker_size
        Marker size for the mean point.
    errorbar_capsize
        Capsize for the error bar.
    """
    ax.errorbar(
        x,
        mean_val,
        yerr=err_val,
        fmt=marker,
        markersize=mean_marker_size,
        capsize=errorbar_capsize,
        color=color,
        linewidth=1.2,
    )


def _draw_grouped_points_and_summary(
    ax,
    *,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    level_col: str,
    value_col: str,
    levels: Sequence,
    x_positions: Mapping,
    color_lookup: Mapping,
    marker_lookup: Mapping,
    point_size: float,
    mean_marker_size: float,
    errorbar_capsize: float,
) -> None:
    """
    Draw raw points and mean/error summaries for grouped categorical data.

    Parameters
    ----------
    ax
        Matplotlib axis.
    raw_df
        Raw plotting dataframe.
    summary_df
        Summary statistics dataframe for the same panel.
    level_col
        Column defining the plotted groups.
    value_col
        Numeric y-value column.
    levels
        Ordered plotted levels.
    x_positions
        Mapping from level to x-position.
    color_lookup
        Mapping level -> color.
    marker_lookup
        Mapping level -> marker.
    point_size
        Raw point size.
    mean_marker_size
        Summary marker size.
    errorbar_capsize
        Error-bar capsize.
    """
    for level in levels:
        vals = pd.to_numeric(
            raw_df.loc[raw_df[level_col] == level, value_col],
            errors="coerce",
        ).dropna()

        if vals.empty:
            continue

        x = x_positions[level]

        _scatter_points(
            ax,
            np.full(len(vals), x, dtype=float),
            vals.values,
            color=color_lookup[level],
            marker=marker_lookup[level],
            point_size=point_size,
        )

        row = summary_df.loc[summary_df[level_col] == level]
        if not row.empty:
            _draw_mean_error(
                ax,
                x,
                row["mean_value"].iloc[0],
                row["error_value"].iloc[0],
                color=color_lookup[level],
                marker=marker_lookup[level],
                mean_marker_size=mean_marker_size,
                errorbar_capsize=errorbar_capsize,
            )


def _build_legend_handles(levels: Sequence, *, color_lookup: Mapping, marker_lookup: Mapping) -> list:
    """
    Build explicit legend handles for consistent legends.

    Parameters
    ----------
    levels
        Ordered legend levels.
    color_lookup
        Mapping level -> color.
    marker_lookup
        Mapping level -> marker.

    Returns
    -------
    list
        Matplotlib legend handles.
    """
    handles = []
    for level in levels:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color_lookup[level],
                marker=marker_lookup[level],
                linestyle="None",
                markersize=7,
                markerfacecolor="none",
                label=str(level),
            )
        )
    return handles


def _add_significance_annotations(
    ax,
    *,
    stats_df: pd.DataFrame | None,
    target: str,
    position_lookup: dict,
    y_limits: tuple[float, float] | None,
    target_col: str = "target",
    group1_col: str = "group_1",
    group2_col: str = "group_2",
    p_col: str = "p_adj",
    show_ns: bool = True,
) -> None:
    """
    Add pairwise significance bars for one plotted target.

    This version places overlapping comparisons on separate levels while
    allowing non-overlapping comparisons to share the same vertical space.
    That produces a more compact and readable annotation layout than simple
    linear stacking.

    Parameters
    ----------
    ax
        Matplotlib axis.
    stats_df
        Statistics dataframe, typically from pairwise comparisons.
    target
        Target gene currently plotted.
    position_lookup
        Mapping from plotted group labels to x positions.
    y_limits
        Y-axis limits used for annotation placement.
    target_col
        Target column in stats_df.
    group1_col, group2_col
        Comparison-group columns in stats_df.
    p_col
        P-value column used for star labels.
    show_ns
        Whether to show 'ns' annotations.
    """
    if stats_df is None or stats_df.empty or not position_lookup:
        return

    _require_columns(stats_df, [target_col, group1_col, group2_col], "significance annotation table")

    sub = stats_df.loc[stats_df[target_col] == target].copy()
    if sub.empty:
        return

    intervals = []
    for _, row in sub.iterrows():
        g1 = str(row[group1_col])
        g2 = str(row[group2_col])

        if g1 not in position_lookup or g2 not in position_lookup:
            continue

        p = row[p_col] if p_col in row and pd.notna(row[p_col]) else row.get("p_value", np.nan)
        label = significance_stars(p)

        if label == "":
            continue
        if label == "ns" and not show_ns:
            continue

        x1 = position_lookup[g1]
        x2 = position_lookup[g2]
        if x1 > x2:
            x1, x2 = x2, x1

        intervals.append({"x1": x1, "x2": x2, "label": label, "span": x2 - x1})

    if not intervals:
        return

    intervals = sorted(intervals, key=lambda d: (d["span"], d["x1"], d["x2"]))

    ymin, ymax = ax.get_ylim() if y_limits is None else y_limits
    yrange = ymax - ymin if ymax > ymin else 1.0

    base_y = ymax - 0.25 * yrange
    bar_height = 0.03 * yrange
    text_offset = 0.012 * yrange
    level_step = 0.085 * yrange

    placed_levels: list[list[tuple[float, float]]] = []

    for item in intervals:
        x1 = item["x1"]
        x2 = item["x2"]
        label = item["label"]

        level_idx = 0
        while True:
            if level_idx >= len(placed_levels):
                placed_levels.append([])
                break

            overlaps = any(not (x2 < px1 or x1 > px2) for px1, px2 in placed_levels[level_idx])
            if not overlaps:
                break

            level_idx += 1

        placed_levels[level_idx].append((x1, x2))
        current_y = base_y + level_idx * level_step

        ax.plot(
            [x1, x1, x2, x2],
            [current_y, current_y + bar_height, current_y + bar_height, current_y],
            color="black",
            linewidth=1.0,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            current_y + bar_height + text_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )


# ============================================================================
# Main plotting functions
# ============================================================================

def plot_expression_grid(
    df: pd.DataFrame,
    *,
    gene_col: str = "target",
    condition_col: str = "group",
    x_col: str = "sample_id",
    value_col: str = "log2_normalized_expression",
    gene_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
    x_order: Sequence[str] | None = None,
    error: str = "sd",
    colors: Mapping | Sequence | None = None,
    markers: Mapping | Sequence | None = None,
    point_size: float = 35,
    mean_marker_size: float = 7,
    errorbar_capsize: float = 4,
    figsize: tuple[float, float] | None = None,
    sharey: bool = True,
    y_limits: tuple[float, float] | None = None,
    y_scale: str = "linear",
    title: str | None = None,
    y_label: str = "log2 normalized expression",
    stats_df: pd.DataFrame | None = None,
    stats_target_col: str = "target",
    stats_group1_col: str = "group_1",
    stats_group2_col: str = "group_2",
    stats_p_col: str = "p_adj",
    annotate_stats: bool = False,
    show_ns: bool = True,
    annotation_factor_cols: Sequence[str] | None = None,
    annotation_sep: str = " | ",
):
    """
    Plot a grid of expression values for multiple genes and conditions.

    Rows correspond to genes, columns correspond to conditions. Within each
    panel, samples are shown as raw points with overlaid summary statistics.

    For significance annotations, this grid uses a clean panel-specific rule:
    only comparisons that belong to the current condition panel are shown.
    With the default setup of x_col='sample_id' and condition_col='group',
    that means strain comparisons within each condition.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing expression values.
    gene_col : str
        Column identifying genes.
    condition_col : str
        Column defining subplot columns.
    x_col : str
        Column defining x-axis categories within each subplot.
    value_col : str
        Column containing values to plot.
    gene_order, condition_order, x_order : sequence of str, optional
        Explicit ordering of categories.
    error : {"sd", "sem"}
        Type of error bars.
    colors, markers : mapping or sequence, optional
        Custom colors/markers.
    point_size, mean_marker_size : float
        Sizes for points and summary markers.
    errorbar_capsize : float
        Width of error bar caps.
    figsize : tuple, optional
        Figure size.
    sharey : bool
        Whether to share y-axis across subplots.
    y_limits : tuple, optional
        Explicit y-axis limits.
    y_scale : {"linear", "log"}
        Y-axis scaling.
    title : str, optional
        Overall figure title.
    y_label : str
        Label for y-axis.
    stats_df : pd.DataFrame, optional
        Pairwise comparison results.
    stats_target_col, stats_group1_col, stats_group2_col, stats_p_col : str
        Column names in stats_df.
    annotate_stats : bool
        Whether to draw significance bars.
    show_ns : bool
        Whether to display non-significant comparisons.
    annotation_factor_cols : sequence of str, optional
        Columns used to reconstruct comparison labels.
        Default = [x_col, condition_col].
    annotation_sep : str
        Separator used when building comparison labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    required = [gene_col, condition_col, x_col, value_col]
    plot_df = _prepare_plot_df(df, required_cols=required, context="plot_expression_grid")

    genes = _ordered_levels(plot_df, gene_col, gene_order)
    conditions = _ordered_levels(plot_df, condition_col, condition_order)
    x_levels = _ordered_levels(plot_df, x_col, x_order)

    color_lookup = _build_color_lookup(x_levels, colors=colors)
    marker_lookup = _build_marker_lookup(x_levels, markers=markers)

    n_rows = len(genes)
    n_cols = len(conditions)

    if figsize is None:
        figsize = (4.0 * n_cols, 2.8 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharey=sharey,
    )

    summary_df = _compute_summary_stats(
        plot_df,
        [gene_col, condition_col, x_col],
        value_col,
        error=error,
    )

    global_y_limits = _compute_y_limits(
        plot_df,
        value_col=value_col,
        y_scale=y_scale,
        y_limits=y_limits,
        upper_pad_multiplier=2.0 if annotate_stats else 1.0,
        pad_fraction=0.10 if annotate_stats else 0.08,
    )

    x_positions = {level: i for i, level in enumerate(x_levels)}
    annotation_factor_cols = [x_col, condition_col] if annotation_factor_cols is None else list(annotation_factor_cols)

    for i, gene in enumerate(genes):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]

            sub_df = plot_df.loc[
                (plot_df[gene_col] == gene) &
                (plot_df[condition_col] == condition)
            ].copy()

            if sub_df.empty:
                ax.set_visible(False)
                continue

            sub_summary = summary_df.loc[
                (summary_df[gene_col] == gene) &
                (summary_df[condition_col] == condition)
            ].copy()

            _draw_grouped_points_and_summary(
                ax,
                raw_df=sub_df,
                summary_df=sub_summary,
                level_col=x_col,
                value_col=value_col,
                levels=x_levels,
                x_positions=x_positions,
                color_lookup=color_lookup,
                marker_lookup=marker_lookup,
                point_size=point_size,
                mean_marker_size=mean_marker_size,
                errorbar_capsize=errorbar_capsize,
            )

            ax.set_xticks(range(len(x_levels)))
            ax.set_xticklabels(x_levels, rotation=45, ha="right")
            ax.set_xlim(-0.5, len(x_levels) - 0.5)
            _style_y_axis(ax, y_scale=y_scale, y_limits=global_y_limits)

            if i == 0:
                ax.set_title(str(condition))
            if j == 0:
                ax.set_ylabel(f"{gene}\n{y_label}")
            else:
                ax.set_ylabel("")

            if annotate_stats and stats_df is not None and not stats_df.empty:
                position_lookup = {
                    _build_annotation_key(
                        {x_col: x_level, condition_col: condition},
                        annotation_factor_cols,
                        sep=annotation_sep,
                    ): x_positions[x_level]
                    for x_level in x_levels
                }

                # Keep only comparisons that belong to this panel:
                # same condition, different x-levels.

                def _extract_condition(label: str, sep: str):
                    return label.split(sep)[-1]
                
                panel_stats = stats_df.loc[
                    (stats_df[stats_target_col] == gene)
                ].copy()
                
                panel_stats = panel_stats[
                    panel_stats[stats_group1_col].apply(lambda x: _extract_condition(str(x), annotation_sep) == str(condition)) &
                    panel_stats[stats_group2_col].apply(lambda x: _extract_condition(str(x), annotation_sep) == str(condition))
                ]

                _add_significance_annotations(
                    ax,
                    stats_df=panel_stats,
                    target=gene,
                    position_lookup=position_lookup,
                    y_limits=global_y_limits,
                    target_col=stats_target_col,
                    group1_col=stats_group1_col,
                    group2_col=stats_group2_col,
                    p_col=stats_p_col,
                    show_ns=show_ns,
                )

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes

# ----------------------------------------------------

def plot_timecourse_grid(
    df: pd.DataFrame,
    *,
    gene_col: str = "target",
    condition_col: str = "group",
    time_col: str = "timepoint",
    line_col: str = "sample_id",
    value_col: str = "log2_normalized_expression",
    gene_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
    time_order: Sequence[str] | None = None,
    line_order: Sequence[str] | None = None,
    error: str = "sd",
    colors: Mapping | Sequence | None = None,
    markers: Mapping | Sequence | None = None,
    point_size: float = 28,
    line_width: float = 1.2,
    mean_marker_size: float = 6,
    errorbar_capsize: float = 3,
    figsize: tuple[float, float] | None = None,
    sharey: bool = True,
    y_limits: tuple[float, float] | None = None,
    y_scale: str = "linear",
    title: str | None = None,
    y_label: str = "log2 normalized expression",
):
    """
    Plot genes as rows, conditions as columns, and timepoints on the x-axis.

    Parameters
    ----------
    df
        Input dataframe with biological replicate-level values.
    gene_col
        Column defining subplot rows.
    condition_col
        Column defining subplot columns.
    time_col
        Timepoint column plotted on the x-axis.
    line_col
        Column defining separate lines within each panel.
    value_col
        Numeric y-axis column.
    gene_order, condition_order, time_order, line_order
        Optional explicit ordering for plotted levels.
    error
        Error metric for summary overlays:
        - 'sd'
        - 'sem'
    colors
        Color mapping or sequence for line levels.
    markers
        Marker mapping or sequence for line levels.
    point_size
        Raw point size.
    line_width
        Width of the summary line.
    mean_marker_size
        Marker size for the summary line.
    errorbar_capsize
        Error-bar capsize.
    figsize
        Figure size. If None, chosen automatically.
    sharey
        Whether subplots share a common y-axis scale.
    y_limits
        Optional manual y-axis limits.
    y_scale
        'linear' or 'log'.
    title
        Optional figure title.
    y_label
        Base y-axis label.

    Returns
    -------
    tuple
        (fig, axes)
    """
    required = [gene_col, condition_col, time_col, line_col, value_col]
    plot_df = _prepare_plot_df(df, required_cols=required, context="plot_timecourse_grid")

    genes = _ordered_levels(plot_df, gene_col, gene_order)
    conditions = _ordered_levels(plot_df, condition_col, condition_order)
    lines = _ordered_levels(plot_df, line_col, line_order)
    times = _ordered_levels(plot_df, time_col, time_order)

    if time_order is None:
        times = _coerce_sortable_timepoint(times)

    color_lookup = _build_color_lookup(lines, colors=colors)
    marker_lookup = _build_marker_lookup(lines, markers=markers)

    n_rows = len(genes)
    n_cols = len(conditions)

    if figsize is None:
        figsize = (4.2 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharey=sharey)

    summary_df = _compute_summary_stats(
        plot_df,
        group_cols=[gene_col, condition_col, time_col, line_col],
        value_col=value_col,
        error=error,
    )

    global_y_limits = _compute_y_limits(
        plot_df,
        value_col=value_col,
        y_scale=y_scale,
        y_limits=y_limits,
    )

    time_pos = {t: i for i, t in enumerate(times)}

    for i, gene in enumerate(genes):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]

            sub_df = plot_df.loc[
                (plot_df[gene_col] == gene) &
                (plot_df[condition_col] == condition)
            ].copy()

            if sub_df.empty:
                ax.set_visible(False)
                continue

            sub_summary = summary_df.loc[
                (summary_df[gene_col] == gene) &
                (summary_df[condition_col] == condition)
            ].copy()

            for line_level in lines:
                line_summary = sub_summary.loc[sub_summary[line_col] == line_level].copy()
                if line_summary.empty:
                    continue

                line_summary["_x"] = line_summary[time_col].map(time_pos)
                line_summary = line_summary.sort_values("_x")

                raw_sub = sub_df.loc[sub_df[line_col] == line_level].copy()
                raw_sub["_x"] = raw_sub[time_col].map(time_pos)

                _scatter_points(
                    ax,
                    raw_sub["_x"].values,
                    pd.to_numeric(raw_sub[value_col], errors="coerce").values,
                    color=color_lookup[line_level],
                    marker=marker_lookup[line_level],
                    point_size=point_size,
                )

                ax.plot(
                    line_summary["_x"],
                    line_summary["mean_value"],
                    linewidth=line_width,
                    marker=marker_lookup[line_level],
                    markersize=mean_marker_size,
                    color=color_lookup[line_level],
                    label=str(line_level),
                )

                ax.errorbar(
                    line_summary["_x"],
                    line_summary["mean_value"],
                    yerr=line_summary["error_value"],
                    fmt="none",
                    capsize=errorbar_capsize,
                    color=color_lookup[line_level],
                    linewidth=1.0,
                )

            ax.set_xticks(range(len(times)))
            ax.set_xticklabels(times, rotation=45, ha="right")
            ax.set_xlim(-0.4, len(times) - 0.6)
            _style_y_axis(ax, y_scale=y_scale, y_limits=global_y_limits)

            if i == 0:
                ax.set_title(str(condition))
            if j == 0:
                ax.set_ylabel(f"{gene}\n{y_label}")
            else:
                ax.set_ylabel("")

    handles = _build_legend_handles(lines, color_lookup=color_lookup, marker_lookup=marker_lookup)
    if handles:
        fig.legend(handles=handles, loc="upper right", frameon=False)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, axes

# ----------------------------------------------------

def plot_single_gene(
    df: pd.DataFrame,
    *,
    gene: str,
    gene_col: str = "target",
    x_col: str = "sample_id",
    hue_col: str = "group",
    value_col: str = "log2_normalized_expression",
    x_order: Sequence[str] | None = None,
    hue_order: Sequence[str] | None = None,
    error: str = "sd",
    colors: Mapping | Sequence | None = None,
    markers: Mapping | Sequence | None = None,
    dodge: float = 0.20,
    point_size: float = 35,
    mean_marker_size: float = 7,
    errorbar_capsize: float = 4,
    figsize: tuple[float, float] = (6.5, 4.5),
    y_limits: tuple[float, float] | None = None,
    y_scale: str = "linear",
    title: str | None = None,
    y_label: str = "log2 normalized expression",
    stats_df: pd.DataFrame | None = None,
    stats_target_col: str = "target",
    stats_group1_col: str = "group_1",
    stats_group2_col: str = "group_2",
    stats_p_col: str = "p_adj",
    annotate_stats: bool = False,
    show_ns: bool = True,
    annotation_factor_cols: Sequence[str] | None = None,
    annotation_sep: str = " | ",
    title_pad: float = 16.0,
    interaction_df: pd.DataFrame | None = None,
    interaction_target_col: str = "target",
    interaction_p_col: str = "p_adj",
    show_interaction: bool = False,
):
    """
    Plot expression values for a single target gene with optional significance annotations.

    This function visualizes individual biological replicates alongside summary
    statistics (mean ± SD or SEM) for one gene. Groups are defined by `x_col`
    (x-axis categories) and `hue_col` (color-coded subgroups).

    Optionally, pairwise statistical comparisons can be overlaid as significance
    bars using results from `run_pairwise_comparisons()`. An optional interaction
    label can also be shown, for example to display the significance of a strain
    × condition interaction extracted from model terms.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing normalized expression values.
    gene : str
        Target gene to plot.
    gene_col : str
        Column identifying target genes.
    x_col : str
        Column defining x-axis categories (e.g. sample_id).
    hue_col : str
        Column defining color-coded groups (e.g. treatment).
    value_col : str
        Column containing values to plot (e.g. normalized_expression).
    x_order, hue_order : sequence of str, optional
        Explicit ordering of x-axis or hue levels.
    error : {"sd", "sem"}
        Type of error bar to display.
    colors, markers : mapping or sequence, optional
        Custom color or marker definitions.
    dodge : float
        Horizontal offset for separating hue groups.
    point_size : float
        Size of individual data points.
    mean_marker_size : float
        Size of summary markers.
    errorbar_capsize : float
        Width of error bar caps.
    figsize : tuple
        Figure size.
    y_limits : tuple, optional
        Explicit y-axis limits.
    y_scale : {"linear", "log"}
        Scaling of y-axis.
    title : str, optional
        Plot title (defaults to gene name).
    y_label : str
        Label for y-axis.
    stats_df : pd.DataFrame, optional
        Pairwise comparison results.
    stats_target_col, stats_group1_col, stats_group2_col, stats_p_col : str
        Column names in stats_df.
    annotate_stats : bool
        Whether to draw significance bars.
    show_ns : bool
        Whether to display "ns" labels.
    annotation_factor_cols : sequence of str, optional
        Columns used to reconstruct group labels.
        Default = [x_col, hue_col].
    annotation_sep : str
        Separator used when building comparison labels.
    title_pad : float
        Padding above plot title.
    interaction_df : pd.DataFrame, optional
        Table containing one interaction p-value per target.
    interaction_target_col : str
        Target column in interaction_df.
    interaction_p_col : str
        P-value column in interaction_df.
    show_interaction : bool
        Whether to show the interaction label.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    required = [gene_col, x_col, hue_col, value_col]
    _require_columns(df, required, "plot_single_gene input")

    plot_df = df.loc[df[gene_col] == gene].copy().dropna(subset=required)
    if plot_df.empty:
        raise ValueError(f"No rows found for gene={gene!r}")

    x_levels = _ordered_levels(plot_df, x_col, x_order)
    hue_levels = _ordered_levels(plot_df, hue_col, hue_order)

    color_lookup = _build_color_lookup(hue_levels, colors=colors)
    marker_lookup = _build_marker_lookup(hue_levels, markers=markers)

    fig, ax = plt.subplots(figsize=figsize)

    summary_df = _compute_summary_stats(
        plot_df,
        [x_col, hue_col],
        value_col,
        error=error,
    )

    computed_y_limits = _compute_y_limits(
        plot_df,
        value_col=value_col,
        y_scale=y_scale,
        y_limits=y_limits,
        upper_pad_multiplier=5.0 if annotate_stats else 1.0,
        pad_fraction=0.16 if annotate_stats else 0.08,
    )

    hue_offsets = (
        np.linspace(-dodge, dodge, len(hue_levels))
        if len(hue_levels) > 1
        else np.array([0.0])
    )

    annotation_factor_cols = [x_col, hue_col] if annotation_factor_cols is None else list(annotation_factor_cols)
    _require_columns(plot_df, annotation_factor_cols, "plot_single_gene annotation factor columns")

    position_lookup = {}

    for i, x_level in enumerate(x_levels):
        for h, hue_level in enumerate(hue_levels):
            vals = pd.to_numeric(
                plot_df.loc[
                    (plot_df[x_col] == x_level) &
                    (plot_df[hue_col] == hue_level),
                    value_col,
                ],
                errors="coerce",
            ).dropna()

            if vals.empty:
                continue

            center = i + hue_offsets[h]

            key_values = {x_col: x_level, hue_col: hue_level}
            position_key = _build_annotation_key(
                key_values,
                annotation_factor_cols,
                sep=annotation_sep,
            )
            position_lookup[position_key] = center

            _scatter_points(
                ax,
                np.full(len(vals), center, dtype=float),
                vals.values,
                color=color_lookup[hue_level],
                marker=marker_lookup[hue_level],
                point_size=point_size,
            )

            row = summary_df.loc[
                (summary_df[x_col] == x_level) &
                (summary_df[hue_col] == hue_level)
            ]
            if not row.empty:
                _draw_mean_error(
                    ax,
                    center,
                    row["mean_value"].iloc[0],
                    row["error_value"].iloc[0],
                    color=color_lookup[hue_level],
                    marker=marker_lookup[hue_level],
                    mean_marker_size=mean_marker_size,
                    errorbar_capsize=errorbar_capsize,
                )

        ax.axvline(i + 0.5, linewidth=0.4, alpha=0.15, color="black")

    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(x_levels) - 0.5)
    _style_y_axis(ax, y_scale=y_scale, y_limits=computed_y_limits)
    ax.set_ylabel(y_label)

    handles = _build_legend_handles(hue_levels, color_lookup=color_lookup, marker_lookup=marker_lookup)
    if handles:
        ax.legend(handles=handles, frameon=False, title=hue_col)

    if annotate_stats:
        _add_significance_annotations(
            ax,
            stats_df=stats_df,
            target=gene,
            position_lookup=position_lookup,
            y_limits=computed_y_limits,
            target_col=stats_target_col,
            group1_col=stats_group1_col,
            group2_col=stats_group2_col,
            p_col=stats_p_col,
            show_ns=show_ns,
        )

    # 1. Draw significance first (so we know how much space is needed)
    if annotate_stats:
        _add_significance_annotations(
            ax,
            stats_df=stats_df,
            target=gene,
            position_lookup=position_lookup,
            y_limits=computed_y_limits,
            target_col=stats_target_col,
            group1_col=stats_group1_col,
            group2_col=stats_group2_col,
            p_col=stats_p_col,
            show_ns=show_ns,
        )
    
    interaction_label = None

    if show_interaction and interaction_df is not None:
        sub = interaction_df.loc[interaction_df[interaction_target_col] == gene]
        if not sub.empty:
            p = sub[interaction_p_col].iloc[0] if interaction_p_col in sub.columns else None
            stars = significance_stars(p)
            if stars:
                interaction_label = f"{' × '.join(annotation_factor_cols)}: {stars}"
        
    title_text = str(gene) if title is None else title

    if interaction_label:
        title_text = f"{title_text}\n{interaction_label}"
    
    ax.set_title(
        title_text,
        pad=28,
        fontsize=14,
        weight="bold",
    )
    
    # 4. Reserve space at top
    fig.tight_layout(rect=(0, 0, 1, 0.82))
        
    return fig, ax


def plot_calibrator_offsets(
    offsets_df: pd.DataFrame,
    *,
    target_col: str = "target",
    plate_col: str = "plate_id",
    offset_col: str = "offset_ct",
    target_order: Sequence[str] | None = None,
    plate_order: Sequence[str] | None = None,
    colors: Mapping | Sequence | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = "Inter-plate calibrator offsets",
):
    """
    Plot calibrator offsets by target and plate.

    Parameters
    ----------
    offsets_df
        Dataframe containing one offset per target and plate.
    target_col
        Column defining subplot rows.
    plate_col
        Column defining bar groups within each subplot.
    offset_col
        Numeric offset column to plot.
    target_order
        Optional explicit target order.
    plate_order
        Optional explicit plate order.
    colors
        Color mapping or sequence for plate levels.
    figsize
        Figure size. If None, chosen automatically.
    title
        Optional figure title.

    Returns
    -------
    tuple
        (fig, axes)
    """
    required = [target_col, plate_col, offset_col]
    plot_df = _prepare_plot_df(offsets_df, required_cols=required, context="plot_calibrator_offsets")

    targets = _ordered_levels(plot_df, target_col, target_order)
    plates = _ordered_levels(plot_df, plate_col, plate_order)
    color_lookup = _build_color_lookup(plates, colors=colors)

    n_rows = len(targets)
    if figsize is None:
        figsize = (6.0, 2.6 * n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False, sharex=True)

    for i, target in enumerate(targets):
        ax = axes[i, 0]
        sub_df = plot_df.loc[plot_df[target_col] == target].copy()
        if sub_df.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(plates))
        y = []
        bar_colors = []

        for plate in plates:
            row = sub_df.loc[sub_df[plate_col] == plate]
            y.append(np.nan if row.empty else row[offset_col].iloc[0])
            bar_colors.append(color_lookup[plate])

        ax.bar(x, y, color=bar_colors)
        ax.axhline(0, linewidth=1.0, linestyle="--", color="black")
        ax.set_ylabel(f"{target}\nΔCt")
        _style_y_axis(ax, y_scale="linear", y_limits=None)

    axes[-1, 0].set_xticks(np.arange(len(plates)))
    axes[-1, 0].set_xticklabels(plates, rotation=45, ha="right")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, axes


__all__ = [
    "significance_stars",
    "has_real_values",
    "save_figure",
    "save_figure_multiple_formats",
    "plot_expression_grid",
    "plot_timecourse_grid",
    "plot_single_gene",
    "plot_calibrator_offsets",
]