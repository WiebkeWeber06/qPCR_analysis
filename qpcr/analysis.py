"""
Analysis utilities for qPCR data.

Design goals
------------
- support one or multiple reference genes
- support efficiency-aware calculations
- keep normalized expression as the primary output
- make fold change optional, not mandatory
- support classic 2^-ΔΔCt as a special case when efficiencies are all 2.0

Conceptual note
---------------
This module follows a quantity-based normalization approach (Pfaffl-style):
Ct values are first converted to relative quantities, which are then normalized
to one or more reference genes.

When:
- all efficiencies are 2.0, and
- one reference gene is used,

this becomes equivalent to the classical 2^-ΔCt / 2^-ΔΔCt framework.

Core idea
---------
1. Start from summarized technical replicates (one row per bio_rep + target)
2. Convert Ct to relative quantity:
       RQ = efficiency ** (-Ct)
   where efficiency is the amplification factor:
       2.0 = 100% efficiency
       1.95 = slightly lower than ideal
3. Combine one or more reference genes per biological sample
4. Normalize target quantity by the combined reference quantity
5. Optionally compare normalized expression to a control group

Typical usage
-------------
>>> from qpcr.analysis import calculate_normalized_expression
>>> result = calculate_normalized_expression(
...     df=summary_df,
...     reference_targets=["GBLP", "RPL13"],
... )
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_ID_COLUMNS: tuple[str, ...] = ("plate_id", "group", "sample_id", "bio_rep")
DEFAULT_TARGET_COLUMN: str = "target"
DEFAULT_CT_COLUMN: str = "ct_mean"
DEFAULT_EFFICIENCY: float = 2.0


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _warn(message: str) -> None:
    """Emit a standard user warning."""
    warnings.warn(message, UserWarning, stacklevel=2)


def _validate_unique_measurements(
    df: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    target_col: str,
    context: str,
) -> None:
    """
    Ensure there is at most one row per biological sample and target.

    This is important after technical-replicate summarization. If duplicates
    remain, normalization can silently misbehave.
    """
    key_cols = list(id_cols) + [target_col]
    duplicates = df.duplicated(subset=key_cols, keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, key_cols].head(10).to_dict("records")
        raise ValueError(
            f"Expected one row per biological sample and target in {context}, "
            f"but found duplicates. Examples: {examples}"
        )


def _geometric_mean(series: pd.Series) -> float:
    """Geometric mean for positive values."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    vals = vals[vals > 0]
    if len(vals) == 0:
        return np.nan
    return float(np.exp(np.log(vals).mean()))


def attach_efficiencies(
    df: pd.DataFrame,
    efficiency_df: pd.DataFrame | None = None,
    *,
    target_col: str = DEFAULT_TARGET_COLUMN,
    efficiency_col: str = "efficiency",
    default_efficiency: float = DEFAULT_EFFICIENCY,
    allow_missing_efficiencies: bool = False,
    warn_on_unusual_efficiencies: bool = True,
    min_reasonable_efficiency: float = 1.5,
    max_reasonable_efficiency: float = 2.2,
) -> pd.DataFrame:
    """
    Attach per-target efficiencies to a dataframe.

    Efficiency is expected as amplification factor:
    - 2.0 means 100% efficiency
    - 1.9 means lower efficiency

    Parameters
    ----------
    df
        Input dataframe with target identities.
    efficiency_df
        Optional dataframe mapping targets to efficiencies.
    target_col
        Target column name.
    efficiency_col
        Name of the efficiency column in efficiency_df.
    default_efficiency
        Default amplification factor to use when no efficiency_df is supplied,
        or when allow_missing_efficiencies=True and some targets are missing.
    allow_missing_efficiencies
        If False, missing efficiencies in efficiency_df raise an error.
        If True, missing targets are filled with default_efficiency.
    warn_on_unusual_efficiencies
        If True, emit a warning when efficiencies fall outside a typical qPCR range.
    min_reasonable_efficiency
        Lower bound for the warning range.
    max_reasonable_efficiency
        Upper bound for the warning range.

    Returns
    -------
    pd.DataFrame
        Dataframe with an `efficiency` column.
    """
    out = df.copy()
    _require_columns(out, [target_col], context="attach_efficiencies input")

    if efficiency_df is None:
        out["efficiency"] = float(default_efficiency)
    else:
        _require_columns(
            efficiency_df,
            [target_col, efficiency_col],
            context="efficiency dataframe",
        )

        eff = efficiency_df[[target_col, efficiency_col]].copy()
        eff = eff.rename(columns={efficiency_col: "efficiency"})

        out = out.merge(eff, on=target_col, how="left", validate="many_to_one")

        missing_mask = out["efficiency"].isna()
        if missing_mask.any():
            missing_targets = out.loc[missing_mask, target_col].dropna().unique().tolist()

            if allow_missing_efficiencies:
                out.loc[missing_mask, "efficiency"] = float(default_efficiency)
            else:
                raise ValueError(
                    "Missing efficiency values for targets: "
                    f"{missing_targets}. Provide them in efficiency_df, "
                    "or set allow_missing_efficiencies=True to fill with "
                    f"default_efficiency={default_efficiency}."
                )

    out["efficiency"] = pd.to_numeric(out["efficiency"], errors="raise")

    if (out["efficiency"] <= 0).any():
        raise ValueError("All efficiencies must be > 0.")

    if warn_on_unusual_efficiencies:
        unusual = out.loc[
            (out["efficiency"] < min_reasonable_efficiency)
            | (out["efficiency"] > max_reasonable_efficiency),
            [target_col, "efficiency"],
        ].drop_duplicates()

        if not unusual.empty:
            examples = unusual.head(10).to_dict("records")
            _warn(
                "Some efficiencies fall outside the typical qPCR range "
                f"({min_reasonable_efficiency}–{max_reasonable_efficiency}). "
                f"Examples: {examples}"
            )

    return out


def calculate_relative_quantity(
    df: pd.DataFrame,
    *,
    ct_col: str = DEFAULT_CT_COLUMN,
    efficiency_col: str = "efficiency",
    output_col: str = "relative_quantity",
) -> pd.DataFrame:
    """
    Convert Ct values to relative quantities.

    Formula
    -------
    relative_quantity = efficiency ** (-Ct)
    """
    out = df.copy()
    _require_columns(out, [ct_col, efficiency_col], context="calculate_relative_quantity input")

    out[ct_col] = pd.to_numeric(out[ct_col], errors="coerce")
    out[efficiency_col] = pd.to_numeric(out[efficiency_col], errors="coerce")

    if (out[efficiency_col] <= 0).any():
        raise ValueError("All efficiencies must be > 0.")

    out[output_col] = out[efficiency_col] ** (-out[ct_col])
    return out


def combine_reference_genes(
    df: pd.DataFrame,
    reference_targets: Sequence[str],
    *,
    id_cols: Sequence[str] = DEFAULT_ID_COLUMNS,
    target_col: str = DEFAULT_TARGET_COLUMN,
    quantity_col: str = "relative_quantity",
    method: str = "geometric_mean",
    require_all_reference_targets: bool = True,
    min_reference_genes: int | None = None,
) -> pd.DataFrame:
    """
    Combine one or more reference genes into one reference quantity per biological sample.

    Parameters
    ----------
    df
        Input dataframe containing relative quantities.
    reference_targets
        Names of the reference genes.
    id_cols
        Columns defining one biological sample.
    target_col
        Column identifying targets.
    quantity_col
        Column containing relative quantity values.
    method
        Aggregation method for combining multiple reference genes.
        Currently only 'geometric_mean' is supported.
    require_all_reference_targets
        If True, require every biological sample to contain all listed
        reference targets.
    min_reference_genes
        Optional minimum number of reference genes required per biological sample.
        This is useful when partial reference coverage should still be allowed,
        but only above a chosen threshold.

    Returns
    -------
    pd.DataFrame
        One row per biological sample with:
        - reference_quantity
        - n_reference_genes
    """
    if not reference_targets:
        raise ValueError("reference_targets must contain at least one target.")

    required = list(id_cols) + [target_col, quantity_col]
    _require_columns(df, required, context="combine_reference_genes input")

    refs = df.loc[df[target_col].isin(reference_targets)].copy()
    if refs.empty:
        raise ValueError(f"No rows found for reference targets: {list(reference_targets)}")

    if method != "geometric_mean":
        raise ValueError("Currently only method='geometric_mean' is supported.")

    grouped = refs.groupby(list(id_cols), dropna=False)

    combined = grouped[quantity_col].agg(_geometric_mean).reset_index()
    combined = combined.rename(columns={quantity_col: "reference_quantity"})

    n_refs = grouped[target_col].nunique().reset_index(name="n_reference_genes")
    combined = combined.merge(n_refs, on=list(id_cols), how="left", validate="one_to_one")

    expected_n_refs = len(set(reference_targets))

    if require_all_reference_targets:
        bad = combined.loc[combined["n_reference_genes"] < expected_n_refs].copy()
        if not bad.empty:
            examples = bad[list(id_cols) + ["n_reference_genes"]].head(10).to_dict("records")
            raise ValueError(
                "Some biological samples are missing one or more reference genes. "
                f"Expected {expected_n_refs} reference genes per sample. "
                f"Examples: {examples}"
            )

    if min_reference_genes is not None:
        if min_reference_genes < 1:
            raise ValueError("min_reference_genes must be >= 1 when provided.")

        bad = combined.loc[combined["n_reference_genes"] < min_reference_genes].copy()
        if not bad.empty:
            examples = bad[list(id_cols) + ["n_reference_genes"]].head(10).to_dict("records")
            raise ValueError(
                "Some biological samples have fewer reference genes than required by "
                f"min_reference_genes={min_reference_genes}. Examples: {examples}"
            )

    return combined


def normalize_to_reference(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    id_cols: Sequence[str] = DEFAULT_ID_COLUMNS,
    quantity_col: str = "relative_quantity",
    reference_col: str = "reference_quantity",
    output_col: str = "normalized_expression",
) -> pd.DataFrame:
    """
    Normalize target relative quantity by combined reference quantity.
    """
    out = df.copy()

    _require_columns(out, list(id_cols) + [quantity_col], context="normalize_to_reference df")
    _require_columns(reference_df, list(id_cols) + [reference_col], context="normalize_to_reference reference_df")

    extra_ref_cols = [col for col in ["n_reference_genes"] if col in reference_df.columns]

    out = out.merge(
        reference_df[list(id_cols) + [reference_col] + extra_ref_cols],
        on=list(id_cols),
        how="left",
        validate="many_to_one",
    )

    if out[reference_col].isna().any():
        raise ValueError(
            "Some rows could not be matched to a combined reference quantity. "
            "Check that each biological sample has all required reference gene rows."
        )

    if (pd.to_numeric(out[reference_col], errors="coerce") <= 0).any():
        raise ValueError("Reference quantities must be positive for normalization.")

    out[output_col] = out[quantity_col] / out[reference_col]
    return out


def add_log2_expression(
    df: pd.DataFrame,
    *,
    expression_col: str = "normalized_expression",
    output_col: str = "log2_normalized_expression",
) -> pd.DataFrame:
    """
    Add log2-transformed normalized expression.

    Values <= 0 are set to NaN because log2 is undefined for non-positive values.
    """
    out = df.copy()
    _require_columns(out, [expression_col], context="add_log2_expression input")

    values = pd.to_numeric(out[expression_col], errors="coerce")
    out[output_col] = np.where(values > 0, np.log2(values), np.nan)
    return out


def calculate_fold_change_vs_control(
    df: pd.DataFrame,
    *,
    control_group: str,
    group_col: str = "group",
    target_col: str = DEFAULT_TARGET_COLUMN,
    normalized_col: str = "normalized_expression",
    output_col: str = "fold_change",
    baseline_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate fold change relative to the mean normalized expression of the control group.

    Parameters
    ----------
    df
        Input dataframe containing normalized expression.
    control_group
        Label identifying the control group in `group_col`.
    group_col
        Column defining treatment/control groups.
    target_col
        Target gene column.
    normalized_col
        Column containing normalized expression values.
    output_col
        Name of the output fold-change column.
    baseline_group_cols
        Columns defining the context within which the control baseline is computed.

        Examples
        --------
        None
            Baseline per target only.

        ["timepoint"]
            Baseline per target and timepoint.

        ["plate_id", "timepoint"]
            Baseline per target, plate, and timepoint.

    Returns
    -------
    pd.DataFrame
        Input dataframe with added:
        - control_mean_expression
        - fold_change
    """
    out = df.copy()
    _require_columns(
        out,
        [group_col, target_col, normalized_col],
        context="calculate_fold_change_vs_control input",
    )

    baseline_group_cols = [] if baseline_group_cols is None else list(baseline_group_cols)
    _require_columns(
        out,
        baseline_group_cols,
        context="calculate_fold_change_vs_control baseline grouping",
    )

    baseline_keys = list(dict.fromkeys([target_col] + baseline_group_cols))

    controls = out.loc[out[group_col] == control_group].copy()
    if controls.empty:
        raise ValueError(f"No rows found for control_group={control_group!r}")

    control_means = (
        controls.groupby(baseline_keys, dropna=False)[normalized_col]
        .mean()
        .reset_index(name="control_mean_expression")
    )

    out = out.merge(
        control_means,
        on=baseline_keys,
        how="left",
        validate="many_to_one",
    )

    if out["control_mean_expression"].isna().any():
        examples = (
            out.loc[out["control_mean_expression"].isna(), baseline_keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "No control-group baseline available for some rows. "
            f"Baseline keys used: {baseline_keys}. Examples: {examples}"
        )

    if (pd.to_numeric(out["control_mean_expression"], errors="coerce") <= 0).any():
        raise ValueError("Control mean expression must be positive for fold-change calculation.")

    out[output_col] = out[normalized_col] / out["control_mean_expression"]
    return out


def calculate_normalized_expression(
    df: pd.DataFrame,
    *,
    reference_targets: Sequence[str],
    control_group: str | None = None,
    efficiency_df: pd.DataFrame | None = None,
    id_cols: Sequence[str] = DEFAULT_ID_COLUMNS,
    target_col: str = DEFAULT_TARGET_COLUMN,
    ct_col: str = DEFAULT_CT_COLUMN,
    reference_aggregation: str = "geometric_mean",
    default_efficiency: float = DEFAULT_EFFICIENCY,
    allow_missing_efficiencies: bool = False,
    warn_on_unusual_efficiencies: bool = True,
    drop_reference_targets_from_output: bool = True,
    add_log2: bool = True,
    include_fold_change: bool = False,
    group_col: str = "group",
    require_all_reference_targets: bool = True,
    min_reference_genes: int | None = None,
    fold_change_baseline_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Full qPCR normalization workflow.

    Steps
    -----
    1. attach efficiencies
    2. calculate relative quantity
    3. combine reference genes
    4. normalize all targets to combined reference
    5. optionally add log2-transformed normalized expression
    6. optionally calculate fold change vs control group

    Parameters
    ----------
    df
        Summarized qPCR dataframe, typically one row per bio_rep + target.
    reference_targets
        One or more reference genes.
    control_group
        Group label to use for fold change. Only required if include_fold_change=True.
    efficiency_df
        Optional dataframe mapping targets to efficiencies.
    id_cols
        Columns defining one biological sample.
    target_col
        Name of the target column.
    ct_col
        Name of the Ct summary column.
    reference_aggregation
        Method used to combine multiple reference genes.
    default_efficiency
        Default amplification factor if no efficiency_df is provided.
    allow_missing_efficiencies
        If True, missing target efficiencies are filled with default_efficiency.
    warn_on_unusual_efficiencies
        If True, emit a warning for efficiency values outside a typical qPCR range.
    drop_reference_targets_from_output
        If True, omit reference-gene rows from the returned dataframe.
    add_log2
        If True, add `log2_normalized_expression`.
    include_fold_change
        If True, calculate fold change relative to `control_group`.
    group_col
        Name of the condition/group column.
    require_all_reference_targets
        If True, require every biological sample to contain all listed reference targets.
    min_reference_genes
        Optional minimum number of reference genes required per biological sample.
    fold_change_baseline_group_cols
        Optional grouping columns defining the context for fold-change baselines.

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with normalized expression as the primary output.
    """
    _require_columns(
        df,
        list(id_cols) + [target_col, ct_col],
        context="calculate_normalized_expression input",
    )
    _validate_unique_measurements(
        df,
        id_cols=id_cols,
        target_col=target_col,
        context="calculate_normalized_expression input",
    )

    out = attach_efficiencies(
        df,
        efficiency_df=efficiency_df,
        target_col=target_col,
        default_efficiency=default_efficiency,
        allow_missing_efficiencies=allow_missing_efficiencies,
        warn_on_unusual_efficiencies=warn_on_unusual_efficiencies,
    )

    out = calculate_relative_quantity(
        out,
        ct_col=ct_col,
        efficiency_col="efficiency",
        output_col="relative_quantity",
    )

    reference_df = combine_reference_genes(
        out,
        reference_targets=reference_targets,
        id_cols=id_cols,
        target_col=target_col,
        quantity_col="relative_quantity",
        method=reference_aggregation,
        require_all_reference_targets=require_all_reference_targets,
        min_reference_genes=min_reference_genes,
    )

    out = normalize_to_reference(
        out,
        reference_df=reference_df,
        id_cols=id_cols,
        quantity_col="relative_quantity",
        reference_col="reference_quantity",
        output_col="normalized_expression",
    )

    if add_log2:
        out = add_log2_expression(
            out,
            expression_col="normalized_expression",
            output_col="log2_normalized_expression",
        )

    if include_fold_change:
        if control_group is None:
            raise ValueError("control_group must be provided when include_fold_change=True.")

        out = calculate_fold_change_vs_control(
            out,
            control_group=control_group,
            group_col=group_col,
            target_col=target_col,
            normalized_col="normalized_expression",
            output_col="fold_change",
            baseline_group_cols=fold_change_baseline_group_cols,
        )

    if drop_reference_targets_from_output:
        out = out.loc[~out[target_col].isin(reference_targets)].copy()

    return out.reset_index(drop=True)


__all__ = [
    "DEFAULT_ID_COLUMNS",
    "DEFAULT_TARGET_COLUMN",
    "DEFAULT_CT_COLUMN",
    "DEFAULT_EFFICIENCY",
    "attach_efficiencies",
    "calculate_relative_quantity",
    "combine_reference_genes",
    "normalize_to_reference",
    "add_log2_expression",
    "calculate_fold_change_vs_control",
    "calculate_normalized_expression",
]