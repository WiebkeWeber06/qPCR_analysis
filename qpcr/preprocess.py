"""
Preprocessing utilities for qPCR data.

This module sits between:
- import / schema validation
- downstream analysis such as ΔCt / ΔΔCt

Main goals
----------
- clean Ct values
- optionally keep or exclude controls and calibrators
- summarize technical replicates in a transparent way

Typical usage
-------------
>>> from qpcr.io import load_biorad_csv
>>> from qpcr.schema import validate_qpcr_dataframe
>>> from qpcr.preprocess import filter_invalid_ct, summarize_technical_replicates
>>>
>>> df = load_biorad_csv("plate1.csv", plate_id="plate_1")
>>> df = validate_qpcr_dataframe(df)
>>> df = filter_invalid_ct(df, drop_missing_ct=True)
>>> summary = summarize_technical_replicates(df)
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


DEFAULT_CONTROL_CONTENTS: tuple[str, ...] = ("NTC", "NRT")
DEFAULT_GROUP_COLUMNS: tuple[str, ...] = ("plate_id", "group", "sample_id", "bio_rep", "target")

DEFAULT_METADATA_COLUMNS_TO_KEEP: tuple[str, ...] = (
    "content",
    "timepoint",
    "units",
    "dilution",
    "input_quantity",
    "standard_id",
    "is_control",
    "is_ntc",
    "is_nrt",
    "is_unknown",
    "is_standard",
    "is_calibrator",
    "calibrator_id",
    "calibrator_sample",
)

def _content_upper(df: pd.DataFrame) -> pd.Series:
    """Return the content column as uppercase strings if present."""
    if "content" not in df.columns:
        return pd.Series(index=df.index, dtype="string")
    return df["content"].astype("string").str.strip().str.upper()


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _safe_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean series for a column, defaulting to False when absent."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def _first_non_null(series: pd.Series):
    """Return the first non-missing value in a series, otherwise NA."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return pd.NA
    return non_null.iloc[0]


def _metadata_is_consistent(series: pd.Series) -> bool:
    """Return True if all non-missing values in a series are identical."""
    non_null = series.dropna().unique()
    return len(non_null) <= 1


def flag_controls(
    df: pd.DataFrame,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
) -> pd.DataFrame:
    """
    Add / refresh boolean flags that identify control rows.

    Parameters
    ----------
    df
        Input qPCR dataframe.
    control_contents
        Values in the `content` column that should count as controls.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with:
        - is_control
        - is_ntc
        - is_nrt
    """
    out = df.copy()
    content_upper = _content_upper(out)
    control_set = {value.upper() for value in control_contents}

    out["is_control"] = content_upper.isin(control_set)
    out["is_ntc"] = content_upper.eq("NTC")
    out["is_nrt"] = content_upper.eq("NRT")

    return out


def flag_calibrators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add / refresh a boolean flag that identifies calibrator rows.

    Parameters
    ----------
    df
        Input qPCR dataframe.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with a normalized `is_calibrator` column.
    """
    out = df.copy()
    out["is_calibrator"] = _safe_bool_series(out, "is_calibrator")
    return out


def filter_controls(
    df: pd.DataFrame,
    *,
    keep_controls: bool = False,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
) -> pd.DataFrame:
    """
    Keep or exclude control rows such as NTC/NRT.
    """
    out = flag_controls(df, control_contents=control_contents)

    if keep_controls:
        return out.reset_index(drop=True)

    return out.loc[~out["is_control"]].reset_index(drop=True)


def separate_controls(
    df: pd.DataFrame,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into experimental rows and control rows.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (experimental_df, controls_df)
    """
    out = flag_controls(df, control_contents=control_contents)
    controls = out.loc[out["is_control"]].copy().reset_index(drop=True)
    experimental = out.loc[~out["is_control"]].copy().reset_index(drop=True)
    return experimental, controls


def separate_calibrators(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into non-calibrator rows and calibrator rows.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (non_calibrators_df, calibrators_df)
    """
    out = flag_calibrators(df)
    non_calibrators = out.loc[~out["is_calibrator"]].copy().reset_index(drop=True)
    calibrators = out.loc[out["is_calibrator"]].copy().reset_index(drop=True)
    return non_calibrators, calibrators


def filter_invalid_ct(
    df: pd.DataFrame,
    *,
    drop_missing_ct: bool = True,
    ct_min: float | None = None,
    ct_max: float | None = None,
    keep_controls: bool = True,
    keep_calibrators: bool = True,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
    ct_col: str = "ct",
) -> pd.DataFrame:
    """
    Filter rows based on Ct validity.

    Parameters
    ----------
    df
        Input qPCR dataframe.
    drop_missing_ct
        If True, remove rows where Ct is missing.
    ct_min
        Optional lower Ct threshold. Rows with Ct < ct_min are removed.
    ct_max
        Optional upper Ct threshold. Rows with Ct > ct_max are removed.
    keep_controls
        Whether to keep control rows such as NTC/NRT.
    keep_calibrators
        Whether to keep calibrator rows.
    control_contents
        Values in `content` that should be treated as controls.
    ct_col
        Name of the Ct column.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Notes
    -----
    Missing Ct values are common in controls and non-amplifying wells.
    Whether to remove them depends on the downstream step.
    """
    _require_columns(df, [ct_col], context="filter_invalid_ct")

    out = flag_controls(df, control_contents=control_contents)
    out = flag_calibrators(out)

    if drop_missing_ct:
        out = out.loc[out[ct_col].notna()].copy()

    if ct_min is not None:
        out = out.loc[out[ct_col] >= ct_min].copy()

    if ct_max is not None:
        out = out.loc[out[ct_col] <= ct_max].copy()

    if not keep_controls:
        out = out.loc[~out["is_control"]].copy()

    if not keep_calibrators:
        out = out.loc[~out["is_calibrator"]].copy()

    return out.reset_index(drop=True)


def summarize_technical_replicates(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    ct_col: str = "ct",
    replicate_col: str = "tech_rep",
    keep_non_numeric_metadata: bool = True,
    validate_metadata_consistency: bool = True,
    exclude_controls: bool = False,
    exclude_calibrators: bool = False,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
    metadata_cols_to_keep: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Summarize technical replicates by Ct mean, standard deviation, and counts.

    Parameters
    ----------
    df
        Input qPCR dataframe.
    group_cols
        Columns defining one biological measurement whose technical replicates
        should be summarized.
    ct_col
        Name of the Ct column. This can be raw Ct (`ct`) or calibrated Ct
        (`ct_calibrated`) depending on the stage of analysis.
    replicate_col
        Name of the technical replicate column. Included for API consistency.
    keep_non_numeric_metadata
        If True, preserve selected metadata columns by taking the first
        non-missing value within each group.
    validate_metadata_consistency
        If True, raise an error when whitelisted metadata are inconsistent
        within a summarized group.
    exclude_controls
        If True, remove control rows before summarization.
    exclude_calibrators
        If True, remove calibrator rows before summarization.
    control_contents
        Values in `content` that should be treated as controls.
    metadata_cols_to_keep
        Explicit whitelist of metadata columns to carry forward after
        summarization. If None, DEFAULT_METADATA_COLUMNS_TO_KEEP is used.

    Returns
    -------
    pd.DataFrame
        One row per summarized technical-replicate group.

    Output columns
    --------------
    - all grouping columns
    - ct_mean
    - ct_std
    - n_rows
    - n_valid_ct
    - plus selected metadata columns

    Notes
    -----
    `n_rows` counts all rows in the group.
    `n_valid_ct` counts only non-missing Ct values.
    Only whitelisted metadata columns are eligible to be carried forward.
    """
    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing grouping columns: {missing_group_cols}")

    _require_columns(df, [ct_col], context="summarize_technical_replicates")

    out = flag_controls(df, control_contents=control_contents)
    out = flag_calibrators(out)

    if exclude_controls:
        out = out.loc[~out["is_control"]].copy()

    if exclude_calibrators:
        out = out.loc[~out["is_calibrator"]].copy()

    summary = (
        out.groupby(list(group_cols), dropna=False)
        .agg(
            ct_mean=(ct_col, "mean"),
            ct_std=(ct_col, "std"),
            n_rows=(ct_col, "size"),
            n_valid_ct=(ct_col, "count"),
        )
        .reset_index()
    )

    if keep_non_numeric_metadata:
        whitelist = (
            DEFAULT_METADATA_COLUMNS_TO_KEEP
            if metadata_cols_to_keep is None
            else tuple(metadata_cols_to_keep)
        )

        metadata_cols = [
            col for col in whitelist
            if col in out.columns and col not in group_cols and col != ct_col and col != replicate_col
        ]

        if metadata_cols and validate_metadata_consistency:
            inconsistent_records: list[pd.DataFrame] = []

            for col in metadata_cols:
                consistency = (
                    out.groupby(list(group_cols), dropna=False)[col]
                    .apply(_metadata_is_consistent)
                    .reset_index(name="is_consistent")
                )
                bad = consistency.loc[~consistency["is_consistent"]].copy()
                if not bad.empty:
                    bad["metadata_column"] = col
                    inconsistent_records.append(bad)

            if inconsistent_records:
                inconsistent_df = pd.concat(inconsistent_records, ignore_index=True)
                examples = inconsistent_df.head(10).to_dict("records")
                raise ValueError(
                    "Inconsistent metadata detected within technical replicate groups "
                    f"for whitelisted metadata columns. Examples: {examples}"
                )

        if metadata_cols:
            metadata = (
                out.groupby(list(group_cols), dropna=False)[metadata_cols]
                .agg(_first_non_null)
                .reset_index()
            )

            summary = summary.merge(
                metadata,
                on=list(group_cols),
                how="left",
                validate="one_to_one",
            )

    ordered_front = list(group_cols) + ["ct_mean", "ct_std", "n_rows", "n_valid_ct"]
    remaining = [col for col in summary.columns if col not in ordered_front]
    summary = summary[ordered_front + remaining]

    return summary.reset_index(drop=True)


def summarize_preprocessing(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> dict[str, object]:
    """
    Return a compact summary of what preprocessing changed.
    """
    summary: dict[str, object] = {
        "n_rows_original": len(original_df),
        "n_rows_processed": len(processed_df),
        "n_rows_removed": len(original_df) - len(processed_df),
    }

    for prefix, df in [("original", original_df), ("processed", processed_df)]:
        if "content" in df.columns:
            summary[f"{prefix}_content_counts"] = df["content"].value_counts(dropna=False).to_dict()
        if "ct" in df.columns:
            summary[f"{prefix}_missing_ct"] = int(df["ct"].isna().sum())
        if "ct_calibrated" in df.columns:
            summary[f"{prefix}_missing_ct_calibrated"] = int(df["ct_calibrated"].isna().sum())
        if "is_control" in df.columns:
            summary[f"{prefix}_n_controls"] = int(df["is_control"].fillna(False).astype(bool).sum())
        if "is_calibrator" in df.columns:
            summary[f"{prefix}_n_calibrators"] = int(df["is_calibrator"].fillna(False).astype(bool).sum())

    return summary


__all__ = [
    "DEFAULT_CONTROL_CONTENTS",
    "DEFAULT_GROUP_COLUMNS",
    "flag_controls",
    "flag_calibrators",
    "filter_controls",
    "filter_invalid_ct",
    "separate_controls",
    "separate_calibrators",
    "summarize_technical_replicates",
    "summarize_preprocessing",
]