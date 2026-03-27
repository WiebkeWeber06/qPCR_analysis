"""
Quality-control utilities for qPCR data.

Purpose
-------
QC should report potential issues clearly before downstream biological analysis.
Unlike preprocessing, QC does not primarily transform the data. It inspects the
current dataframe and returns summaries / flagged subsets that help you decide
whether the dataset looks trustworthy.

Typical usage
-------------
>>> from qpcr.qc import qc_report
>>> report = qc_report(df, expected_tech_reps=3)
>>> print(report["overview"])
>>> print(report["control_summary"])
>>> print(report["replicate_issues"])
>>> print(report["replicate_variability"])
>>> print(report["missing_metadata"])

The returned object is a dictionary of pandas DataFrames / dictionaries so that
you can inspect them in Python or export them to Excel.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


DEFAULT_REQUIRED_METADATA: tuple[str, ...] = ("group", "sample_id", "bio_rep", "target")
DEFAULT_CONTROL_CONTENTS: tuple[str, ...] = ("NTC", "NRT")
DEFAULT_GROUP_COLS: tuple[str, ...] = ("plate_id", "group", "sample_id", "bio_rep", "target")


def _content_upper(df: pd.DataFrame) -> pd.Series:
    """Return the content column as uppercase strings if present."""
    if "content" not in df.columns:
        return pd.Series(index=df.index, dtype="string")
    return df["content"].astype("string").str.strip().str.upper()


def _control_mask(
    df: pd.DataFrame,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
) -> pd.Series:
    """
    Return a boolean mask identifying control rows.

    Prefers boolean QC columns if present, otherwise falls back to the content column.
    """
    if "is_ntc" in df.columns or "is_nrt" in df.columns:
        is_ntc = (
            df["is_ntc"].fillna(False).astype(bool)
            if "is_ntc" in df.columns
            else pd.Series(False, index=df.index)
        )
        is_nrt = (
            df["is_nrt"].fillna(False).astype(bool)
            if "is_nrt" in df.columns
            else pd.Series(False, index=df.index)
        )
        return is_ntc | is_nrt

    control_set = {c.upper() for c in control_contents}
    return _content_upper(df).isin(control_set)


def _safe_empty_df(columns: Sequence[str]) -> pd.DataFrame:
    """Return an empty dataframe with stable columns."""
    return pd.DataFrame(columns=list(columns))


def find_missing_metadata(
    df: pd.DataFrame,
    required_columns: Sequence[str] = DEFAULT_REQUIRED_METADATA,
    *,
    exclude_controls: bool = True,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
) -> pd.DataFrame:
    """
    Return rows missing any of the required metadata columns.

    Parameters
    ----------
    df
        Input dataframe.
    required_columns
        Metadata columns that must be present for downstream analysis.
    exclude_controls
        If True, exclude control rows such as NTC/NRT from this check.
    control_contents
        Which content values count as controls when boolean control flags
        are unavailable.

    Returns
    -------
    pd.DataFrame
        Subset of rows with one or more missing required metadata values.
        Adds:
        - missing_metadata
        - n_missing_metadata
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required metadata columns in dataframe: {missing_cols}")

    work_df = df.copy()
    if exclude_controls:
        work_df = work_df.loc[~_control_mask(work_df, control_contents=control_contents)].copy()

    if work_df.empty:
        return _safe_empty_df(list(df.columns) + ["missing_metadata", "n_missing_metadata"])

    mask = work_df[list(required_columns)].isna().any(axis=1)
    out = work_df.loc[mask].copy()

    if out.empty:
        return _safe_empty_df(list(work_df.columns) + ["missing_metadata", "n_missing_metadata"])

    missing_matrix = out[list(required_columns)].isna()
    out["missing_metadata"] = missing_matrix.apply(
        lambda row: ", ".join([col for col in required_columns if row[col]]),
        axis=1,
    )
    out["n_missing_metadata"] = missing_matrix.sum(axis=1)

    return out.reset_index(drop=True)


def summarize_controls(
    df: pd.DataFrame,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
    ct_col: str = "ct",
    group_cols: Sequence[str] = ("plate_id", "content"),
) -> pd.DataFrame:
    """
    Summarize control wells such as NTC and NRT.

    Parameters
    ----------
    df
        Input dataframe.
    control_contents
        Which `content` values count as controls.
    ct_col
        Name of Ct column.
    group_cols
        Grouping columns for the summary. Useful defaults are plate-level summaries.

    Returns
    -------
    pd.DataFrame
        Summary table per control group with counts and Ct statistics.
    """
    required_output_cols = list(group_cols) + [
        "n_rows",
        "n_with_ct",
        "n_missing_ct",
        "min_ct",
        "max_ct",
        "mean_ct",
    ]

    if ct_col not in df.columns:
        return _safe_empty_df(required_output_cols)

    controls = df.loc[_control_mask(df, control_contents=control_contents)].copy()
    if controls.empty:
        return _safe_empty_df(required_output_cols)

    if "content" in controls.columns:
        controls["content"] = _content_upper(controls)

    usable_group_cols = [col for col in group_cols if col in controls.columns]
    if not usable_group_cols:
        usable_group_cols = ["content"] if "content" in controls.columns else []

    if not usable_group_cols:
        return _safe_empty_df(required_output_cols)

    summary = (
        controls.groupby(usable_group_cols, dropna=False)[ct_col]
        .agg(["size", "count", "min", "max", "mean"])
        .reset_index()
        .rename(
            columns={
                "size": "n_rows",
                "count": "n_with_ct",
                "min": "min_ct",
                "max": "max_ct",
                "mean": "mean_ct",
            }
        )
    )
    summary["n_missing_ct"] = summary["n_rows"] - summary["n_with_ct"]

    ordered = usable_group_cols + [
        "n_rows",
        "n_with_ct",
        "n_missing_ct",
        "min_ct",
        "max_ct",
        "mean_ct",
    ]
    return summary[ordered].reset_index(drop=True)


def flag_suspicious_controls(
    df: pd.DataFrame,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
    ct_col: str = "ct",
    suspicious_ct_threshold: float = 35.0,
) -> pd.DataFrame:
    """
    Return control wells with Ct values, emphasizing earlier-than-expected amplification.

    Parameters
    ----------
    df
        Input dataframe.
    control_contents
        Which content labels count as controls.
    ct_col
        Name of Ct column.
    suspicious_ct_threshold
        Controls with Ct <= this threshold are often more concerning than
        very late amplification.

    Returns
    -------
    pd.DataFrame
        Control rows with non-missing Ct values. Adds `control_flag`:
        - 'suspicious_amplification' if Ct <= threshold
        - 'late_amplification' otherwise
    """
    expected_cols = list(df.columns) + ["control_flag"]

    if ct_col not in df.columns:
        return _safe_empty_df(expected_cols)

    out = df.loc[_control_mask(df, control_contents=control_contents) & df[ct_col].notna()].copy()
    if out.empty:
        return _safe_empty_df(expected_cols)

    out["control_flag"] = out[ct_col].apply(
        lambda x: "suspicious_amplification"
        if x <= suspicious_ct_threshold
        else "late_amplification"
    )
    return out.reset_index(drop=True)


def check_expected_tech_reps(
    df: pd.DataFrame,
    expected_tech_reps: int = 3,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    ct_col: str = "ct",
    *,
    exclude_controls: bool = True,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
) -> pd.DataFrame:
    """
    Check whether groups have the expected number of technical replicates.

    Parameters
    ----------
    df
        Input dataframe.
    expected_tech_reps
        Expected number of technical replicates per group.
    group_cols
        Columns defining one biological sample/target combination.
    ct_col
        Ct column used to count non-missing technical replicates.
    exclude_controls
        If True, exclude control rows such as NTC/NRT.
    control_contents
        Which content values count as controls when boolean control flags
        are unavailable.

    Returns
    -------
    pd.DataFrame
        Group-level table containing only groups where either:
        - total number of rows differs from expected_tech_reps, or
        - number of valid Ct values differs from expected_tech_reps

        Adds:
        - n_rows
        - n_valid_ct
        - expected_tech_reps
        - replicate_issue
    """
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns for replicate QC: {missing}")
    if ct_col not in df.columns:
        raise ValueError(f"Missing Ct column: {ct_col!r}")

    work_df = df.copy()
    if exclude_controls:
        work_df = work_df.loc[~_control_mask(work_df, control_contents=control_contents)].copy()

    if work_df.empty:
        return _safe_empty_df(
            list(group_cols) + ["n_rows", "n_valid_ct", "expected_tech_reps", "replicate_issue"]
        )

    counts = (
        work_df.groupby(list(group_cols), dropna=False)
        .agg(
            n_rows=(ct_col, "size"),
            n_valid_ct=(ct_col, "count"),
        )
        .reset_index()
    )

    issues = counts.loc[
        (counts["n_rows"] != expected_tech_reps)
        | (counts["n_valid_ct"] != expected_tech_reps)
    ].copy()

    if issues.empty:
        return _safe_empty_df(
            list(group_cols) + ["n_rows", "n_valid_ct", "expected_tech_reps", "replicate_issue"]
        )

    def _label_issue(row: pd.Series) -> str:
        if row["n_rows"] < expected_tech_reps:
            return "too_few_rows"
        if row["n_rows"] > expected_tech_reps:
            return "too_many_rows"
        if row["n_valid_ct"] < expected_tech_reps:
            return "missing_ct_in_replicates"
        if row["n_valid_ct"] > expected_tech_reps:
            return "too_many_valid_ct"
        return "unexpected_replicate_structure"

    issues["expected_tech_reps"] = expected_tech_reps
    issues["replicate_issue"] = issues.apply(_label_issue, axis=1)

    ordered = list(group_cols) + ["n_rows", "n_valid_ct", "expected_tech_reps", "replicate_issue"]
    return issues[ordered].reset_index(drop=True)


def flag_ct_threshold_violations(
    df: pd.DataFrame,
    ct_col: str = "ct",
    ct_min: float | None = None,
    ct_max: float | None = 35.0,
) -> pd.DataFrame:
    """
    Flag rows with Ct values outside a chosen range.

    Parameters
    ----------
    df
        Input dataframe.
    ct_col
        Name of Ct column.
    ct_min
        Lower threshold. If None, no lower bound is applied.
    ct_max
        Upper threshold. If None, no upper bound is applied.

    Returns
    -------
    pd.DataFrame
        Rows outside the specified Ct range. Adds `ct_qc_flag`.
    """
    expected_cols = list(df.columns) + ["ct_qc_flag"]

    if ct_col not in df.columns:
        raise ValueError(f"Missing Ct column: {ct_col!r}")

    low_mask = pd.Series(False, index=df.index)
    if ct_min is not None:
        low_mask = df[ct_col].notna() & (df[ct_col] < ct_min)

    high_mask = pd.Series(False, index=df.index)
    if ct_max is not None:
        high_mask = df[ct_col].notna() & (df[ct_col] > ct_max)

    out = df.loc[low_mask | high_mask].copy()
    if out.empty:
        return _safe_empty_df(expected_cols)

    def _flag(val: float) -> str:
        if ct_min is not None and val < ct_min:
            return "ct_below_min"
        return "ct_above_max"

    out["ct_qc_flag"] = out[ct_col].apply(_flag)
    return out.reset_index(drop=True)


def flag_ct_outliers(
    df: pd.DataFrame,
    ct_col: str = "ct",
    ct_min: float | None = None,
    ct_max: float | None = 35.0,
) -> pd.DataFrame:
    """
    Backward-compatible alias for threshold-based Ct QC.

    Note:
        This function is threshold-based, not a statistical outlier detector.
    """
    return flag_ct_threshold_violations(
        df,
        ct_col=ct_col,
        ct_min=ct_min,
        ct_max=ct_max,
    )


def flag_variable_tech_reps(
    df: pd.DataFrame,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    ct_col: str = "ct",
    *,
    exclude_controls: bool = True,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
    max_ct_range: float = 0.5,
    min_valid_reps: int = 2,
) -> pd.DataFrame:
    """
    Flag groups whose technical replicates vary more than expected.

    Parameters
    ----------
    df
        Input dataframe.
    group_cols
        Columns defining one biological sample/target combination.
    ct_col
        Ct column.
    exclude_controls
        If True, exclude control rows such as NTC/NRT.
    control_contents
        Which content values count as controls when boolean control flags
        are unavailable.
    max_ct_range
        Flag groups where max(Ct) - min(Ct) exceeds this threshold.
    min_valid_reps
        Minimum number of valid Ct values required before evaluating variability.

    Returns
    -------
    pd.DataFrame
        Group-level rows flagged for large within-group Ct spread.
        Includes:
        - n_valid_ct
        - ct_min
        - ct_max
        - ct_range
        - max_allowed_ct_range
    """
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns for replicate variability QC: {missing}")
    if ct_col not in df.columns:
        raise ValueError(f"Missing Ct column: {ct_col!r}")

    work_df = df.copy()
    if exclude_controls:
        work_df = work_df.loc[~_control_mask(work_df, control_contents=control_contents)].copy()

    work_df = work_df.loc[work_df[ct_col].notna()].copy()
    if work_df.empty:
        return _safe_empty_df(
            list(group_cols) + ["n_valid_ct", "ct_min", "ct_max", "ct_range", "max_allowed_ct_range"]
        )

    summary = (
        work_df.groupby(list(group_cols), dropna=False)[ct_col]
        .agg(["count", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_valid_ct", "min": "ct_min", "max": "ct_max"})
    )
    summary["ct_range"] = summary["ct_max"] - summary["ct_min"]

    flagged = summary.loc[
        (summary["n_valid_ct"] >= min_valid_reps)
        & (summary["ct_range"] > max_ct_range)
    ].copy()

    if flagged.empty:
        return _safe_empty_df(
            list(group_cols) + ["n_valid_ct", "ct_min", "ct_max", "ct_range", "max_allowed_ct_range"]
        )

    flagged["max_allowed_ct_range"] = max_ct_range
    ordered = list(group_cols) + [
        "n_valid_ct",
        "ct_min",
        "ct_max",
        "ct_range",
        "max_allowed_ct_range",
    ]
    return flagged[ordered].reset_index(drop=True)


def qc_report(
    df: pd.DataFrame,
    *,
    expected_tech_reps: int = 3,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    required_metadata: Sequence[str] = DEFAULT_REQUIRED_METADATA,
    control_contents: Sequence[str] = DEFAULT_CONTROL_CONTENTS,
    suspicious_control_ct_threshold: float = 35.0,
    ct_outlier_min: float | None = None,
    ct_outlier_max: float | None = 35.0,
    replicate_max_ct_range: float = 0.5,
    exclude_controls_from_metadata_qc: bool = True,
    exclude_controls_from_replicate_qc: bool = True,
) -> dict[str, object]:
    """
    Generate a compact QC report from a qPCR dataframe.

    Returns
    -------
    dict
        Keys include:
        - overview
        - warnings
        - missing_metadata
        - replicate_issues
        - replicate_variability
        - control_summary
        - suspicious_controls
        - ct_outliers
    """
    warnings_list: list[str] = []

    control_mask = _control_mask(df, control_contents=control_contents)
    non_control_mask = ~control_mask

    overview = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_controls": int(control_mask.sum()),
        "n_non_controls": int(non_control_mask.sum()),
        "n_missing_metadata_rows": None,
        "n_replicate_issue_groups": None,
        "n_replicate_variability_groups": None,
        "n_suspicious_controls": None,
        "n_ct_outliers": None,
    }

    report: dict[str, object] = {
        "overview": overview,
        "warnings": warnings_list,
    }

    try:
        missing_metadata = find_missing_metadata(
            df,
            required_columns=required_metadata,
            exclude_controls=exclude_controls_from_metadata_qc,
            control_contents=control_contents,
        )
    except Exception as exc:
        warnings_list.append(f"Missing metadata QC skipped: {exc}")
        missing_metadata = _safe_empty_df(list(df.columns) + ["missing_metadata", "n_missing_metadata"])
    report["missing_metadata"] = missing_metadata
    overview["n_missing_metadata_rows"] = int(len(missing_metadata))

    try:
        replicate_issues = check_expected_tech_reps(
            df,
            expected_tech_reps=expected_tech_reps,
            group_cols=group_cols,
            exclude_controls=exclude_controls_from_replicate_qc,
            control_contents=control_contents,
        )
    except Exception as exc:
        warnings_list.append(f"Replicate count QC skipped: {exc}")
        replicate_issues = _safe_empty_df(
            list(group_cols) + ["n_rows", "n_valid_ct", "expected_tech_reps", "replicate_issue"]
        )
    report["replicate_issues"] = replicate_issues
    overview["n_replicate_issue_groups"] = int(len(replicate_issues))

    try:
        replicate_variability = flag_variable_tech_reps(
            df,
            group_cols=group_cols,
            exclude_controls=exclude_controls_from_replicate_qc,
            control_contents=control_contents,
            max_ct_range=replicate_max_ct_range,
        )
    except Exception as exc:
        warnings_list.append(f"Replicate variability QC skipped: {exc}")
        replicate_variability = _safe_empty_df(
            list(group_cols) + ["n_valid_ct", "ct_min", "ct_max", "ct_range", "max_allowed_ct_range"]
        )
    report["replicate_variability"] = replicate_variability
    overview["n_replicate_variability_groups"] = int(len(replicate_variability))

    try:
        control_summary = summarize_controls(
            df,
            control_contents=control_contents,
        )
    except Exception as exc:
        warnings_list.append(f"Control summary QC skipped: {exc}")
        control_summary = _safe_empty_df(
            ["plate_id", "content", "n_rows", "n_with_ct", "n_missing_ct", "min_ct", "max_ct", "mean_ct"]
        )
    report["control_summary"] = control_summary

    try:
        suspicious_controls = flag_suspicious_controls(
            df,
            control_contents=control_contents,
            suspicious_ct_threshold=suspicious_control_ct_threshold,
        )
    except Exception as exc:
        warnings_list.append(f"Suspicious control QC skipped: {exc}")
        suspicious_controls = _safe_empty_df(list(df.columns) + ["control_flag"])
    report["suspicious_controls"] = suspicious_controls
    overview["n_suspicious_controls"] = int(len(suspicious_controls))

    try:
        ct_outliers = flag_ct_threshold_violations(
            df,
            ct_min=ct_outlier_min,
            ct_max=ct_outlier_max,
        )
    except Exception as exc:
        warnings_list.append(f"Ct threshold QC skipped: {exc}")
        ct_outliers = _safe_empty_df(list(df.columns) + ["ct_qc_flag"])
    report["ct_outliers"] = ct_outliers
    overview["n_ct_outliers"] = int(len(ct_outliers))

    return report


__all__ = [
    "DEFAULT_REQUIRED_METADATA",
    "DEFAULT_CONTROL_CONTENTS",
    "DEFAULT_GROUP_COLS",
    "find_missing_metadata",
    "summarize_controls",
    "flag_suspicious_controls",
    "check_expected_tech_reps",
    "flag_ct_threshold_violations",
    "flag_ct_outliers",
    "flag_variable_tech_reps",
    "qc_report",
]
