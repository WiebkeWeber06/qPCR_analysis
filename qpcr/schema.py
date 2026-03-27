"""
Schema validation for qPCR dataframes.

This module defines:
- required and optional columns
- validation of core data types
- validation of core categorical values
- light sanity checks to catch obvious import problems early

Typical usage
-------------
>>> from qpcr.io import load_biorad_csv
>>> from qpcr.schema import validate_qpcr_dataframe
>>>
>>> df = load_biorad_csv("plate1.csv", plate_id="plate_1")
>>> validate_qpcr_dataframe(df)
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "well",
    "target",
    "content",
    "sample_id",
    "ct",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "plate_id",
    "fluor",
    "group",
    "timepoint",
    "bio_rep",
    "tech_rep",
    "well_note",
    "machine_ct_mean",
    "machine_ct_std_dev",
    "machine_starting_quantity",
    "machine_sq_std_dev",
    "melt_temperature",
    "peak_height",
    "begin_temperature",
    "end_temperature",
    "is_standard",
    "standard_id",
    "dilution",
    "input_quantity",
    "units",
    "is_calibrator",
    "calibrator_id",
    "calibrator_sample",
    "is_ntc",
    "is_nrt",
    "is_unknown",
    "row",
    "column",
)

STANDARD_CURVE_COLUMNS: tuple[str, ...] = (
    "is_standard",
    "standard_id",
    "dilution",
    "input_quantity",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "ct",
    "machine_ct_mean",
    "machine_ct_std_dev",
    "machine_starting_quantity",
    "machine_sq_std_dev",
    "melt_temperature",
    "peak_height",
    "begin_temperature",
    "end_temperature",
    "dilution",
    "input_quantity",
    "tech_rep",
    "bio_rep",
    "column",
)

BOOLEAN_COLUMNS: tuple[str, ...] = (
    "is_standard",
    "is_ntc",
    "is_nrt",
    "is_unknown",
    "is_calibrator",
)

ALLOWED_CONTENT_VALUES: tuple[str, ...] = (
    "UNKN",
    "NTC",
    "NRT",
    "STD",
    "STANDARD",
)

CONTROL_CONTENT_VALUES: tuple[str, ...] = (
    "NTC",
    "NRT",
)

DUPLICATE_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("plate_id", "well", "target"),
    ("plate_id", "well"),
    ("well", "target"),
    ("well",),
)


def _as_upper_string(series: pd.Series) -> pd.Series:
    """Convert a series to stripped uppercase pandas strings."""
    return series.astype("string").str.strip().str.upper()


def _find_missing_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> list[str]:
    """Return required columns that are missing from the dataframe."""
    return [col for col in required_columns if col not in df.columns]


def _warn(message: str) -> None:
    """Emit a standard user warning."""
    warnings.warn(message, UserWarning, stacklevel=2)


def check_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    """
    Raise an error if required columns are missing.
    """
    missing = _find_missing_columns(df, required_columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Present columns are: {list(df.columns)}"
        )


def check_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str] = NUMERIC_COLUMNS,
) -> None:
    """
    Validate that selected columns are numeric when present.

    Missing values are allowed. The important requirement is that the column
    dtype itself is numeric once imported/standardized.
    """
    for col in columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column {col!r} must be numeric.")


def check_boolean_columns(
    df: pd.DataFrame,
    columns: Iterable[str] = BOOLEAN_COLUMNS,
) -> None:
    """
    Validate that selected columns contain boolean-like values when present.

    Accepts actual booleans and pandas nullable BooleanDtype columns.
    """
    for col in columns:
        if col not in df.columns:
            continue

        values = df[col].dropna()
        if values.empty:
            continue

        invalid = values.loc[~values.isin([True, False])]
        if not invalid.empty:
            examples = invalid.astype("string").unique().tolist()[:10]
            raise TypeError(
                f"Column {col!r} must contain boolean values. "
                f"Unexpected examples: {examples}"
            )


def check_ct_column(df: pd.DataFrame) -> None:
    """
    Validate that the Ct column exists and is numeric.

    Missing Ct values are allowed because NTC/NRT/empty measurements may not
    have amplification. The important thing is that the column dtype is numeric.
    """
    if "ct" not in df.columns:
        raise ValueError("Missing required column: 'ct'")

    if not pd.api.types.is_numeric_dtype(df["ct"]):
        raise TypeError("Column 'ct' must be numeric.")


def check_required_columns_not_all_missing(df: pd.DataFrame) -> None:
    """
    Catch dataframes that technically contain required columns but no usable data.

    Some columns are treated as critical and some as suspicious:
    - 'well' and 'content' entirely missing are considered fatal
    - 'target' and 'sample_id' entirely missing are suspicious and trigger warnings
    - 'ct' entirely missing is allowed only for pure control datasets (NTC/NRT)
    """
    critical_cols = ("well", "content")
    suspicious_cols = ("target", "sample_id")

    for col in critical_cols:
        if col in df.columns and df[col].isna().all():
            raise ValueError(f"Column {col!r} is present but entirely empty.")

    for col in suspicious_cols:
        if col in df.columns and df[col].isna().all():
            _warn(
                f"Column {col!r} is present but entirely empty. "
                "This may be valid for some inputs, but it can also indicate "
                "an import or merge problem."
            )

    if "ct" in df.columns and df["ct"].isna().all():
        if "content" not in df.columns:
            raise ValueError(
                "Column 'ct' is entirely missing and 'content' is unavailable, "
                "so the dataframe cannot be interpreted safely."
            )

        content_upper = _as_upper_string(df["content"]).dropna()
        only_controls = set(content_upper.unique()).issubset(set(CONTROL_CONTENT_VALUES))

        if not only_controls:
            raise ValueError(
                "Column 'ct' is entirely missing. This may indicate an import problem "
                "or a file exported without Ct/Cq values."
            )


def check_well_format(df: pd.DataFrame, strict: bool = False) -> None:
    """
    Check whether well identifiers look plausible.

    Accepts values like:
    - A1
    - A01
    - B12
    - H09

    In non-strict mode unusual values trigger a warning.
    In strict mode they raise a ValueError.
    """
    if "well" not in df.columns:
        return

    wells = df["well"].dropna().astype("string").str.strip().str.upper()
    if wells.empty:
        return

    valid = wells.str.match(r"^[A-Z]+0?\d{1,2}$")
    invalid_values = wells.loc[~valid].unique().tolist()

    if invalid_values:
        message = (
            f"Some well identifiers look unusual: {invalid_values[:10]}. "
            "This may be fine, but it can also indicate import issues."
        )
        if strict:
            raise ValueError(message)
        _warn(message)


def check_content_values(df: pd.DataFrame, strict: bool = False) -> None:
    """
    Check whether `content` contains expected qPCR categories.

    This is intentionally permissive because different machines or export
    settings may use additional labels.
    """
    if "content" not in df.columns:
        return

    values = _as_upper_string(df["content"]).dropna()
    unexpected = sorted(set(values.unique()) - set(ALLOWED_CONTENT_VALUES))

    if unexpected:
        message = (
            f"Unexpected values found in 'content': {unexpected}. "
            f"Allowed defaults are: {list(ALLOWED_CONTENT_VALUES)}"
        )
        if strict:
            raise ValueError(message)
        _warn(message)


def check_standard_curve_columns(df: pd.DataFrame) -> None:
    """
    Validate standard-curve metadata if any rows are marked as standard.

    Rules:
    - if `is_standard` exists and any row is True, standard rows must have at
      least one of `dilution` or `input_quantity`
    - missing `standard_id` is not fatal, but produces a warning
    """
    if "is_standard" not in df.columns:
        return

    is_standard = df["is_standard"].fillna(False).astype(bool)
    if not is_standard.any():
        return

    standard_df = df.loc[is_standard].copy()

    has_dilution = "dilution" in standard_df.columns
    has_input_quantity = "input_quantity" in standard_df.columns

    if not (has_dilution or has_input_quantity):
        raise ValueError(
            "Rows are marked as standard curves (`is_standard=True`), but neither "
            "'dilution' nor 'input_quantity' is present."
        )

    dilution_missing = (
        standard_df["dilution"].isna() if has_dilution
        else pd.Series(True, index=standard_df.index)
    )
    input_quantity_missing = (
        standard_df["input_quantity"].isna() if has_input_quantity
        else pd.Series(True, index=standard_df.index)
    )

    missing_both = dilution_missing & input_quantity_missing
    if missing_both.any():
        bad_rows = standard_df.loc[missing_both]
        example_wells = bad_rows["well"].dropna().astype("string").tolist()[:10] if "well" in bad_rows.columns else []
        raise ValueError(
            "Some standard-curve rows are missing both 'dilution' and "
            f"'input_quantity'. Example wells: {example_wells}"
        )

    if "standard_id" in standard_df.columns:
        missing_standard_id = standard_df["standard_id"].isna()
        if missing_standard_id.any():
            bad_rows = standard_df.loc[missing_standard_id]
            example_wells = bad_rows["well"].dropna().astype("string").tolist()[:10] if "well" in bad_rows.columns else []
            _warn(
                "Some standard-curve rows are missing 'standard_id'. "
                f"Example wells: {example_wells}"
            )


def check_duplicate_keys(
    df: pd.DataFrame,
    candidate_keys: Sequence[Sequence[str]] = DUPLICATE_KEY_CANDIDATES,
    strict: bool = False,
) -> None:
    """
    Check for duplicated logical identifiers.

    The first candidate key whose columns all exist in the dataframe is used.
    In non-strict mode duplicates trigger a warning.
    In strict mode duplicates raise a ValueError.
    """
    selected_keys: list[str] | None = None

    for keys in candidate_keys:
        if all(col in df.columns for col in keys):
            selected_keys = list(keys)
            break

    if not selected_keys:
        return

    subset = df[selected_keys].copy()
    subset = subset.dropna(how="any")
    if subset.empty:
        return

    duplicates = subset.duplicated(subset=selected_keys, keep=False)
    if not duplicates.any():
        return

    examples = subset.loc[duplicates, selected_keys].head(10).to_dict("records")
    message = (
        f"Found duplicated rows for key columns {selected_keys}. "
        f"Examples: {examples}"
    )

    if strict:
        raise ValueError(message)
    _warn(message)


def summarize_schema(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a compact summary of the dataframe structure.
    """
    summary: dict[str, object] = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_required_columns": _find_missing_columns(df, REQUIRED_COLUMNS),
    }

    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            summary[f"{col}_missing_values"] = int(df[col].isna().sum())

    if "well" in df.columns:
        summary["n_unique_wells"] = int(df["well"].nunique(dropna=True))

    if "content" in df.columns:
        summary["content_counts"] = df["content"].value_counts(dropna=False).to_dict()
        unexpected = sorted(
            set(_as_upper_string(df["content"]).dropna().unique()) - set(ALLOWED_CONTENT_VALUES)
        )
        summary["unexpected_content_values"] = unexpected

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            summary[f"{col}_counts"] = df[col].value_counts(dropna=False).to_dict()

    return summary


def validate_qpcr_dataframe(
    df: pd.DataFrame,
    *,
    strict_well_format: bool = False,
    strict_content_values: bool = False,
    strict_duplicate_keys: bool = False,
) -> pd.DataFrame:
    """
    Validate a qPCR dataframe against the expected schema.

    Parameters
    ----------
    df
        Dataframe to validate.
    strict_well_format
        If True, unusual well names raise an error. Otherwise they trigger a warning.
    strict_content_values
        If True, unexpected content labels raise an error. Otherwise they trigger a warning.
    strict_duplicate_keys
        If True, duplicated logical identifiers raise an error. Otherwise they trigger a warning.

    Returns
    -------
    pd.DataFrame
        The original dataframe, unchanged. Returning it allows a convenient style:
        df = validate_qpcr_dataframe(df)

    Raises
    ------
    ValueError or TypeError
        If critical schema problems are found.
    """
    check_required_columns(df)
    check_ct_column(df)
    check_numeric_columns(df)
    check_boolean_columns(df)
    check_required_columns_not_all_missing(df)
    check_well_format(df, strict=strict_well_format)
    check_content_values(df, strict=strict_content_values)
    check_standard_curve_columns(df)
    check_duplicate_keys(df, strict=strict_duplicate_keys)

    return df


__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "STANDARD_CURVE_COLUMNS",
    "NUMERIC_COLUMNS",
    "BOOLEAN_COLUMNS",
    "ALLOWED_CONTENT_VALUES",
    "check_required_columns",
    "check_numeric_columns",
    "check_boolean_columns",
    "check_ct_column",
    "check_required_columns_not_all_missing",
    "check_well_format",
    "check_content_values",
    "check_standard_curve_columns",
    "check_duplicate_keys",
    "summarize_schema",
    "validate_qpcr_dataframe",
]
