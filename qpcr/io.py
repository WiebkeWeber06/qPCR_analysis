"""
Input/output utilities for qPCR data.

This module supports two main input sources:
1. Bio-Rad machine export files
2. Human-friendly plate setup files that are transformed into the internal schema

Typical usage
-------------
>>> from qpcr.io import load_biorad_csv, read_plate_setup, merge_plate_design
>>> data_df = load_biorad_csv("plate1.csv", plate_id="plate_1")
>>> design_df = read_plate_setup("plate_setup.csv", plate_id="plate_1")
>>> df = merge_plate_design(data_df, design_df, on=("plate_id", "well"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

PathLike = str | Path

DEFAULT_COLUMN_MAP: dict[str, str] = {
    "Well": "well",
    "Fluor": "fluor",
    "Target": "target",
    "Content": "content",
    "Replicate": "tech_rep",
    "Sample": "sample_id",
    "Biological Set Name": "group",
    "Well Note": "well_note",
    "Cq": "ct",
    "Starting Quantity (SQ)": "machine_starting_quantity",
    "Cq Mean": "machine_ct_mean",
    "Cq Std. Dev": "machine_ct_std_dev",
    "SQ Std. Dev": "machine_sq_std_dev",
    "Melt Temperature": "melt_temperature",
    "Peak Height": "peak_height",
    "Begin Temperature": "begin_temperature",
    "End Temperature": "end_temperature",
}

PLATE_SETUP_COLUMN_MAP: dict[str, str] = {
    "Row": "row",
    "Column": "column",
    "Sample Type": "content",
    "*Sample Type": "content",
    "Replicate #": "tech_rep",
    "*Replicate #": "tech_rep",
    "tech_rep": "tech_rep",
    "Target Name": "target",
    "*Target Name": "target",
    "target": "target",
    "Sample Name": "sample_id",
    "*Sample Name": "sample_id",
    "sample_id": "sample_id",
    "Biological Group": "group",
    "*Biological Group": "group",
    "group": "group",
    "Well Note": "well_note",
    "*Well Note": "well_note",
    "Starting Quantity": "input_quantity",
    "*Starting Quantity": "input_quantity",
    "Units": "units",
    "bio_rep": "bio_rep",
    "dilution": "dilution",
    "standard_id": "standard_id",
    "timepoint": "timepoint",
    "calibrator_id": "calibrator_id",
    "Calibrator ID": "calibrator_id",
    "calibrator_sample": "calibrator_sample",
    "Calibrator Sample": "calibrator_sample",
    "is_calibrator": "is_calibrator",
    "Is Calibrator": "is_calibrator",
}

NUMERIC_COLUMNS: tuple[str, ...] = (
    "ct",
    "machine_starting_quantity",
    "machine_ct_mean",
    "machine_ct_std_dev",
    "machine_sq_std_dev",
    "melt_temperature",
    "peak_height",
    "begin_temperature",
    "end_temperature",
    "dilution",
    "input_quantity",
    "tech_rep",
    "bio_rep",
)

TEXT_COLUMNS: tuple[str, ...] = (
    "well",
    "fluor",
    "target",
    "content",
    "sample_id",
    "group",
    "well_note",
    "standard_id",
    "row",
    "units",
    "timepoint",
    "calibrator_id",
    "calibrator_sample",
)

NA_TEXT_VALUES: dict[str, object] = {
    "": pd.NA,
    "None": pd.NA,
    "none": pd.NA,
    "NaN": pd.NA,
    "nan": pd.NA,
    "N/A": pd.NA,
    "n/a": pd.NA,
}

BIORAD_COLUMN_ORDER: list[str] = [
    "plate_id",
    "well",
    "fluor",
    "target",
    "content",
    "sample_id",
    "group",
    "timepoint",
    "bio_rep",
    "tech_rep",
    "ct",
    "machine_ct_mean",
    "machine_ct_std_dev",
    "machine_starting_quantity",
    "melt_temperature",
    "peak_height",
    "is_standard",
    "standard_id",
    "dilution",
    "input_quantity",
    "is_calibrator",
    "calibrator_id",
    "calibrator_sample",
    "is_ntc",
    "is_nrt",
    "is_unknown",
    "well_note",
]

PLATE_SETUP_COLUMN_ORDER: list[str] = [
    "plate_id",
    "row",
    "column",
    "well",
    "content",
    "target",
    "sample_id",
    "group",
    "timepoint",
    "bio_rep",
    "tech_rep",
    "dilution",
    "input_quantity",
    "units",
    "is_standard",
    "standard_id",
    "is_calibrator",
    "calibrator_id",
    "calibrator_sample",
    "is_ntc",
    "is_nrt",
    "is_unknown",
    "well_note",
]


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    """Coerce common text/numeric boolean representations to pandas BooleanDtype."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "y": True,
                "n": False,
            }
        )
        .astype("boolean")
    )


def _find_header_row(filepath: PathLike, header_token: str = "Well") -> int:
    """Find the row index containing the measurement table header."""
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for idx, line in enumerate(handle):
            fields = [field.strip() for field in line.split(",")]
            if header_token in fields:
                return idx

    raise ValueError(
        f"Could not find a measurement table header containing {header_token!r} "
        f"in file: {filepath}"
    )


def _normalize_text_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace(NA_TEXT_VALUES)
            )
    return df


def _coerce_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_well_series(series: pd.Series) -> pd.Series:
    """
    Normalize wells to a consistent format like A01, B12, etc.
    Accepts inputs such as A1, A01, a1, ' A 1 '.
    """
    s = series.astype("string").str.strip().str.upper()
    extracted = s.str.extract(r"^([A-Z]+)\s*0*([0-9]+)$")
    row = extracted[0]
    col = pd.to_numeric(extracted[1], errors="coerce")
    col_str = col.astype("Int64").astype("string").str.zfill(2)
    return (row + col_str).where(row.notna() & col.notna(), pd.NA)


def _add_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    content_upper = (
        df.get("content", pd.Series(index=df.index, dtype="string"))
        .astype("string")
        .str.upper()
    )

    df["is_ntc"] = content_upper.eq("NTC")
    df["is_nrt"] = content_upper.eq("NRT")
    df["is_unknown"] = content_upper.eq("UNKN")
    df["is_standard"] = content_upper.isin(["STD", "STANDARD"])

    if "is_calibrator" in df.columns:
        coerced = _coerce_bool_column(df["is_calibrator"])
        df["is_calibrator"] = coerced.fillna(False).astype(bool)
    else:
        df["is_calibrator"] = False

    return df


def _make_default_plate_id(filepath: PathLike) -> str:
    return Path(filepath).stem


def _make_well_from_row_column(row: pd.Series, column: pd.Series) -> pd.Series:
    row = row.astype("string").str.strip().str.upper()
    col_num = pd.to_numeric(column, errors="coerce")
    col_str = col_num.astype("Int64").astype("string").str.zfill(2)
    well = row + col_str
    return well.where(row.notna() & col_num.notna(), pd.NA)


def _read_tabular_file(filepath: PathLike, **read_kwargs) -> pd.DataFrame:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(filepath, **read_kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(filepath, **read_kwargs)

    raise ValueError(f"Unsupported file type: {suffix}")


def _reorder_columns(df: pd.DataFrame, preferred_order: Sequence[str]) -> pd.DataFrame:
    existing_first = [col for col in preferred_order if col in df.columns]
    remaining = [col for col in df.columns if col not in existing_first]
    return df[existing_first + remaining]


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {context}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _standardize_qpcr_dataframe(
    df: pd.DataFrame,
    *,
    text_columns: Sequence[str] = TEXT_COLUMNS,
    numeric_columns: Sequence[str] = NUMERIC_COLUMNS,
) -> pd.DataFrame:
    """Apply common qPCR dataframe cleanup steps."""
    df = _normalize_text_columns(df, text_columns)
    df = _coerce_numeric_columns(df, numeric_columns)

    if "well" in df.columns:
        df["well"] = _normalize_well_series(df["well"])

    return _add_qc_flags(df)


def standardize_columns(
    df: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    column_map = DEFAULT_COLUMN_MAP if column_map is None else dict(column_map)
    return df.rename(columns=column_map)


def load_biorad_csv(
    filepath: PathLike,
    plate_id: str | None = None,
    column_map: Mapping[str, str] | None = None,
    drop_empty_wells: bool = True,
) -> pd.DataFrame:
    """
    Load a Bio-Rad export CSV and transform it into the internal schema.
    """
    filepath = Path(filepath)
    header_row = _find_header_row(filepath)

    df = pd.read_csv(filepath, skiprows=header_row, encoding="utf-8-sig")
    df = standardize_columns(df, column_map=column_map)
    _require_columns(df, ["well"], context=f"Bio-Rad file {filepath}")

    df = _standardize_qpcr_dataframe(df)
    df["plate_id"] = plate_id if plate_id is not None else _make_default_plate_id(filepath)

    if drop_empty_wells:
        has_target = df["target"].notna() if "target" in df.columns else pd.Series(False, index=df.index)
        has_sample = df["sample_id"].notna() if "sample_id" in df.columns else pd.Series(False, index=df.index)
        has_ct = df["ct"].notna() if "ct" in df.columns else pd.Series(False, index=df.index)
        mask_keep = has_target | has_sample | has_ct
        df = df.loc[mask_keep].copy()

    df = _reorder_columns(df, BIORAD_COLUMN_ORDER)
    return df.reset_index(drop=True)


def load_multiple_plates(
    filepaths: Sequence[PathLike],
    plate_ids: Sequence[str | None] | None = None,
    column_map: Mapping[str, str] | None = None,
    drop_empty_wells: bool = True,
) -> pd.DataFrame:
    """
    Load multiple Bio-Rad export files and concatenate them into one dataframe.
    """
    filepaths = list(filepaths)

    if plate_ids is None:
        plate_ids = [None] * len(filepaths)
    else:
        plate_ids = list(plate_ids)
        if len(plate_ids) != len(filepaths):
            raise ValueError("plate_ids must have the same length as filepaths.")

    frames = [
        load_biorad_csv(
            filepath=fp,
            plate_id=pid,
            column_map=column_map,
            drop_empty_wells=drop_empty_wells,
        )
        for fp, pid in zip(filepaths, plate_ids)
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def read_plate_setup(
    filepath: PathLike,
    plate_id: str | None = None,
    column_map: Mapping[str, str] | None = None,
    drop_empty_wells: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Read a human-friendly plate setup file and transform it into the internal schema.
    """
    filepath = Path(filepath)
    raw = _read_tabular_file(filepath, **read_kwargs)

    effective_map = dict(PLATE_SETUP_COLUMN_MAP)
    if column_map is not None:
        effective_map.update(dict(column_map))

    df = raw.rename(columns=effective_map)
    df = _normalize_text_columns(df, TEXT_COLUMNS)
    df = _coerce_numeric_columns(df, NUMERIC_COLUMNS)

    has_well = "well" in df.columns
    has_row_and_column = {"row", "column"}.issubset(df.columns)

    if not has_well and not has_row_and_column:
        raise ValueError(
            f"Plate setup file {filepath} must contain either 'well' or both "
            f"'row' and 'column' columns after renaming."
        )

    if not has_well and has_row_and_column:
        df["well"] = _make_well_from_row_column(df["row"], df["column"])

    df = _standardize_qpcr_dataframe(df)
    df["plate_id"] = plate_id if plate_id is not None else _make_default_plate_id(filepath)

    if drop_empty_wells:
        useful_cols = [
            col
            for col in ["well", "target", "sample_id", "group", "bio_rep", "tech_rep"]
            if col in df.columns
        ]
        if useful_cols:
            mask_keep = df[useful_cols].notna().any(axis=1)
            df = df.loc[mask_keep].copy()

    df = _reorder_columns(df, PLATE_SETUP_COLUMN_ORDER)
    return df.reset_index(drop=True)


def merge_plate_design(
    data_df: pd.DataFrame,
    plate_design: pd.DataFrame,
    on: str | Sequence[str] = ("plate_id", "well"),
    how: str = "left",
    validate: str = "many_to_one",
) -> pd.DataFrame:
    """
    Merge measurement data with plate design metadata.

    For overlapping metadata columns, values from plate_design are preferred and
    used to fill missing values in data_df.
    """
    if isinstance(on, str):
        on = [on]
    else:
        on = list(on)

    for key in on:
        if key not in data_df.columns:
            raise KeyError(f"Merge key {key!r} is missing from data_df.")
        if key not in plate_design.columns:
            raise KeyError(f"Merge key {key!r} is missing from plate_design.")

    left = data_df.copy()
    right = plate_design.copy()

    if "well" in left.columns:
        left["well"] = _normalize_well_series(left["well"])
    if "well" in right.columns:
        right["well"] = _normalize_well_series(right["well"])

    merged = left.merge(
        right,
        on=on,
        how=how,
        validate=validate,
        suffixes=("", "_design"),
    )

    preferred_design_cols = [
        "content",
        "target",
        "sample_id",
        "group",
        "timepoint",
        "bio_rep",
        "tech_rep",
        "dilution",
        "input_quantity",
        "units",
        "standard_id",
        "is_calibrator",
        "calibrator_id",
        "calibrator_sample",
        "well_note",
        "row",
        "column",
    ]

    for col in preferred_design_cols:
        design_col = f"{col}_design"
        if design_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[design_col].combine_first(merged[col])
                merged = merged.drop(columns=[design_col])
            else:
                merged = merged.rename(columns={design_col: col})

    merged = _add_qc_flags(merged)
    return merged


def read_plate_design_raw(
    filepath: PathLike,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Read a tabular plate design/setup file without transforming it.
    """
    return _read_tabular_file(filepath, **read_kwargs)


__all__ = [
    "DEFAULT_COLUMN_MAP",
    "PLATE_SETUP_COLUMN_MAP",
    "NUMERIC_COLUMNS",
    "TEXT_COLUMNS",
    "load_biorad_csv",
    "load_multiple_plates",
    "read_plate_setup",
    "merge_plate_design",
    "read_plate_design_raw",
    "standardize_columns",
]