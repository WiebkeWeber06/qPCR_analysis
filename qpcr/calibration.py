"""
Calibration module for qPCR multi-plate experiments.

Overview
--------
This module implements target-specific inter-plate calibration (IPC) to correct
for systematic plate-to-plate variation in qPCR experiments.

In multi-plate experiments, identical calibrator samples are included on each
plate for every target gene. Differences in their measured Ct values reflect
technical variation (e.g. pipetting, reagent efficiency, machine variation)
rather than biological differences.

This module estimates plate-specific offsets based on calibrator measurements
and applies these offsets to all samples, aligning Ct values across plates.


Conceptual Model
----------------
Calibration is performed independently for each target gene.

For each (plate, target) combination:

    offset_ct(plate, target) =
        mean_ct(calibrator wells on that plate and target)
        - reference_ct(target)

The reference Ct is defined either as:

    - the global mean across all plates for that target, OR
    - the calibrator Ct from a designated reference plate

Corrected Ct values are then computed as:

    Ct_calibrated = Ct_raw - offset_ct(plate, target)

This ensures that all plates are aligned to a common reference level for each
target gene.


Key Assumptions
---------------
This calibration approach relies on the following assumptions:

1. Target-specific calibrators
   Each target gene has dedicated calibrator wells on every plate.

2. Identical calibrator sample
   The calibrator represents the same biological material across all plates
   (e.g. pooled cDNA).

3. Technical replicates
   Calibrator wells are measured in technical replicates (≥3 recommended).

4. Additive plate effects
   Plate-to-plate variation is assumed to be additive in Ct space.

5. Stable amplification efficiency
   PCR efficiency is assumed to be comparable across plates for a given target.


Required Input Columns
---------------------
Input dataframe must contain:

- plate_id : identifier of the plate
- target   : target gene name
- ct       : raw Ct (Cq) value
- is_calibrator : boolean indicating calibrator wells

Optional but recommended:

- calibrator_id
- calibrator_sample
- tech_rep
- bio_rep
- group


Workflow
--------
Typical workflow:

1. Summarize calibrator measurements

    >>> summary = summarize_calibrators(df)

   Output includes:
   - mean Ct per (plate, target)
   - standard deviation
   - number of replicates

2. Compute calibration offsets

    >>> offsets = calculate_plate_calibrator_offsets(summary)

   Options:
   - reference_method="global_mean"
   - reference_method="reference_plate"

3. Apply calibration to all data

    >>> df_calibrated = apply_plate_calibration(df, offsets)

   Adds:
   - ct_calibrated
   - offset_ct
   - was_calibrated


Interpretation of Outputs
------------------------
- offset_ct
    Plate- and target-specific shift applied to Ct values.
    Positive offset → plate measured higher Ct than reference.

- ct_calibrated
    Corrected Ct value aligned across plates.

- was_calibrated
    Boolean indicating whether calibration was applied
    (False if offset was unavailable).


Quality Control Considerations
------------------------------
Calibration should always be inspected before downstream analysis.

Key checks:

1. Replicate consistency
   - Are calibrator Ct values consistent within plate?
   - High SD suggests unreliable offsets.

2. Replicate count
   - Low replicate numbers (<3) reduce reliability.

3. Plate coverage
   - Each target should have calibrators on every plate.

4. Offset magnitude
   - Very large offsets may indicate technical issues.

5. Missing offsets
   - Rows without offsets will not be calibrated.


Limitations
-----------
- Does not correct for efficiency differences between targets
- Assumes linear additive plate effects in Ct space
- Does not handle missing calibrator targets gracefully unless checked externally
- Not suitable if calibrator identity differs across plates


Best Practices
--------------
- Use a pooled cDNA sample as calibrator
- Include calibrators on every plate for every target
- Use ≥3 technical replicates
- Keep experimental conditions consistent across plates
- Inspect calibrator variability before applying correction
- Remove calibrator wells before downstream biological analysis


References
----------
- Hellemans et al., Genome Biology (2007) – qBase framework
- MIQE Guidelines (Bustin et al., 2009)
- Ruijter et al. – qPCR data analysis best practices
"""

from __future__ import annotations

import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _validate_unique_keys(df: pd.DataFrame, keys: list[str], context: str) -> None:
    duplicates = df.duplicated(subset=keys, keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, keys].head(10).to_dict("records")
        raise ValueError(
            f"Expected unique rows for keys {keys} in {context}, but found duplicates. "
            f"Examples: {examples}"
        )


def summarize_calibrators(
    df: pd.DataFrame,
    *,
    plate_col: str = "plate_id",
    target_col: str = "target",
    ct_col: str = "ct",
    is_calibrator_col: str = "is_calibrator",
    min_calibrator_reps: int | None = 3,
) -> pd.DataFrame:
    """
    Summarize calibrator Ct values per plate and target.

    Assumes target-specific calibrators with technical replicates.
    """
    _require_columns(
        df,
        [plate_col, target_col, ct_col, is_calibrator_col],
        context="input dataframe for calibrator summary",
    )

    out = df.loc[df[is_calibrator_col].fillna(False).astype(bool)].copy()
    out = out.loc[out[ct_col].notna()].copy()

    if out.empty:
        return pd.DataFrame(
            columns=[
                plate_col,
                target_col,
                "calibrator_ct_mean",
                "calibrator_ct_std",
                "n_calibrator_reps",
                "low_replicate_warning",
            ]
        )

    summary = (
        out.groupby([plate_col, target_col], dropna=False)[ct_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "calibrator_ct_mean",
                "std": "calibrator_ct_std",
                "count": "n_calibrator_reps",
            }
        )
    )

    if min_calibrator_reps is not None:
        summary["low_replicate_warning"] = summary["n_calibrator_reps"] < min_calibrator_reps
    else:
        summary["low_replicate_warning"] = False

    return summary


def calculate_plate_calibrator_offsets(
    calibrator_summary_df: pd.DataFrame,
    *,
    plate_col: str = "plate_id",
    target_col: str = "target",
    calibrator_mean_col: str = "calibrator_ct_mean",
    reference_method: str = "global_mean",
    reference_plate: str | None = None,
) -> pd.DataFrame:
    """
    Compute target-specific plate offsets.

    Offset is calculated separately for each target:
        offset_ct = calibrator_ct_mean(plate, target) - reference_calibrator_ct(target)
    """
    _require_columns(
        calibrator_summary_df,
        [plate_col, target_col, calibrator_mean_col],
        context="calibrator summary",
    )

    out = calibrator_summary_df.copy()
    _validate_unique_keys(out, [plate_col, target_col], context="calibrator summary")

    if reference_method == "global_mean":
        ref = (
            out.groupby(target_col, dropna=False)[calibrator_mean_col]
            .mean()
            .reset_index(name="reference_calibrator_ct")
        )
    elif reference_method == "reference_plate":
        if reference_plate is None:
            raise ValueError(
                "reference_plate must be provided when reference_method='reference_plate'."
            )

        ref = out.loc[out[plate_col] == reference_plate, [target_col, calibrator_mean_col]].copy()
        if ref.empty:
            raise ValueError(f"No calibrator rows found for reference plate {reference_plate!r}.")

        _validate_unique_keys(ref, [target_col], context="reference plate calibrator rows")
        ref = ref.rename(columns={calibrator_mean_col: "reference_calibrator_ct"})
    else:
        raise ValueError("reference_method must be 'global_mean' or 'reference_plate'.")

    out = out.merge(ref, on=target_col, how="left", validate="many_to_one")
    out["offset_ct"] = out[calibrator_mean_col] - out["reference_calibrator_ct"]

    return out


def apply_plate_calibration(
    df: pd.DataFrame,
    offsets_df: pd.DataFrame,
    *,
    plate_col: str = "plate_id",
    target_col: str = "target",
    ct_col: str = "ct",
    offset_col: str = "offset_ct",
    output_col: str = "ct_calibrated",
    keep_calibrator_rows: bool = True,
    is_calibrator_col: str = "is_calibrator",
) -> pd.DataFrame:
    """
    Apply target-specific plate offsets to all rows.

    Each row is corrected using the offset for its own plate and target.
    """
    _require_columns(df, [plate_col, target_col, ct_col], context="input dataframe")
    _require_columns(offsets_df, [plate_col, target_col, offset_col], context="offset dataframe")

    offset_subset = offsets_df[[plate_col, target_col, offset_col]].copy()
    _validate_unique_keys(offset_subset, [plate_col, target_col], context="offset dataframe")

    out = df.copy()
    out = out.merge(
        offset_subset,
        on=[plate_col, target_col],
        how="left",
        validate="many_to_one",
    )

    out[output_col] = out[ct_col]
    mask = out[ct_col].notna() & out[offset_col].notna()
    out.loc[mask, output_col] = out.loc[mask, ct_col] - out.loc[mask, offset_col]
    out["was_calibrated"] = mask

    if not keep_calibrator_rows and is_calibrator_col in out.columns:
        out = out.loc[~out[is_calibrator_col].fillna(False).astype(bool)].copy()

    return out.reset_index(drop=True)