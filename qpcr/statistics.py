"""
Statistics utilities for qPCR data.

Design principles
-----------------
- analyze each target gene separately
- work on normalized / log2-normalized expression values
- keep factor names fully user-defined
- support complex experimental designs via statsmodels formulas
- provide transparent, tidy outputs for reporting and plotting

What this module does
---------------------
1. Fit one linear model per target gene
2. Extract an ANOVA-style table for each target
3. Perform all pairwise comparisons within each target based on user-chosen factor columns
4. Adjust p-values for multiple testing
5. Summarize means / SD / SEM for plotting

Important note
--------------
This module does not hard-code specific factors such as genotype, condition, or timepoint.
The user decides which columns are experimental factors and which model formula to use.

Typical usage
-------------
>>> from qpcr.statistics import (
...     fit_models_by_target,
...     run_pairwise_comparisons,
...     summarize_for_plotting,
... )
>>>
>>> model_terms_df, coefficient_df = fit_models_by_target(
...     analysis_df,
...     formula="log2_normalized_expression ~ C(genotype) * C(condition) + C(plate_id)",
... )
>>>
>>> pairwise_df = run_pairwise_comparisons(
...     analysis_df,
...     factor_cols=["genotype", "condition"],
... )
"""

from __future__ import annotations

import itertools
from typing import Sequence

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


DEFAULT_TARGET_COL: str = "target"
DEFAULT_VALUE_COL: str = "log2_normalized_expression"


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _make_group_label(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.Series:
    """
    Build a combined comparison label from user-specified factor columns.

    Example
    -------
    factor_cols = ["genotype", "condition", "timepoint"]
    -> WT | BL | 0h
    """
    _require_columns(df, factor_cols, context="group label construction")
    label_df = df[list(factor_cols)].copy().astype("string")
    return label_df.fillna("NA").agg(" | ".join, axis=1)

def extract_interaction_pvalues(
    model_terms_df: pd.DataFrame,
    *,
    target_col: str = "target",
    term_col: str = "term",
    p_col: str = "p_adj",
    interaction_term: str = "C(sample_id):C(group)",
) -> pd.DataFrame:
    """
    Extract one interaction term per target from model ANOVA tables.

    Parameters
    ----------
    model_terms_df
        Output from fit_models_by_target().
    target_col
        Target column name.
    term_col
        Model-term column name.
    p_col
        P-value column to use for reporting, typically adjusted p-values.
    interaction_term
        Exact interaction term to extract.

    Returns
    -------
    pd.DataFrame
        One row per target with interaction significance information.
    """
    _require_columns(model_terms_df, [target_col, term_col], "interaction extraction")

    out = model_terms_df.loc[model_terms_df[term_col] == interaction_term].copy()
    if out.empty:
        return pd.DataFrame(columns=[target_col, term_col, p_col, "significance"])

    if p_col not in out.columns:
        out[p_col] = pd.NA

    out["significance"] = out[p_col].apply(_significance_stars)
    return out.reset_index(drop=True)

def _add_significance_columns(
    df: pd.DataFrame,
    *,
    raw_p_col: str,
    adjusted_p_col: str = "p_adj",
    method: str = "fdr_bh",
    significance_source: str = "adjusted",
) -> pd.DataFrame:
    """
    Add adjusted p-values, reject flags, and significance labels.

    Parameters
    ----------
    df
        Input dataframe with a raw p-value column.
    raw_p_col
        Column containing raw p-values.
    adjusted_p_col
        Output column for adjusted p-values.
    method
        Multiple-testing correction method.
    significance_source
        Whether significance labels should be based on:
        - "adjusted"
        - "raw"
    """
    out = df.copy()

    if raw_p_col not in out.columns:
        out[adjusted_p_col] = pd.NA
        out["reject_null"] = pd.NA
        out["p_adjust_method"] = pd.NA
        out["significance"] = ""
        return out

    mask = out[raw_p_col].notna()
    out[adjusted_p_col] = pd.NA
    out["reject_null"] = pd.NA
    out["p_adjust_method"] = pd.NA

    if mask.any():
        reject, p_adj, _, _ = multipletests(out.loc[mask, raw_p_col], method=method)
        out.loc[mask, adjusted_p_col] = p_adj
        out.loc[mask, "reject_null"] = reject
        out.loc[mask, "p_adjust_method"] = method

    if significance_source == "adjusted":
        source_col = adjusted_p_col
    elif significance_source == "raw":
        source_col = raw_p_col
    else:
        raise ValueError("significance_source must be 'adjusted' or 'raw'.")

    out["significance"] = out[source_col].apply(_significance_stars)
    return out


def fit_target_model(
    df: pd.DataFrame,
    *,
    target: str,
    formula: str,
    target_col: str = DEFAULT_TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one formula-based linear model for a single target gene.

    Parameters
    ----------
    df
        Input dataframe containing one or more targets.
    target
        Target gene to subset and model.
    formula
        statsmodels formula, e.g.
        'log2_normalized_expression ~ C(genotype) * C(condition) + C(plate_id)'
    target_col
        Name of target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (anova_table, coefficient_table)
    """
    _require_columns(df, [target_col], context="fit_target_model input")

    sub_df = df.loc[df[target_col] == target].copy()
    if sub_df.empty:
        raise ValueError(f"No rows found for target={target!r}")

    model = smf.ols(formula=formula, data=sub_df).fit()

    anova_df = anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
    anova_df.insert(0, "target", target)
    anova_df["formula"] = formula
    anova_df["n_rows"] = len(sub_df)
    anova_df = _add_significance_columns(
        anova_df,
        raw_p_col="PR(>F)",
        adjusted_p_col="p_adj",
        method="fdr_bh",
        significance_source="adjusted",
    )

    coef_df = model.summary2().tables[1].reset_index().rename(columns={"index": "coefficient"})
    coef_df.insert(0, "target", target)
    coef_df["formula"] = formula
    coef_df["n_rows"] = len(sub_df)

    if "P>|t|" in coef_df.columns:
        coef_df = _add_significance_columns(
            coef_df,
            raw_p_col="P>|t|",
            adjusted_p_col="p_adj",
            method="fdr_bh",
            significance_source="adjusted",
        )

    return anova_df, coef_df


def fit_models_by_target(
    df: pd.DataFrame,
    *,
    formula: str,
    target_col: str = DEFAULT_TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the same model formula separately for each target gene.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (all_anova_tables, all_coefficient_tables)
    """
    _require_columns(df, [target_col], context="fit_models_by_target input")

    targets = df[target_col].dropna().unique().tolist()
    if not targets:
        raise ValueError("No targets found in dataframe.")

    anova_tables = []
    coefficient_tables = []

    for target in targets:
        try:
            anova_df, coef_df = fit_target_model(
                df,
                target=target,
                formula=formula,
                target_col=target_col,
            )
            anova_tables.append(anova_df)
            coefficient_tables.append(coef_df)
        except Exception as exc:
            error_row = pd.DataFrame(
                {
                    "target": [target],
                    "term": ["MODEL_ERROR"],
                    "formula": [formula],
                    "error": [str(exc)],
                    "n_rows": [int((df[target_col] == target).sum())],
                }
            )
            anova_tables.append(error_row)

    anova_out = pd.concat(anova_tables, ignore_index=True) if anova_tables else pd.DataFrame()
    coef_out = pd.concat(coefficient_tables, ignore_index=True) if coefficient_tables else pd.DataFrame()

    return anova_out, coef_out


def run_pairwise_comparisons(
    df: pd.DataFrame,
    *,
    factor_cols: Sequence[str],
    target_col: str = DEFAULT_TARGET_COL,
    value_col: str = DEFAULT_VALUE_COL,
    equal_var: bool = False,
    min_n_per_group: int = 2,
    p_adjust_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Run pairwise comparisons within each target gene.

    By default this uses Welch's t-test (equal_var=False) on the chosen value column.
    This is intended as a transparent post-hoc comparison layer after the model fit.
    It is not a replacement for the target-wise linear model.
    """
    _require_columns(
        df,
        list(factor_cols) + [target_col, value_col],
        context="pairwise comparisons input",
    )

    out = df.copy()
    out["comparison_group"] = _make_group_label(out, factor_cols)

    results = []

    for target, sub_df in out.groupby(target_col, dropna=False):
        groups = sub_df["comparison_group"].dropna().unique().tolist()

        for group_1, group_2 in itertools.combinations(groups, 2):
            vals_1 = pd.to_numeric(
                sub_df.loc[sub_df["comparison_group"] == group_1, value_col],
                errors="coerce",
            ).dropna()
            vals_2 = pd.to_numeric(
                sub_df.loc[sub_df["comparison_group"] == group_2, value_col],
                errors="coerce",
            ).dropna()

            if len(vals_1) < min_n_per_group or len(vals_2) < min_n_per_group:
                continue

            statistic, p_value = ttest_ind(vals_1, vals_2, equal_var=equal_var)

            results.append(
                {
                    "target": target,
                    "group_1": group_1,
                    "group_2": group_2,
                    "n_1": len(vals_1),
                    "n_2": len(vals_2),
                    "mean_1": vals_1.mean(),
                    "mean_2": vals_2.mean(),
                    "sd_1": vals_1.std(ddof=1),
                    "sd_2": vals_2.std(ddof=1),
                    "difference_mean_1_minus_2": vals_1.mean() - vals_2.mean(),
                    "test": "Welch_t_test" if not equal_var else "Student_t_test",
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        result_df["p_adj"] = pd.Series(dtype="float64")
        result_df["reject_null"] = pd.Series(dtype="boolean")
        result_df["p_adjust_method"] = pd.Series(dtype="string")
        result_df["significance"] = pd.Series(dtype="string")
        return result_df

    adjusted_frames = []
    for target, sub_df in result_df.groupby("target", dropna=False):
        sub_df = sub_df.copy()
        sub_df = _add_significance_columns(
            sub_df,
            raw_p_col="p_value",
            adjusted_p_col="p_adj",
            method=p_adjust_method,
            significance_source="adjusted",
        )
        adjusted_frames.append(sub_df)

    return pd.concat(adjusted_frames, ignore_index=True)


def summarize_for_plotting(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_col: str = DEFAULT_VALUE_COL,
    value_label: str | None = None,
) -> pd.DataFrame:
    """
    Summarize biological replicate data for plotting.

    Returns
    -------
    pd.DataFrame
        One row per group with:
        - mean_value
        - sd_value
        - sem_value
        - n_bio_reps
        - summarized_value_col
        - summarized_value_label
    """
    _require_columns(
        df,
        list(group_cols) + [value_col],
        context="plot summary input",
    )

    out = (
        df.groupby(list(group_cols), dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_value",
                "std": "sd_value",
                "count": "n_bio_reps",
            }
        )
    )

    out["sem_value"] = out["sd_value"] / out["n_bio_reps"] ** 0.5
    out["summarized_value_col"] = value_col
    out["summarized_value_label"] = value_label if value_label is not None else value_col

    return out


def adjust_pvalues(
    df: pd.DataFrame,
    *,
    p_col: str = "p_value",
    by: str | Sequence[str] | None = None,
    method: str = "fdr_bh",
    output_col: str = "p_adj",
) -> pd.DataFrame:
    """
    Adjust p-values in an existing dataframe.
    """
    _require_columns(df, [p_col], context="adjust_pvalues input")
    out = df.copy()

    if by is None:
        mask = out[p_col].notna()
        out[output_col] = pd.NA
        out["reject_null"] = pd.NA
        out["p_adjust_method"] = pd.NA

        if mask.any():
            reject, p_adj, _, _ = multipletests(out.loc[mask, p_col], method=method)
            out.loc[mask, output_col] = p_adj
            out.loc[mask, "reject_null"] = reject
            out.loc[mask, "p_adjust_method"] = method

        return out

    by_cols = [by] if isinstance(by, str) else list(by)
    _require_columns(out, by_cols, context="adjust_pvalues grouping")

    frames = []
    for _, sub_df in out.groupby(by_cols, dropna=False):
        sub_df = sub_df.copy()
        mask = sub_df[p_col].notna()
        sub_df[output_col] = pd.NA
        sub_df["reject_null"] = pd.NA
        sub_df["p_adjust_method"] = pd.NA

        if mask.any():
            reject, p_adj, _, _ = multipletests(sub_df.loc[mask, p_col], method=method)
            sub_df.loc[mask, output_col] = p_adj
            sub_df.loc[mask, "reject_null"] = reject
            sub_df.loc[mask, "p_adjust_method"] = method

        frames.append(sub_df)

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "DEFAULT_TARGET_COL",
    "DEFAULT_VALUE_COL",
    "fit_target_model",
    "fit_models_by_target",
    "run_pairwise_comparisons",
    "summarize_for_plotting",
    "adjust_pvalues",
]