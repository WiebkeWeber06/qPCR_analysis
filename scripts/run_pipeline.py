# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:30:08 2026

@author: wiewe372
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qpcr.analysis import calculate_normalized_expression
from qpcr.calibration import (
    apply_plate_calibration,
    calculate_plate_calibrator_offsets,
    summarize_calibrators,
)
from qpcr.io import load_biorad_csv, merge_plate_design, read_plate_setup
from qpcr.plotting import (
    plot_calibrator_offsets,
    plot_expression_grid,
    plot_single_gene,
    plot_timecourse_grid,
    save_figure_multiple_formats,
)
from qpcr.preprocess import (
    filter_invalid_ct,
    separate_calibrators,
    summarize_technical_replicates,
)
from qpcr.qc import qc_report
from qpcr.schema import validate_qpcr_dataframe
from qpcr.statistics import (
    fit_models_by_target,
    run_pairwise_comparisons,
    summarize_for_plotting,
)


# =========================================================
# User settings
# =========================================================

DATA_FILE = Path("tests/data/SsoAdv_Univ_SYBR_cDNA_30s_Wiebke-13-02-26.csv")
PLATE_SETUP_FILE = Path("tests/data/plate_setup_quick-plate_384_wells_13-02-26_manually_improved.csv")
PLATE_ID = "plate_1"

REFERENCE_TARGETS = ["GBLP", "RPL13"]
EXPECTED_TECH_REPS = 3

USE_CALIBRATORS = True
CALIBRATOR_REFERENCE_METHOD = "global_mean"   # "global_mean" or "reference_plate"
CALIBRATOR_REFERENCE_PLATE = None

INCLUDE_PLATE_IN_MODEL = True

OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "qpcr_results.xlsx"

# =========================================================
# Plotting settings
# =========================================================

FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_FORMATS = ("png", "pdf")

GROUP_COL_FOR_PLOTS = "group"
X_COL_FOR_PLOTS = "sample_id"
GENE_COL_FOR_PLOTS = "target"

PLOT_COLORS = {
    "CC-5415": "#1f77b4",
    "GAPR4_KO": "#d62728",
}

PLOT_VALUE_MODE = "normalized"

if PLOT_VALUE_MODE == "normalized":
    VALUE_COL_FOR_PLOTS = "normalized_expression"
    Y_LABEL_FOR_PLOTS = "normalized expression"
elif PLOT_VALUE_MODE == "log2":
    VALUE_COL_FOR_PLOTS = "log2_normalized_expression"
    Y_LABEL_FOR_PLOTS = "log2 normalized expression"
else:
    raise ValueError("PLOT_VALUE_MODE must be 'normalized' or 'log2'")

PLOT_Y_SCALE = "linear"
PLOT_Y_LIMITS = None


# =========================================================
# Small helpers
# =========================================================

def _sorted_preview_columns(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [col for col in preferred if col in df.columns]


def _print_df_preview(title: str, df: pd.DataFrame, columns: list[str], n: int = 10) -> None:
    print(f"\n{title}")
    if df.empty:
        print("DataFrame is empty.")
        return
    preview_cols = _sorted_preview_columns(df, columns)
    print(df[preview_cols].head(n).to_string(index=False))


def _has_real_values(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _save_and_close(fig, path_stem: Path, formats: tuple[str, ...]) -> None:
    save_figure_multiple_formats(fig, path_stem, formats=formats)
    plt.close(fig)
    saved_files = [str(path_stem.with_suffix(f".{ext}")) for ext in formats]
    print("Saved:", ", ".join(saved_files))
    
def _make_stats_factor_term(col: str) -> str:
    """Return the statsmodels categorical term for one factor column."""
    return f"C({col})"


def _find_interaction_term(
    model_terms_df: pd.DataFrame,
    factor_cols: list[str],
) -> str | None:
    """
    Find the highest-priority interaction term present in the model terms table.

    Preference order:
    1. Full interaction across all varying factors
    2. Two-way interactions, prioritizing sample_id:group if present
    3. Any remaining interaction term found in the table

    Parameters
    ----------
    model_terms_df
        ANOVA/model-term table returned by fit_models_by_target().
    factor_cols
        Biological factor columns that were used to build the model.

    Returns
    -------
    str or None
        Matching interaction term name if found, otherwise None.
    """
    if model_terms_df.empty or "term" not in model_terms_df.columns or not factor_cols:
        return None

    available_terms = set(model_terms_df["term"].dropna().astype(str))

    factor_terms = [_make_stats_factor_term(col) for col in factor_cols]

    # 1) Full interaction across all factors, e.g. C(sample_id):C(group):C(timepoint)
    if len(factor_terms) >= 2:
        full_term = ":".join(factor_terms)
        if full_term in available_terms:
            return full_term

    # 2) Prefer strain × condition interaction if both exist
    preferred_pairs = []
    if "sample_id" in factor_cols and "group" in factor_cols:
        preferred_pairs.append(
            ":".join([_make_stats_factor_term("sample_id"), _make_stats_factor_term("group")])
        )

    # add all remaining two-way combinations
    for i in range(len(factor_terms)):
        for j in range(i + 1, len(factor_terms)):
            pair_term = ":".join([factor_terms[i], factor_terms[j]])
            if pair_term not in preferred_pairs:
                preferred_pairs.append(pair_term)

    for term in preferred_pairs:
        if term in available_terms:
            return term

    # 3) Fallback: first interaction-like term found
    interaction_terms = [term for term in available_terms if ":" in term]
    if interaction_terms:
        return sorted(interaction_terms)[0]

    return None


def _extract_interaction_df(
    model_terms_df: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """
    Extract the most relevant interaction term from the model results.

    Parameters
    ----------
    model_terms_df
        ANOVA/model-term table returned by fit_models_by_target().
    factor_cols
        Biological factor columns used in the model.

    Returns
    -------
    pd.DataFrame
        Subset of model_terms_df containing the selected interaction term.
        Returns an empty dataframe if no interaction term is found.
    """
    interaction_term = _find_interaction_term(model_terms_df, factor_cols)
    if interaction_term is None:
        return pd.DataFrame()

    out = model_terms_df.loc[model_terms_df["term"] == interaction_term].copy()

    if not out.empty:
        out["interaction_term"] = interaction_term
        if "p_adj" in out.columns:
            out["significance"] = out["p_adj"].apply(
                lambda p: "" if pd.isna(p) else ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")
            )

    return out    

# =========================================================
# Load and merge data
# =========================================================

data_df = load_biorad_csv(DATA_FILE, plate_id=PLATE_ID)
design_df = read_plate_setup(PLATE_SETUP_FILE, plate_id=PLATE_ID)

merged_df = merge_plate_design(data_df, design_df, on=("plate_id", "well"))
merged_df = validate_qpcr_dataframe(merged_df)

print(f"Loaded machine data: {data_df.shape}")
print(f"Loaded plate setup:  {design_df.shape}")
print(f"Merged dataframe:    {merged_df.shape}")


# =========================================================
# Build dynamic metadata structure
# =========================================================

group_cols = ["plate_id", "group", "sample_id"]
id_cols = ["plate_id", "group", "sample_id"]
required_metadata = ["group", "sample_id", "bio_rep", "target"]

has_timepoint = _has_real_values(merged_df, "timepoint")

if has_timepoint:
    group_cols.append("timepoint")
    id_cols.append("timepoint")
    required_metadata.append("timepoint")

group_cols.extend(["bio_rep", "target"])
id_cols.append("bio_rep")

print("\nGrouping columns used for tech-rep summarization:")
print(group_cols)

print("\nID columns used for normalization:")
print(id_cols)


# =========================================================
# QC on merged data before filtering
# =========================================================

report = qc_report(
    merged_df,
    expected_tech_reps=EXPECTED_TECH_REPS,
    group_cols=group_cols,
    required_metadata=required_metadata,
)

overview_df = pd.DataFrame([report["overview"]])

print("\nQC overview:")
print(overview_df.to_string(index=False))


# =========================================================
# Optional inter-plate calibration
# =========================================================

calibrator_summary_df = pd.DataFrame()
calibrator_offsets_df = pd.DataFrame()
ct_col_for_downstream = "ct"

has_calibrator_rows = (
    "is_calibrator" in merged_df.columns
    and merged_df["is_calibrator"].fillna(False).astype(bool).any()
)

if USE_CALIBRATORS and has_calibrator_rows:
    print("\nApplying calibrator-based plate normalization...")

    calibrator_summary_df = summarize_calibrators(
        merged_df,
        plate_col="plate_id",
        target_col="target",
        ct_col="ct",
        is_calibrator_col="is_calibrator",
        min_calibrator_reps=3,
    )

    _print_df_preview(
        "Calibrator summary preview:",
        calibrator_summary_df,
        [
            "plate_id",
            "target",
            "calibrator_ct_mean",
            "calibrator_ct_std",
            "n_calibrator_reps",
            "low_replicate_warning",
        ],
    )

    calibrator_offsets_df = calculate_plate_calibrator_offsets(
        calibrator_summary_df,
        plate_col="plate_id",
        target_col="target",
        calibrator_mean_col="calibrator_ct_mean",
        reference_method=CALIBRATOR_REFERENCE_METHOD,
        reference_plate=CALIBRATOR_REFERENCE_PLATE,
    )

    _print_df_preview(
        "Plate calibrator offsets preview:",
        calibrator_offsets_df,
        [
            "plate_id",
            "target",
            "calibrator_ct_mean",
            "reference_calibrator_ct",
            "offset_ct",
        ],
    )

    merged_df = apply_plate_calibration(
        merged_df,
        calibrator_offsets_df,
        plate_col="plate_id",
        target_col="target",
        ct_col="ct",
        offset_col="offset_ct",
        output_col="ct_calibrated",
        keep_calibrator_rows=True,
        is_calibrator_col="is_calibrator",
    )

    ct_col_for_downstream = "ct_calibrated"

    n_calibrated_rows = int(
        merged_df["was_calibrated"].sum()
        if "was_calibrated" in merged_df.columns
        else 0
    )

    print(f"\nCalibration applied. Rows with calibrated Ct: {n_calibrated_rows}")

elif USE_CALIBRATORS and not has_calibrator_rows:
    print("\nUSE_CALIBRATORS=True, but no calibrator rows were found. Skipping calibration.")
    merged_df = merged_df.copy()
    merged_df["ct_calibrated"] = merged_df["ct"]
    merged_df["was_calibrated"] = False

else:
    print("\nCalibration disabled.")
    merged_df = merged_df.copy()
    merged_df["ct_calibrated"] = merged_df["ct"]
    merged_df["was_calibrated"] = False

print(f"Downstream analyses will use: {ct_col_for_downstream!r}")


# =========================================================
# Preprocessing
# =========================================================

processed_df = filter_invalid_ct(
    merged_df,
    drop_missing_ct=True,
    ct_col=ct_col_for_downstream,
    keep_controls=False,
    keep_calibrators=True,
)

analysis_input_df, calibrator_only_df = separate_calibrators(processed_df)

print(f"\nRows after preprocessing: {processed_df.shape}")
print(f"Rows used for summary:    {analysis_input_df.shape}")
print(f"Calibrator-only rows:     {calibrator_only_df.shape}")


# =========================================================
# Summarize technical replicates
# =========================================================

summary_df = summarize_technical_replicates(
    analysis_input_df,
    group_cols=group_cols,
    ct_col=ct_col_for_downstream,
    exclude_controls=True,
    exclude_calibrators=True,
)

summary_sort_cols = _sorted_preview_columns(
    summary_df,
    ["group", "sample_id", "timepoint", "bio_rep", "target"],
)
summary_df = summary_df.sort_values(summary_sort_cols).reset_index(drop=True)

_print_df_preview(
    "Technical replicate summary preview:",
    summary_df,
    [
        "plate_id",
        "group",
        "sample_id",
        "timepoint",
        "bio_rep",
        "target",
        "ct_mean",
        "ct_std",
        "n_rows",
        "n_valid_ct",
    ],
)


# =========================================================
# Normalization / analysis
# =========================================================

analysis_df = calculate_normalized_expression(
    df=summary_df,
    reference_targets=REFERENCE_TARGETS,
    id_cols=id_cols,
    include_fold_change=False,
    add_log2=True,
)

analysis_sort_cols = _sorted_preview_columns(
    analysis_df,
    ["group", "sample_id", "timepoint", "bio_rep", "target"],
)
analysis_df = analysis_df.sort_values(analysis_sort_cols).reset_index(drop=True)

if VALUE_COL_FOR_PLOTS not in analysis_df.columns:
    raise ValueError(
        f"Chosen plotting column {VALUE_COL_FOR_PLOTS!r} is missing from analysis_df. "
        f"Available columns: {analysis_df.columns.tolist()}"
    )

_print_df_preview(
    "Normalized expression preview:",
    analysis_df,
    [
        "plate_id",
        "group",
        "sample_id",
        "timepoint",
        "bio_rep",
        "target",
        "normalized_expression",
        "log2_normalized_expression",
    ],
)


# =========================================================
# Statistics
# =========================================================

candidate_factor_cols = ["sample_id", "group"]
if has_timepoint:
    candidate_factor_cols.append("timepoint")

varying_factor_cols = []
factor_terms = []

for col in candidate_factor_cols:
    if col in analysis_df.columns and analysis_df[col].dropna().nunique() > 1:
        varying_factor_cols.append(col)
        factor_terms.append(f"C({col})")

if not factor_terms:
    raise ValueError(
        "No varying biological factor columns were found for the statistics model. "
        "Check that sample_id/group/timepoint are present and contain more than one level."
    )

interaction_formula = " * ".join(factor_terms)

n_unique_plates = analysis_df["plate_id"].dropna().nunique() if "plate_id" in analysis_df.columns else 0

if INCLUDE_PLATE_IN_MODEL and n_unique_plates > 1:
    stats_formula = f"log2_normalized_expression ~ {interaction_formula} + C(plate_id)"
else:
    stats_formula = f"log2_normalized_expression ~ {interaction_formula}"

pairwise_factor_cols = varying_factor_cols.copy()

print("\nStatistics factor columns used:")
print(varying_factor_cols)

print("\nStatistics formula:")
print(stats_formula)

model_terms_df, coefficient_df = fit_models_by_target(
    analysis_df,
    formula=stats_formula,
)

interaction_df = _extract_interaction_df(
    model_terms_df,
    varying_factor_cols,
)

if not interaction_df.empty:
    print("\nInteraction term used for plotting:")
    print(interaction_df["interaction_term"].iloc[0])
else:
    print("\nNo interaction term found for plotting.")

pairwise_df = run_pairwise_comparisons(
    analysis_df,
    factor_cols=pairwise_factor_cols,
    value_col="log2_normalized_expression",
    equal_var=False,
    min_n_per_group=2,
    p_adjust_method="fdr_bh",
)

plot_group_cols = ["target", "sample_id", "group"]
if has_timepoint:
    plot_group_cols.append("timepoint")

plot_summary_df = summarize_for_plotting(
    analysis_df,
    group_cols=plot_group_cols,
    value_col=VALUE_COL_FOR_PLOTS,
    value_label=Y_LABEL_FOR_PLOTS,
)

_print_df_preview(
    "Model terms preview:",
    model_terms_df,
    ["target", "term", "sum_sq", "df", "F", "PR(>F)", "p_adj", "significance", "formula", "n_rows"],
)

_print_df_preview(
    "Coefficient preview:",
    coefficient_df,
    ["target", "coefficient", "Coef.", "Std.Err.", "t", "P>|t|", "p_adj", "significance", "formula", "n_rows"],
)

if not pairwise_df.empty:
    _print_df_preview(
        "Pairwise comparison preview:",
        pairwise_df,
        [
            "target",
            "group_1",
            "group_2",
            "n_1",
            "n_2",
            "difference_mean_1_minus_2",
            "p_value",
            "p_adj",
            "significance",
        ],
    )
else:
    print("\nPairwise comparison preview:")
    print("No pairwise comparisons generated.")


# =========================================================
# Figures
# =========================================================

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plot_suffix = "normalized" if PLOT_VALUE_MODE == "normalized" else "log2"

pairwise_df_plot = pairwise_df.copy()

pairwise_df_plot = pairwise_df_plot[
    pairwise_df_plot["group_1"].str.contains("|".join(analysis_df["sample_id"].unique())) &
    pairwise_df_plot["group_2"].str.contains("|".join(analysis_df["sample_id"].unique()))
]

print("\nStarting figure generation...")
print("Plotting mode:", PLOT_VALUE_MODE)
print("Plotting column:", VALUE_COL_FOR_PLOTS)
print("Figure output dir:", FIGURE_DIR.resolve())

# 1) Expression grid
fig1, axes1 = plot_expression_grid(
    analysis_df,
    gene_col=GENE_COL_FOR_PLOTS,
    condition_col=GROUP_COL_FOR_PLOTS,
    x_col=X_COL_FOR_PLOTS,
    value_col=VALUE_COL_FOR_PLOTS,
    error="sd",
    colors=PLOT_COLORS,
    title="qPCR expression overview",
    y_label=Y_LABEL_FOR_PLOTS,
    sharey=True,
    y_scale=PLOT_Y_SCALE,
    y_limits=PLOT_Y_LIMITS,
    stats_df=pairwise_df,
    stats_target_col="target",
    stats_group1_col="group_1",
    stats_group2_col="group_2",
    stats_p_col="p_adj",
    annotate_stats=True,
    show_ns=True,
    annotation_factor_cols=[X_COL_FOR_PLOTS, GROUP_COL_FOR_PLOTS],
)
_save_and_close(fig1, FIGURE_DIR / f"expression_grid_{plot_suffix}", FIGURE_FORMATS)

# 2) Optional timecourse grid
if _has_real_values(analysis_df, "timepoint"):
    fig2, axes2 = plot_timecourse_grid(
        analysis_df,
        gene_col=GENE_COL_FOR_PLOTS,
        condition_col=GROUP_COL_FOR_PLOTS,
        time_col="timepoint",
        line_col=X_COL_FOR_PLOTS,
        value_col=VALUE_COL_FOR_PLOTS,
        error="sd",
        colors=PLOT_COLORS,
        title="qPCR time-course",
        y_label=Y_LABEL_FOR_PLOTS,
        sharey=True,
        y_scale=PLOT_Y_SCALE,
        y_limits=PLOT_Y_LIMITS,
    )
    _save_and_close(fig2, FIGURE_DIR / f"timecourse_grid_{plot_suffix}", FIGURE_FORMATS)

# 3) Single-gene plots
for gene in analysis_df[GENE_COL_FOR_PLOTS].dropna().unique():
    safe_gene = str(gene).replace("/", "_").replace(" ", "_")

    fig_gene, ax_gene = plot_single_gene(
        analysis_df,
        gene=gene,
        gene_col=GENE_COL_FOR_PLOTS,
        x_col=X_COL_FOR_PLOTS,
        hue_col=GROUP_COL_FOR_PLOTS,
        value_col=VALUE_COL_FOR_PLOTS,
        error="sd",
        title=str(gene),
        y_label=Y_LABEL_FOR_PLOTS,
        y_scale=PLOT_Y_SCALE,
        y_limits=PLOT_Y_LIMITS,
        stats_df=pairwise_df,
        stats_target_col="target",
        stats_group1_col="group_1",
        stats_group2_col="group_2",
        stats_p_col="p_adj",
        annotate_stats=True,
        show_ns=True,
        annotation_factor_cols=[X_COL_FOR_PLOTS, GROUP_COL_FOR_PLOTS],
        interaction_df=interaction_df,
        interaction_target_col="target",
        interaction_p_col="p_adj",
        show_interaction=True,
    )

    _save_and_close(
        fig_gene,
        FIGURE_DIR / f"{safe_gene}_single_gene_{plot_suffix}",
        FIGURE_FORMATS,
    )

# 4) Optional calibrator offsets plot
if not calibrator_offsets_df.empty:
    fig_cal, axes_cal = plot_calibrator_offsets(
        calibrator_offsets_df,
        target_col="target",
        plate_col="plate_id",
        offset_col="offset_ct",
        title="Inter-plate calibrator offsets",
    )
    _save_and_close(
        fig_cal,
        FIGURE_DIR / "calibrator_offsets",
        FIGURE_FORMATS,
    )

print(f"\nSaved figures to: {FIGURE_DIR.resolve()}")


# =========================================================
# Export to Excel
# =========================================================

OUTPUT_DIR.mkdir(exist_ok=True)

with pd.ExcelWriter(OUTPUT_FILE) as writer:
    data_df.to_excel(writer, sheet_name="machine_data", index=False)
    design_df.to_excel(writer, sheet_name="plate_setup", index=False)
    merged_df.to_excel(writer, sheet_name="merged_data", index=False)
    processed_df.to_excel(writer, sheet_name="processed_data", index=False)
    summary_df.to_excel(writer, sheet_name="tech_rep_summary", index=False)
    analysis_df.to_excel(writer, sheet_name="normalized_expression", index=False)

    model_terms_df.to_excel(writer, sheet_name="stats_model_terms", index=False)
    coefficient_df.to_excel(writer, sheet_name="stats_coefficients", index=False)
    pairwise_df.to_excel(writer, sheet_name="stats_pairwise", index=False)
    plot_summary_df.to_excel(writer, sheet_name="stats_plot_summary", index=False)

    overview_df.to_excel(writer, sheet_name="qc_overview", index=False)
    report["missing_metadata"].to_excel(writer, sheet_name="qc_missing_metadata", index=False)
    report["replicate_issues"].to_excel(writer, sheet_name="qc_replicate_issues", index=False)
    report["replicate_variability"].to_excel(writer, sheet_name="qc_replicate_variability", index=False)
    report["control_summary"].to_excel(writer, sheet_name="qc_control_summary", index=False)
    report["suspicious_controls"].to_excel(writer, sheet_name="qc_suspicious_controls", index=False)
    report["ct_outliers"].to_excel(writer, sheet_name="qc_ct_outliers", index=False)

    if not calibrator_only_df.empty:
        calibrator_only_df.to_excel(writer, sheet_name="calibrator_rows", index=False)

    if not calibrator_summary_df.empty:
        calibrator_summary_df.to_excel(writer, sheet_name="calibrator_summary", index=False)

    if not calibrator_offsets_df.empty:
        calibrator_offsets_df.to_excel(writer, sheet_name="calibrator_offsets", index=False)

print(f"\nSaved results to: {OUTPUT_FILE.resolve()}")