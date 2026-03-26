# qPCR Analysis Workflow

## Overview

This project provides a modular and reusable pipeline for qPCR data analysis.
The workflow is designed to handle diverse datasets, including experiments spanning multiple plates, while maintaining a consistent internal data structure.

The pipeline is divided into the following stages:

1. Data Import & Standardization
2. Data Validation
3. Preprocessing
4. Standard Curve Handling
5. Analysis
6. Statistics
7. Visualization

---

## Core Design Principle

All modules operate on a **standardized tidy dataframe format**.

### Canonical Data Format

Each row represents one measurement:

| Column Name  | Description |
|--------------|------------|
| sample_id    | Unique sample identifier |
| plate_id     | Plate identifier (important for multi-plate experiments) |
| well         | Well position (optional) |
| target       | Gene/target name |
| ct           | Ct value |
| group        | Experimental group |
| bio_rep      | Biological replicate |
| tech_rep     | Technical replicate |
| reference    | Reference gene indicator (optional) |
| treatment    | Treatment condition (optional) |
| dilution     | Dilution factor or concentration for standard curves (optional) |
| standard_id  | Identifier for standard curve samples (optional) |
| is_standard  | Boolean flag indicating whether a row belongs to a standard curve (optional) |

---

## Module Structure

```text
qpcr/
├── __init__.py
├── io.py
├── schema.py
├── preprocess.py
├── standard_curve.py
├── analysis.py
├── statistics.py
├── plotting.py
├── config.py
└── utils.py
```

---

## Workflow Steps

### 1. Data Import & Standardization (`io.py`)

**Goal:** Convert raw qPCR exports into the canonical dataframe format.

**Responsibilities:**
- Load CSV or Excel files
- Handle different vendor formats
- Rename columns using a mapping (`column_map`)
- Reshape wide → long format if necessary
- Combine multiple plates into one dataframe
- Add `plate_id` metadata
- Preserve metadata needed for standard curves, such as dilution series or known input concentrations

**Key Functions:**
- `load_qpcr_file()`
- `load_multiple_plates()`
- `standardize_columns()`

---

### 2. Data Validation (`schema.py`)

**Goal:** Ensure all required information is present and correctly formatted.

**Checks:**
- Required columns exist:
  - `sample_id`, `plate_id`, `target`, `ct`
- Data types are valid
- No unexpected missing values in critical fields
- Standard curve rows contain the required dilution or concentration metadata when `is_standard = True`

**Key Functions:**
- `validate_qpcr_dataframe()`

---

### 3. Preprocessing (`preprocess.py`)

**Goal:** Clean and prepare data for analysis.

**Steps:**
- Remove invalid or missing Ct values
- Apply Ct cutoffs (e.g., undetermined values)
- Detect and handle outliers in technical replicates
- Aggregate technical replicates (e.g., mean Ct)
- Optionally preprocess standard curve replicates separately from experimental samples

**Key Functions:**
- `filter_invalid_ct()`
- `summarize_technical_replicates()`

---

### 4. Standard Curve Handling (`standard_curve.py`)

**Goal:** Support primer validation and efficiency-aware analysis using standard curves.

**Use Cases:**
- Estimate primer efficiency
- Evaluate assay linearity
- Compare primer performance across plates
- Decide whether a primer set passes quality thresholds
- Use efficiency-corrected quantification if needed

**Typical Calculations:**
- Fit Ct versus log10(input amount or dilution)
- Calculate slope
- Calculate intercept
- Calculate R²
- Calculate amplification efficiency

**Common Quality Outputs:**
- Slope
- R²
- Efficiency percentage
- Pass/fail flag based on configurable thresholds

**Typical Workflow:**
1. Identify standard curve rows
2. Group by primer/target and optionally plate
3. Fit regression model
4. Store standard curve metrics in a summary table
5. Optionally merge efficiency estimates into downstream analysis

**Key Functions:**
- `fit_standard_curve()`
- `summarize_standard_curves()`
- `calculate_efficiency()`
- `flag_poor_standard_curves()`

---

### 5. Analysis (`analysis.py`)

**Goal:** Perform core qPCR calculations.

**Typical Calculations:**
- Mean Ct per sample/target
- ΔCt = Ct(target) − Ct(reference)
- ΔΔCt = ΔCt(sample) − ΔCt(control)
- Fold change = 2^(-ΔΔCt)

**Optional Extension:**
- Efficiency-corrected relative quantification using primer-specific standard curve efficiencies

**Key Functions:**
- `calculate_dct()`
- `calculate_ddct()`
- `calculate_fold_change()`
- `calculate_efficiency_corrected_expression()`

---

### 6. Statistics (`statistics.py`)

**Goal:** Perform statistical comparisons between groups.

**Examples:**
- Two groups → t-test
- Multiple groups → ANOVA
- Complex designs → linear or mixed models (optional)

**Important:**
Statistics operate on already processed data, not raw Ct values.
Depending on the workflow, statistics may be run on ΔCt values, fold changes, or efficiency-corrected values.

**Key Functions:**
- `compare_groups()`

---

### 7. Visualization (`plotting.py`)

**Goal:** Generate plots for quality control and results.

**Plot Types:**
- Ct distributions
- Technical replicate variability
- Fold change boxplots
- Grouped barplots with raw data points
- Plate heatmaps (optional)
- Standard curve plots with regression line and annotated slope, R², and efficiency

**Key Functions:**
- `plot_ct_distribution()`
- `plot_fold_change()`
- `plot_standard_curve()`

---

## Plate Handling Strategy

- Always retain `plate_id` in the dataset
- Allow:
  - Per-plate normalization
  - Cross-plate analysis
  - Per-plate standard curve fitting when relevant
- Enable inclusion of `plate_id` in statistical models if needed

---

## Standard Curve Strategy

Standard curves should be treated as a first-class part of the pipeline rather than an afterthought.

### Recommended Design Choices
- Keep standard curve data in the same canonical dataframe where possible
- Mark them explicitly with `is_standard`
- Preserve dilution or concentration metadata
- Allow one standard curve per target per plate
- Allow optional aggregation across plates if the assay design justifies it
- Store curve metrics in a separate summary dataframe for downstream use

### Typical Output Table for Standard Curves

| target | plate_id | slope | intercept | r_squared | efficiency | pass_qc |
|--------|----------|-------|-----------|-----------|------------|---------|

---

## Minimum Viable Product (MVP)

The first working version should support:

- Import of one or more qPCR files
- Conversion to canonical dataframe format
- Validation of required columns
- Technical replicate averaging
- Standard curve fitting for primer validation
- Calculation of slope, R², and efficiency
- ΔCt, ΔΔCt, and fold change calculation
- Basic fold change plotting
- Basic standard curve plotting

---

## First Milestone

> Given one or more qPCR export files, produce a clean, validated, and standardized dataframe with one row per measurement, including support for standard curve samples.

---

## Recommended Development Order

1. Define canonical dataframe schema
2. Implement `schema.py`
3. Implement `io.py`
4. Test on a small example dataset
5. Implement `preprocess.py`
6. Implement `standard_curve.py`
7. Add `analysis.py`
8. Add `statistics.py`
9. Add `plotting.py`

---

## Notes

- Avoid hardcoding column names → use configurable mappings
- Keep modules independent and reusable
- Build small, testable functions instead of large scripts
- Preserve flexibility for different experimental designs
- Treat standard curve metadata as essential input, not optional after processing begins

---

