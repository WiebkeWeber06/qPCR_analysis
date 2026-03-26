# qPCR Analysis Workflow

## Overview

This project provides a modular and reusable pipeline for qPCR data analysis.  
The workflow is designed to handle diverse datasets, including experiments spanning multiple plates, while maintaining a consistent internal data structure.

The pipeline is divided into the following stages:

1. Data Import & Standardization
2. Data Validation
3. Preprocessing
4. Analysis
5. Statistics
6. Visualization

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

---

## Module Structure

qpcr/
├── __init__.py
├── io.py              # import raw files, merge plates, reshape into tidy format
├── schema.py          # column validation, required fields, data checks
├── preprocess.py      # cleaning, Ct filtering, technical replicate handling
├── analysis.py        # ΔCt, ΔΔCt, fold change, efficiency correction if needed
├── statistics.py      # t-tests, ANOVA, mixed models if needed
├── plotting.py        # QC plots, boxplots, stripplots, fold-change figures
├── config.py          # optional defaults for column names / thresholds
└── utils.py           # shared helper functions


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

**Key Functions:**
- `filter_invalid_ct()`
- `summarize_technical_replicates()`

---

### 4. Analysis (`analysis.py`)

**Goal:** Perform core qPCR calculations.

**Typical Calculations:**
- Mean Ct per sample/target
- ΔCt = Ct(target) − Ct(reference)
- ΔΔCt = ΔCt(sample) − ΔCt(control)
- Fold change = 2^(-ΔΔCt)

**Key Functions:**
- `calculate_dct()`
- `calculate_ddct()`
- `calculate_fold_change()`

---

### 5. Statistics (`statistics.py`)

**Goal:** Perform statistical comparisons between groups.

**Examples:**
- Two groups → t-test
- Multiple groups → ANOVA
- Complex designs → linear or mixed models (optional)

**Important:**  
Statistics operate on already processed data (e.g., fold change), not raw Ct values.

**Key Functions:**
- `compare_groups()`

---

### 6. Visualization (`plotting.py`)

**Goal:** Generate plots for quality control and results.

**Plot Types:**
- Ct distributions
- Technical replicate variability
- Fold change boxplots
- Grouped barplots with raw data points
- Plate heatmaps (optional)

**Key Functions:**
- `plot_ct_distribution()`
- `plot_fold_change()`

---

## Plate Handling Strategy

- Always retain `plate_id` in the dataset
- Allow:
  - Per-plate normalization
  - Cross-plate analysis
- Enable inclusion of `plate_id` in statistical models if needed

---

## Minimum Viable Product (MVP)

The first working version should support:

- Import of one or more qPCR files
- Conversion to canonical dataframe format
- Validation of required columns
- Technical replicate averaging
- ΔCt, ΔΔCt, and fold change calculation
- Basic fold change plotting

---

## First Milestone

> Given one or more qPCR export files, produce a clean, validated, and standardized dataframe with one row per measurement.

---

## Recommended Development Order

1. Define canonical dataframe schema
2. Implement `schema.py`
3. Implement `io.py`
4. Test on a small example dataset
5. Implement `preprocess.py`
6. Add `analysis.py`
7. Add `statistics.py`
8. Add `plotting.py`

---

## Notes

- Avoid hardcoding column names → use configurable mappings
- Keep modules independent and reusable
- Build small, testable functions instead of large scripts
- Preserve flexibility for different experimental designs

---
