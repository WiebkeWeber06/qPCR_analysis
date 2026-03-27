# qPCR Analysis Workflow

This document describes the full data analysis workflow implemented in the qPCR pipeline, from raw input files to final results and figures.

---

## Overview of the Pipeline

The workflow consists of:

1. Data import and merging
2. Metadata definition
3. Quality control (QC)
4. Optional inter-plate calibration
5. Preprocessing
6. Technical replicate summarization
7. Normalization
8. Statistical analysis
9. Visualization
10. Export

---

## 1. Data Import and Merging

### Inputs

* qPCR machine output file (CSV)
* Plate setup file (metadata)

### Process

* Load Ct values
* Load metadata
* Merge using:

  * plate_id
  * well

### Output

* `merged_df`

---

## 2. Metadata Structure

Dynamic column detection enables flexible designs.

### Key structures

* **group_cols** → define technical replicates
* **id_cols** → define biological identity
* **required_metadata** → required fields

Optional columns (e.g. timepoint) are included automatically if present.

---

## 3. Quality Control (QC)

Performed before filtering.

### Checks

* Missing metadata
* Replicate consistency
* Ct variability
* Control behavior
* Outliers

### Output

* QC report dictionary with multiple tables

---

## 4. Inter-Plate Calibration (Optional)

If calibrators are present:

### Steps

1. Identify calibrators
2. Compute mean Ct per plate & target
3. Define reference (global or plate)
4. Compute offset
5. Apply correction

```
ct_calibrated = ct + offset
```

---

## 5. Preprocessing

* Remove missing Ct values
* Remove controls (optional)
* Keep calibrators (optional)

### Output

* `processed_df`

Split into:

* `analysis_input_df`
* `calibrator_only_df`

---

## 6. Technical Replicate Summarization

Grouped by:

* plate_id
* group
* sample_id
* bio_rep
* target
* (optional) timepoint

### Metrics

* mean Ct
* standard deviation
* replicate count

### Output

* `summary_df`

---

## 7. Normalization

Reference gene-based normalization:

```
normalized_expression
log2_normalized_expression
```

### Inputs

* summary_df
* reference targets

### Output

* `analysis_df`

---

## 8. Statistical Analysis

### Linear models

One model per gene:

```
log2_normalized_expression ~ factors (+ optional plate effect)
```

Factors:

* sample_id
* group
* timepoint (optional)

### Outputs

* ANOVA table (`model_terms_df`)
* coefficients (`coefficient_df`)

---

### Pairwise comparisons

* Welch t-test
* Multiple testing correction (FDR)

### Output

* `pairwise_df`

---

### Interaction effects

* Extracted from model terms
* Used to evaluate differential responses

---

## 9. Visualization

### Plot types

* Expression grid
* Time-course plots
* Single-gene plots
* Calibrator offset plots

### Principles

* Raw data always shown
* Summary statistics overlaid
* Statistical annotations optional
* Fully data-driven

---

## 10. Export

### Excel output

```
output/qpcr_results.xlsx
```

Includes:

* processed data
* normalized data
* statistics
* QC

---

### Figures

```
output/figures/
```

---

## Reproducibility

* Entire workflow runs from `run_pipeline.py`
* No manual steps required
* Modular and inspectable

---

## Notes

* Designed for flexibility across experiments
* Requires tidy input data
* Statistical interpretation depends on biological context

