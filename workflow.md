# qPCR Analysis Workflow

This document describes the full data analysis workflow implemented in the qPCR pipeline, from raw input files to final results and figures.

The workflow is modular and designed to be reproducible, transparent, and adaptable to different experimental designs.

---

## Overview of the Pipeline

The analysis consists of the following major steps:

1. Data import and merging
2. Metadata validation and structure definition
3. Quality control (QC)
4. Optional inter-plate calibration
5. Preprocessing and filtering
6. Technical replicate summarization
7. Normalization (reference gene-based)
8. Statistical analysis
9. Visualization
10. Export of results

Each step operates on well-defined intermediate DataFrames, allowing inspection and debugging at every stage.

---

## 1. Data Import and Merging

### Inputs

* qPCR machine output file (e.g. Bio-Rad CSV)
* Plate setup file containing metadata per well

### Process

* Load raw Ct values from machine output
* Load experimental metadata (sample_id, group, target, etc.)
* Merge both datasets by:

  * plate_id
  * well position

### Output

* `merged_df`: combined dataset with Ct values and metadata

---

## 2. Metadata Structure Definition

The pipeline dynamically determines which metadata columns are present and relevant.

### Key concepts

* **group_cols**: define technical replicate grouping
* **id_cols**: define biological identity for normalization
* **required_metadata**: columns required for valid analysis

### Example

* group_cols:

  ```
  plate_id, group, sample_id, (optional: timepoint), bio_rep, target
  ```
* id_cols:

  ```
  plate_id, group, sample_id, (optional: timepoint), bio_rep
  ```

This allows flexible handling of experiments with or without time series.

---

## 3. Quality Control (QC)

Performed on the merged raw dataset before filtering.

### Checks include

* Missing metadata
* Unexpected number of technical replicates
* High variability between replicates
* Control well behavior
* Ct outliers

### Output

* QC report dictionary containing:

  * overview
  * missing_metadata
  * replicate_issues
  * replicate_variability
  * control_summary
  * suspicious_controls
  * ct_outliers

---

## 4. Optional Inter-Plate Calibration

If calibrator samples are present, inter-plate normalization is applied.

### Steps

1. Identify calibrator wells (`is_calibrator == True`)
2. Compute per-plate, per-target mean Ct
3. Define reference:

   * global mean OR
   * specific reference plate
4. Calculate offset:

   ```
   offset = reference_ct - plate_ct
   ```
5. Apply correction:

   ```
   ct_calibrated = ct + offset
   ```

### Output

* `ct_calibrated` column
* `calibrator_summary_df`
* `calibrator_offsets_df`

If no calibrators are present, raw Ct values are used unchanged.

---

## 5. Preprocessing and Filtering

### Steps

* Remove missing Ct values
* Remove control wells (optional)
* Retain calibrators for QC (optional)

### Output

* `processed_df`

Then split into:

* `analysis_input_df` (used for downstream analysis)
* `calibrator_only_df` (kept for reporting)

---

## 6. Technical Replicate Summarization

Technical replicates are aggregated into a single value per biological unit.

### Grouping

Defined by `group_cols`, typically including:

* plate_id
* group
* sample_id
* bio_rep
* target
* (optional: timepoint)

### Metrics computed

* mean Ct (`ct_mean`)
* standard deviation (`ct_std`)
* number of replicates

### Output

* `summary_df`

---

## 7. Normalization

Expression values are normalized using reference genes.

### Approach

* ΔCt-style normalization:

  ```
  normalized_expression = target_expression / reference_expression
  ```

* Optionally log-transformed:

  ```
  log2_normalized_expression
  ```

### Inputs

* `summary_df`
* reference targets (e.g. GBLP, RPL13)

### Output

* `analysis_df`

---

## 8. Statistical Analysis

### Model fitting

Linear models are fit separately per gene:

```
log2_normalized_expression ~ biological factors (+ optional plate effect)
```

Factors are automatically detected based on variability:

* sample_id
* group
* timepoint (if present)

### Outputs

* `model_terms_df`: ANOVA-style table
* `coefficient_df`: model coefficients

---

### Pairwise Comparisons

* Performed between relevant biological groups
* Welch’s t-test (unequal variance)
* Multiple testing correction (FDR, Benjamini-Hochberg)

### Output

* `pairwise_df`

---

### Interaction Effects

Interaction terms (e.g. sample_id × group) are extracted from the model:

* Used to assess whether responses differ between groups
* Stored in `model_terms_df`
* Can be visualized in single-gene plots

---

## 9. Visualization

### Plot types

#### 1. Expression grid

* Rows: genes
* Columns: conditions
* Shows:

  * raw replicate points
  * mean ± error
  * selected pairwise comparisons (within each panel)

#### 2. Time-course grid (optional)

* X-axis: timepoint
* Lines: biological groups

#### 3. Single gene plots

* Detailed visualization per gene
* Includes:

  * pairwise comparisons
  * optional interaction annotation

#### 4. Calibrator offset plots

* Visualize inter-plate correction quality

---

### Design principles for plotting

* Raw data is always visible
* Summary statistics are overlaid
* Statistical annotations are filtered for clarity
* No hard-coded biological assumptions

---

## 10. Export

All results are exported to an Excel file:

```
output/qpcr_results.xlsx
```

### Sheets include:

* machine_data
* plate_setup
* merged_data
* processed_data
* tech_rep_summary
* normalized_expression
* stats_model_terms
* stats_coefficients
* stats_pairwise
* stats_plot_summary
* QC reports
* calibrator data (if present)

Figures are saved in:

```
output/figures/
```

---

## Reproducibility

* All steps are executed from a single script: `run_pipeline.py`
* Intermediate DataFrames can be inspected at each stage
* No manual intervention is required once inputs are defined

---

## Notes

* The pipeline is designed to be flexible but assumes tidy input data
* Statistical interpretation should always consider biological context
* Plot aesthetics may require minor adjustments for publication

---
