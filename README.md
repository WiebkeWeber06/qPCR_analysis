# qPCR Analysis Pipeline

A modular and reproducible Python pipeline for analyzing qPCR data, including:

* preprocessing and QC
* technical replicate summarization
* normalization using reference genes
* optional inter-plate calibration
* statistical testing
* automated plotting and reporting

---

## Overview

This project provides a structured workflow for qPCR data analysis starting from raw instrument output and plate setup files, through normalization and statistical analysis, to final figures and Excel reports.

The goal is to move away from one-off analysis scripts toward a reusable and transparent pipeline.

This project was developed in collaboration with an AI assistant (ChatGPT).
The overall design, structure, and analytical decisions were led by the author, while code implementation was assisted by AI. All outputs were critically evaluated, adapted, and validated to ensure correctness, robustness, and biological relevance.

---

## Features

* ✔️ Import of Bio-Rad qPCR output files
* ✔️ Plate design integration
* ✔️ Quality control (QC) reporting
* ✔️ Technical replicate summarization
* ✔️ Reference gene normalization (ΔCt / ΔΔCt-style)
* ✔️ Optional inter-plate calibration using calibrators
* ✔️ Linear-model-based statistical analysis
* ✔️ Pairwise comparisons with multiple testing correction
* ✔️ Flexible plotting (grid, single gene, time course)
* ✔️ Export of all results to Excel

---

## Project Structure

```
qpcr/
├── analysis.py        # normalization logic
├── calibration.py    # inter-plate calibration
├── io.py             # data loading and merging
├── plotting.py       # visualization functions
├── preprocess.py     # filtering and data cleaning
├── qc.py             # quality control reports
├── schema.py         # dataframe validation
├── statistics.py     # modeling and statistical tests
```

```
scripts/
└── run_pipeline.py   # main analysis script
```

```
tests/data/
├── raw qPCR files
└── plate setup files
```

---

## How to Run

1. Activate your environment:

```bash
conda activate qpcr_env
```

2. Run the pipeline:

```bash
python scripts/run_pipeline.py
```

---

## Input Data

The pipeline requires:

### 1. qPCR machine output

* Example: Bio-Rad CSV export

### 2. Plate setup file

* Contains metadata per well:

  * sample_id
  * group (e.g. condition)
  * target gene
  * replicate info
  * optional calibrator flag

---

## Output

All results are saved in:

```
output/
├── qpcr_results.xlsx
└── figures/
```

### Excel file contains:

* raw data
* processed data
* technical replicate summaries
* normalized expression values
* statistical results
* QC reports

### Figures include:

* expression grid plots
* time-course plots (if applicable)
* single gene plots with statistics
* calibrator offset plots (if used)

---

## Statistical Analysis

The pipeline uses linear models of the form:

```
expression ~ biological factors (+ optional plate effect)
```

* Automatically detects relevant factors (e.g. group, sample_id, timepoint)
* Supports interaction terms
* Performs:

  * model fitting per gene
  * pairwise comparisons
  * multiple testing correction (FDR)

---

## Design Principles

* Reproducibility over ad hoc analysis
* Clear separation of steps (IO, preprocessing, analysis, plotting)
* Flexible handling of experimental designs
* Minimal hard-coded biological assumptions
* Transparent and inspectable outputs

---

## Notes

* Plotting is designed for exploratory and publication-ready visualization,
  but may still require minor layout adjustments depending on the dataset.
* Statistical interpretation should always be considered in the biological context.

---

## Future Improvements

* Improved automatic plot annotation layout
* Enhanced support for complex experimental designs
* CLI interface for easier usage
* Packaging for pip installation

---

## Author

Wiebke
PhD project — qPCR data analysis and pipeline development

