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

---

## Quick start (TL;DR)

```bash
conda env create -f environment.yml
conda activate qpcr_env
python scripts/run_pipeline.py
```

---

## Requirements

Python ≥ 3.10 is recommended.

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate qpcr_env
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Prepare input data

You need:

* A qPCR machine output file (CSV)
* A plate setup file with metadata

The plate setup file should include columns such as:

* well
* target
* sample_id
* group
* bio_rep
* (optional) timepoint
* (optional) is_calibrator

---

### 2. Configure the pipeline

Edit:

```bash
scripts/run_pipeline.py
```

Update the file paths:

```python
DATA_FILE = Path("path/to/your/data.csv")
PLATE_SETUP_FILE = Path("path/to/your/plate_setup.csv")
```

You can also adjust:

* reference genes
* plotting settings
* calibration options

---

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py
```

---

## Outputs

### Excel results

```
output/qpcr_results.xlsx
```

Contains:

* processed data
* normalized expression
* statistical results
* QC reports

---

### Figures

```
output/figures/
```

Includes:

* expression grid plots
* time-course plots (if applicable)
* single-gene plots with statistical annotations
* calibrator diagnostics

---

## Workflow

A detailed explanation of the analysis steps is provided in:

```
workflow.md
```

## Example Data

Example input files are provided in:

tests/data/

These can be used to run the pipeline directly.

---

## Project Structure

```
qpcr/
    analysis.py
    statistics.py
    plotting.py
    ...

scripts/
    run_pipeline.py

workflow.md
requirements.txt
environment.yml
```

---

## Notes

* The pipeline is designed to be flexible and data-driven
* No biological assumptions are hard-coded
* Statistical results should always be interpreted in biological context

---

## Future Work

Planned future improvements to this pipeline include:

* Improving the plotting system, particularly:

  * more robust and readable significance annotations
  * better handling of complex experimental designs
  * refinement of layout and aesthetics for publication-ready figures

* Implementing support for standard curves to enable:

  * absolute quantification
  * efficiency correction
  * improved comparison across experiments

These additions will further increase the flexibility and applicability of the pipeline for a broader range of qPCR analyses.

---

## Author

**Wiebke Weber**
PhD Candidate, Uppsala University

This project was developed as part of my PhD work.
The overall design, structure, and analytical strategy were led by me.
Implementation support and iterative improvements were developed in collaboration with ChatGPT, whose suggestions I critically evaluated and refined throughout the project.
