# Algorithmic Fairness: Bias Measurement and Mitigation with AIF360

This repository contains a Jupyter notebook exploring algorithmic fairness
using IBM's [AIF360](https://github.com/Trusted-AI/AIF360) (AI Fairness 360)
toolkit. Using the German Credit Data dataset, the notebook measures bias
across sensitive attributes and applies a pre-processing bias mitigation
algorithm (Learning Fair Representations) to reduce that bias.

## Overview

The German Credit Data dataset classifies loan applicants as good or bad
credit risks. This notebook treats **sex** and **age** as protected
attributes and investigates whether the dataset (and downstream classifiers
trained on it) treat privileged and unprivileged groups differently — a
central concern in algorithmic fairness.

## Contents

| Section | Description |
|---|---|
| **Loading the German Credit Data** | Loads the dataset via `aif360.datasets.GermanDataset`, converts it to a pandas DataFrame, and remaps the credit label (`2.0` → `0`) |
| **Defining privileged/unprivileged groups** | Specifies `{'sex': 1, 'age': 1}` as the privileged group and `{'sex': 0, 'age': 0}` as the unprivileged group |
| **Dataset-level fairness metrics** | Computes **Disparate Impact**, **Statistical Parity Difference**, and the individual fairness metric **Consistency** directly from the original dataset labels, before any classifier is involved |
| **Baseline classifier** | Trains a `LogisticRegression` model on a stratified train/test split of the (unmitigated) data and evaluates it with a confusion matrix |
| **Bias mitigation — Learning Fair Representations (LFR)** | Applies AIF360's `LFR` pre-processing algorithm, which learns a fair intermediate representation of the data by jointly optimizing for reconstruction quality, fairness (statistical parity), and prediction accuracy, then transforms the training data |
| **Post-mitigation fairness metrics** | Recomputes Disparate Impact, Statistical Parity Difference, and Consistency on the LFR-transformed training data to assess how much bias was removed |
| **Setup notes** | Documents how to install AIF360 and manually place the German Credit dataset file in AIF360's expected data directory (`aif360/data/raw/german/`), since the raw data is not bundled with the library |

## Key findings

- On the **original, unmitigated dataset**, the fairness metrics show clear
  bias against the unprivileged group:
  - **Disparate Impact = 0.748** (values below 0.8 typically indicate a
    fairness concern by the commonly used 80% rule)
  - **Statistical Parity Difference = -0.186** (the unprivileged group
    receives the favorable outcome at a meaningfully lower rate)
  - **Consistency = 0.682** (moderate individual-level inconsistency —
    similar individuals aren't always treated similarly)
- After applying **Learning Fair Representations (LFR)** to the training
  data, all three metrics move to their ideal values:
  - **Disparate Impact = 1.000**
  - **Statistical Parity Difference = 0.000**
  - **Consistency = 1.000**
- This demonstrates LFR's ability to transform the training data into a
  representation that satisfies group and individual fairness criteria,
  though 206 out of the training labels were altered in the process — a
  reminder that fairness interventions typically involve a trade-off with
  fidelity to the original data/labels.

## Requirements

This project uses Python with the following packages:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib aif360
```

- `numpy` / `pandas` — data handling
- `scikit-learn` — `LogisticRegression`, train/test splitting, `StandardScaler`, and evaluation via confusion matrix
- `imbalanced-learn` (`imblearn`) — SMOTE (imported for related coursework; not applied to the fairness pipeline itself)
- `aif360` — fairness metrics (`BinaryLabelDatasetMetric`) and bias mitigation algorithms (`LFR`), plus the built-in `GermanDataset` loader

**Important setup step:** `aif360` does not bundle the German Credit dataset
due to licensing. Before running the notebook, download `german.data` from
the [UCI Statlog (German Credit Data) repository](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)
and place it in:
```
<site-packages>/aif360/data/raw/german/german.data
```
(e.g. `/usr/local/lib/python3.13/dist-packages/aif360/data/raw/german/` in a
typical Google Colab session). This step must be repeated each time a fresh
Colab runtime is started, since installed-package directories don't persist
across sessions.

## Repository structure

```
.
├── algorithmic_fairness.ipynb    # Jupyter notebook source
├── algorithmic_fairness.pdf       # Rendered PDF export of the notebook
└── README.md
```

## Usage

Open `algorithmic_fairness.ipynb` in Jupyter or Google Colab, ensure the
German Credit dataset is placed as described above, then run all cells top
to bottom. The notebook was authored and exported from Google Colab, using
`nbconvert` and `xelatex` to produce the accompanying PDF.
