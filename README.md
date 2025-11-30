# KPMG-1D: AI Workload Energy & Economic Index Analysis

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives & Goals](#objectives--goals)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Create and Activate a Virtual Environment](#create-and-activate-a-virtual-environment)
  - [Install Python Dependencies](#install-python-dependencies)
  - [Set Up Data Sources](#set-up-data-sources)
- [How to Run the Project](#how-to-run-the-project)
  - [1. Train the Energy-Intensity Model](#1-train-the-energy-intensity-model)
  - [2. Evaluate Model Performance](#2-evaluate-model-performance)
  - [3. Explore the Anthropic Economic Index](#3-explore-the-anthropic-economic-index)
- [Methodology](#methodology)
  - [Data Sources](#data-sources)
  - [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
  - [Modeling](#modeling)
  - [EDA & Data Quality](#eda--data-quality)
- [Results & Key Findings](#results--key-findings)
- [User Guides](#user-guides)
  - [Energy Modeling Workflow](#energy-modeling-workflow)
  - [Economic Index Workflow](#economic-index-workflow)
- [API / CLI Documentation](#api--cli-documentation)
  - [`scripts/export_mlenergy_snapshot.py`](#scriptsexport_mlenergy_snapshotpy)
  - [`scripts/fetch_anthropic_econ_index.py`](#scriptsfetch_anthropic_econ_indexpy)
  - [`scripts/fetch_corporate_carbon.py`](#scriptsfetch_corporate_carbonpy)
- [Potential Next Steps](#potential-next-steps)
- [Individual Contributions](#individual-contributions)

---

## Project Overview

This project explores how much energy different AI workloads consume and how that relates to broader economic activity and corporate climate reporting. Using Anthropic’s **ML.ENERGY** benchmark as a starting point, we:

- Build a cleaned dataset of per-run energy measurements for LLM and diffusion workloads on **A100** and **H100** GPUs.
- Train a first-pass model that predicts whether a workload is **Low / Medium / High** energy intensity based on configuration and runtime features.
- Set up data pipelines for the **Anthropic Economic Index** and **corporate environmental reports** (Google and Microsoft), so that workload-level energy can eventually be connected to economic tasks and organizational emissions.

The repo currently contains:

- A working modeling pipeline for ML.ENERGY (`notebooks/llm-energy-output-modeling.ipynb`).
- Initial ingestion and data-quality workflows for:
  - Anthropic Economic Index.
  - Corporate carbon disclosure PDFs.

---

## Objectives & Goals

1. **Ingest and standardize key datasets**
   - ML workload–level energy data (Anthropic **ML.ENERGY** snapshot).
   - Task-level economic activity (Anthropic **Economic Index**).
   - Organization-level climate disclosure PDFs (Google & Microsoft 2024).

2. **Build a baseline energy-intensity model**
   - Predict a **3-class energy label** (Low / Medium / High) from workload configuration and runtime metrics.
   - Measure how much predictive signal there is without using model names or GPU labels.

3. **Characterize drivers of energy use**
   - Understand which features (batch size, latency, frames, steps, parallelism, throughput, etc.) are most associated with higher energy consumption.
   - Compare behavior across **NVIDIA A100** vs **H100** GPUs.

4. **Lay the groundwork for cross-scale linkage**
   - Prepare data and scripts so future work can connect:
     - Workload-level energy → economic tasks → corporate emissions and reporting.

---

## Installation & Setup

### Prerequisites

- **Python**: 3.9 or newer.
- **Git** (for pulling the ML.ENERGY submodule).
- Optional but recommended:
  - `conda` or `venv` for virtual environments.
  - `jupyter` or `jupyterlab` for running notebooks.

### Clone the Repository

```bash
git clone --recurse-submodules https://github.com/<your-org-or-username>/KPMG-1D-repo.git
cd KPMG-1D-repo
```

> If you already cloned without `--recurse-submodules`, run:
> ```bash
> git submodule update --init --recursive
> ```

This fetches the `external/mlenergy` submodule (ML.ENERGY leaderboard repo).

### Create and Activate a Virtual Environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate    # On macOS/Linux
# .venv\Scripts\activate     # On Windows (PowerShell or CMD)
```

Using `conda` (alternative):

```bash
conda create -n kpmg-1d python=3.10
conda activate kpmg-1d
```

### Install Python Dependencies

There is no `requirements.txt` yet, but you can install the required packages with:

```bash
pip install \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  scikit-learn \
  datasets \
  jupyter
```

> If you want, you can also create your own `requirements.txt`:

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
datasets
jupyter
```

then install via:

```bash
pip install -r requirements.txt
```

### Set Up Data Sources

All data is stored under the `data/` directory, created by the scripts below.

#### 1. Export ML.ENERGY Snapshot

This copies structured ML.ENERGY data from the `external/mlenergy` submodule into `data/mlenergy/raw/`:

```bash
python scripts/export_mlenergy_snapshot.py
```

- Input: `external/mlenergy/data` (from the submodule).
- Output:
  - JSON/CSV/JSONL files in `data/mlenergy/raw/`.
  - A provenance file `data/mlenergy/README.md` with the submodule commit hash.

#### 2. Fetch Anthropic Economic Index

This downloads the **Anthropic/EconomicIndex** dataset from Hugging Face and stores it as CSV + Parquet:

```bash
python scripts/fetch_anthropic_econ_index.py
```

- Output directory: `data/anthropic_econ_index/processed/`
  - `anthropic_econ_index.train.parquet` / `.csv`
  - `anthropic_econ_index.validation.parquet` / `.csv`
  - `anthropic_econ_index.test.parquet` / `.csv` (if present)

#### 3. Fetch Corporate Carbon PDFs

This downloads selected corporate environmental reports (Google and Microsoft) to the repo:

```bash
python scripts/fetch_corporate_carbon.py
```

- Output directory: `data/corporate_carbon/raw/`
  - `google_2024_environmental_report.pdf`
  - `microsoft_2024_environmental_sustainability_report.pdf`
  - `microsoft_2024_env_data_fact_sheet.pdf`

> These PDFs are *not yet* parsed into structured tables in this repo, but are staged for future use.

---

## How to Run the Project

### 1. Train the Energy-Intensity Model

The main modeling pipeline lives in:

- `notebooks/llm-energy-output-modeling.ipynb`

**Step-by-step:**

1. Make sure your environment is activated and dependencies installed.
2. Launch Jupyter:

   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

3. In the Jupyter UI, open:
   - `notebooks/llm-energy-output-modeling.ipynb`

4. Run the notebook **top to bottom**:
   - The notebook:
     - Parses ML.ENERGY data into a flat `pandas` DataFrame (`mlenergy_df`).
     - Engineers a unified `Energy (J)` target and log-scaled version.
     - Creates a 3-class label: `Low`, `Medium`, `High`.
     - Trains a **logistic regression** classifier.
     - Reports accuracy and confusion matrix.

The model will be trained in-memory (no model checkpoint is written to disk by default, but you can add that if desired).

### 2. Evaluate Model Performance

The evaluation is integrated into the same notebook:

- **Train–test split**:

  ```python
  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
  )
  ```

- **Model training and prediction**:

  ```python
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, confusion_matrix

  model = LogisticRegression(
      max_iter=1000,
      solver='liblinear',
      class_weight='balanced'
  )
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  ```

- **Metrics and confusion matrix** (already computed and displayed in the notebook):

  ```python
  acc = accuracy_score(y_test, pred)
  cm = confusion_matrix(y_test, pred, labels=['Low', 'Medium', 'High'])
  ```

You can re-run these cells to reproduce the ~**81.5%** test accuracy and see the confusion matrix.

### 3. Explore the Anthropic Economic Index

The Economic Index analysis lives in:

- `notebooks/anthropic-preprocessing-json-file-combining-in-d.ipynb`
- `notebooks/anthropic-preprocessing-json-file-combining-in-d (1).ipynb`
- `notebooks/anthropic-preprocessing-json-file-combining-in-d (2).ipynb`
- `notebooks/data_quality.ipynb`

Typical workflow:

1. Open `notebooks/data_quality.ipynb`.
2. Run the cells to:
   - Load the processed Economic Index CSV/Parquet files (from `data/anthropic_econ_index/processed/`).
   - Inspect basic dataset properties: `df.shape`, `df.nunique()`, etc.
   - Generate frequency plots of task / interaction types.

3. Open one of the `anthropic-preprocessing-*.ipynb` notebooks for more advanced EDA or baseline modeling (if present).

---

## Methodology

### Data Sources

**ML.ENERGY benchmark (snapshot)**

- Source: `external/mlenergy/` submodule → `data/mlenergy/raw/`.
- Contains per-run metrics for:
  - Diffusion / image-to-video models.
  - LLM text-generation workloads.
- GPUs:
  - **NVIDIA A100-SXM4-40GB**
  - **NVIDIA H100 80GB HBM3**
- Key columns (after parsing in the notebook):
  - `Model`, `GPU`
  - `Energy/video (J)`, `Energy/image (J)`, `Energy/req (J)`
  - `Batch latency (s)`, `Batch size`, `Denoising steps`, `Frames`
  - `TP`, `PP`
  - `Avg TPOT (s)`, `Token tput (tok/s)`
  - `Avg Output Tokens`, `Avg BS (reqs)`, `Max BS (reqs)`

**Anthropic Economic Index**

- Fetched from Hugging Face with `datasets.load_dataset("Anthropic/EconomicIndex")`.
- Stored as CSV + Parquet under `data/anthropic_econ_index/processed/`.
- Used for:
  - Data-quality checks.
  - Frequency tables of interaction/task categories.
  - Potential baseline models (depending on the notebook).

**Corporate Carbon Disclosure PDFs**

- Downloaded via `scripts/fetch_corporate_carbon.py`.
- Stored under `data/corporate_carbon/raw/`.
- Currently not parsed into structured tables in this repo.

### Preprocessing & Feature Engineering

In `llm-energy-output-modeling.ipynb`:

1. **Parse ML.ENERGY data** into a `pandas` DataFrame (`mlenergy_df`).
2. **Handle missing values**:
   ```python
   mlenergy_df = mlenergy_df.fillna(0)
   ```
3. **Define a unified energy target**:
   ```python
   mlenergy_df['Energy (J)'] = mlenergy_df[
       ['Energy/video (J)', 'Energy/image (J)', 'Energy/req (J)']
   ].max(axis=1)
   ```
4. **Log-scale the energy**:
   ```python
   mlenergy_df['Energy Log Scaled'] = np.log1p(mlenergy_df['Energy (J)'])
   ```
5. **Create categorical labels using tertiles of log-energy**:
   ```python
   mlenergy_df['Energy Output Label'] = pd.qcut(
       mlenergy_df['Energy Log Scaled'],
       q=3,
       labels=['Low', 'Medium', 'High']
   )
   ```
6. **Feature selection**:
   - Drop direct energy and text fields; keep numeric configuration/runtime columns:
     ```python
     X = mlenergy_df.drop(columns=[
         'Energy (J)', 'Energy Log Scaled', 'Energy Output Label',
         'Energy/video (J)', 'Energy/image (J)', 'Energy/req (J)',
         'Model', 'GPU'
     ])
     y = mlenergy_df['Energy Output Label']
     ```

### Modeling

- **Model**: Multiclass logistic regression (one-vs-rest, balanced class weights).
- **Split**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
  )
  ```
- **Training**:
  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(
      max_iter=1000,
      solver='liblinear',
      class_weight='balanced'
  )
  model.fit(X_train, y_train)
  ```
- **Evaluation**:
  ```python
  from sklearn.metrics import accuracy_score, confusion_matrix
  pred = model.predict(X_test)
  acc = accuracy_score(y_test, pred)
  cm = confusion_matrix(y_test, pred, labels=['Low', 'Medium', 'High'])
  ```

### EDA & Data Quality

- **Anthropic Economic Index**:
  - Check missingness, cardinality, and distributions.
  - Create bar plots of interaction/task type frequencies.

- **GPU-specific views (ML.ENERGY)**:
  - Subset to H100 and A100 workloads.
  - Plot histograms of `Energy/req (J)` per GPU.
  - Draw correlation heatmaps of numeric features within each GPU subset.

---

## Results & Key Findings

### Dataset Summary (ML.ENERGY subset)

- Total runs: **431**
- GPUs:
  - H100: **221** runs
  - A100: **210** runs

**`Energy (J)` distribution (overall):**

- Mean: ≈ **576 J**
- Std: ≈ **1887 J**
- Min: **5.6 J**
- Median: **119.0 J**
- Max: **16,916 J**

**Label distribution (Low / Medium / High):**

- Low: **144** runs  
- Medium: **143** runs  
- High: **144** runs  

By label:

| Label  | Mean Energy (J) | Median (J) | Max (J)     |
|--------|-----------------|------------|-------------|
| Low    | 28.5            | 23.2       | 54.9        |
| Medium | 123.5           | 119.0      | 245.8       |
| High   | 1574.1          | 711.5      | 16,915.9    |

High-energy workloads are roughly an order of magnitude more energy-intensive than medium ones and ~50× more than low-energy runs on average.

### Classifier Performance

- **Task:** Predict `Low` / `Medium` / `High` from configuration/runtime features only.
- **Model:** Logistic Regression (balanced).
- **Test accuracy:** ~**81.5%** on a 30% stratified hold-out.

Confusion matrix (rows = true labels, cols = predicted):

|              | Pred: Low | Pred: Medium | Pred: High |
|--------------|-----------|--------------|------------|
| **True Low**    | 39        | 4            | 0          |
| **True Medium** | 10        | 28           | 5          |
| **True High**   | 0         | 5            | 39         |

Most errors occur in the **Medium** band, which overlaps with both low and high workloads.

### Drivers of Energy Use (Correlations)

Selected correlations with `Energy (J)`:

- Strong positive:
  - `Batch latency (s)` (~0.81)
  - `Frames` (~0.72)
- Moderate positive:
  - `Denoising steps` (~0.33)
- Negative:
  - `Token tput (tok/s)` (~−0.31)
  - `PP` (pipeline parallelism) (~−0.26)
  - `Avg Output Tokens` (~−0.18)

**Interpretation:**

- Slow, frame-heavy, many-step workloads are high-energy.
- Higher throughput and more parallelism correlate with lower energy per run in this snapshot.

---

## User Guides

### Energy Modeling Workflow

1. **Data preparation**
   - Run `python scripts/export_mlenergy_snapshot.py`.
   - Optionally verify that `data/mlenergy/raw/` contains structured energy files.

2. **Modeling**
   - Open and run `notebooks/llm-energy-output-modeling.ipynb`.
   - Inspect:
     - The summary statistics of `Energy (J)`.
     - The `Energy Output Label` distribution.
     - Correlation heatmaps.

3. **Evaluation**
   - View the reported accuracy and confusion matrix cells.
   - Experiment with changing:
     - Feature subsets.
     - Train–test split ratio.
     - Logistic regression hyperparameters.

4. **What you can replicate**
   - Reproduce energy labels and distributions.
   - Reproduce the ~81.5% accuracy logistic regression baseline.
   - Compare A100 vs H100 workloads on energy distributions.

### Economic Index Workflow

1. **Fetch data**
   - Run:
     ```bash
     python scripts/fetch_anthropic_econ_index.py
     ```

2. **Open notebooks**
   - Start Jupyter and open:
     - `notebooks/data_quality.ipynb`
     - `notebooks/anthropic-preprocessing-json-file-combining-in-d*.ipynb`

3. **Run EDA cells**
   - Inspect:
     - Dataset sizes, column names, and types.
     - Value counts for key categorical features.
   - Plot:
     - Bar charts of interaction types.
     - Any baseline model performance if defined in the notebook.

4. **Future extension**
   - Add your own models to predict labels in the Economic Index dataset.
   - Connect task categories to energy estimates from ML.ENERGY.

---

## API / CLI Documentation

This repo does not expose a formal Python package API yet, but it **does** provide a small CLI-style interface via `scripts/`.

### `scripts/export_mlenergy_snapshot.py`

**Purpose:**

- Copy structured ML.ENERGY data from the `external/mlenergy` submodule into `data/mlenergy/raw/` and record the submodule commit hash for provenance.

**Usage:**

```bash
python scripts/export_mlenergy_snapshot.py
```

**Behavior:**

- Expects `external/mlenergy/data` to exist (from the submodule).
- Recursively scans for `.csv`, `.json`, `.jsonl` files.
- Copies them into `data/mlenergy/raw/` preserving relative paths.
- Writes `data/mlenergy/README.md` with the source commit SHA.

---

### `scripts/fetch_anthropic_econ_index.py`

**Purpose:**

- Download the Anthropic Economic Index dataset from Hugging Face and save it locally in CSV and Parquet formats.

**Usage:**

```bash
python scripts/fetch_anthropic_econ_index.py
```

**Behavior:**

- Uses `datasets.load_dataset("Anthropic/EconomicIndex")`.
- For each split (e.g., `train`, `validation`, `test`):
  - Writes `anthropic_econ_index.<split>.parquet`.
  - Writes `anthropic_econ_index.<split>.csv`.
- Output directory: `data/anthropic_econ_index/processed/`.

---

### `scripts/fetch_corporate_carbon.py`

**Purpose:**

- Download selected corporate environmental/climate reports from public URLs (Google, Microsoft) and store them locally.

**Usage:**

```bash
python scripts/fetch_corporate_carbon.py
```

**Behavior:**

- Creates `data/corporate_carbon/raw/` if needed.
- Downloads:
  - Google 2024 Environmental Report (PDF).
  - Microsoft 2024 Environmental Sustainability Report (PDF).
  - Microsoft 2024 Environmental Data Fact Sheet (PDF).
- Skips re-download if the files already exist.

---

## Potential Next Steps

- **Modeling**
  - Upgrade to tree-based or boosting models (Random Forest, XGBoost, LightGBM).
  - Switch from classification to regression on `Energy (J)`.

- **Data fusion**
  - Align Economic Index tasks with ML.ENERGY workloads where possible.
  - Parse corporate PDFs into tables and link energy to emissions (CO₂e).

- **Tooling**
  - Factor the modeling notebook into reusable Python modules.
  - Add a `requirements.txt` or `pyproject.toml` and simple CLI (`python -m kpmg1d.train`).

---

## Individual Contributions

> Customize this section with your own names and roles.

- **Kai – ML.ENERGY Modeling & Analysis**
  - Parsed and cleaned ML.ENERGY into `mlenergy_df`.
  - Engineered `Energy (J)` and the Low/Medium/High labels.
  - Built and evaluated the logistic regression baseline.

- **Amanda – Economic Index Data & Baselines**
  - Implemented the Economic Index preprocessing and data-quality notebooks.
  - Performed frequency and distributional analysis over interaction/task types.
  - Prototyped baseline models on Economic Index data (where applicable).

- **Masumi – Data Pipelines & Corporate Carbon Reports**
  - Wrote `fetch_anthropic_econ_index.py`, `export_mlenergy_snapshot.py`, and/or `fetch_corporate_carbon.py`.
  - Organized the `data/` directory structure and documented ML.ENERGY provenance.


---
