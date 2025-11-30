# KPMG-1D: AI Workload Energy & Economic Index Analysis

## Project Overview

This project explores how much energy different AI workloads consume and how that relates to broader economic activity and corporate climate reporting. Using Anthropic’s ML.ENERGY benchmark as a starting point, we:

- Build a cleaned dataset of per-run energy measurements for LLM and diffusion workloads on A100 and H100 GPUs.
- Train a first-pass model that predicts whether a workload is **Low / Medium / High** energy intensity from configuration and runtime features.
- Set up data pipelines for the **Anthropic Economic Index** and for **corporate environmental reports** (Google and Microsoft), so that workload-level energy can eventually be connected to economic tasks and organizational emissions.

The repo currently contains a working modeling pipeline for ML.ENERGY and initial data-ingestion + data-quality workflows for the Economic Index and corporate carbon reports.

---

## Objectives & Goals

1. **Ingest and standardize key datasets**
   - ML workload–level energy data (Anthropic **ML.ENERGY** snapshot).
   - Task-level economic activity (Anthropic **Economic Index**).
   - Organization-level climate disclosure PDFs (Google and Microsoft 2024 reports).

2. **Build a baseline energy-intensity model**
   - Predict a **3-class energy label** (Low / Medium / High) from workload configuration and runtime metrics.
   - Quantify how much signal there is *before* using model names or vendor metadata.

3. **Characterize drivers of energy use**
   - Understand which features (batch size, latency, frames, steps, parallelism, throughput, etc.) are most associated with higher energy consumption.
   - Compare behavior across **NVIDIA A100** vs **H100** GPUs.

4. **Lay groundwork for cross-scale linkage**
   - Prepare data and scripts so future work can connect:
     - Workload-level energy → economic tasks → corporate emissions and reporting.

---

## Methodology

### Data Sources

- **ML.ENERGY benchmark (snapshot)**
  - Location: `data/mlenergy/raw/` + a curated subset embedded in `notebooks/llm-energy-output-modeling.ipynb`.
  - Contains per-run metrics for:
    - Diffusion / image-to-video models.
    - LLM text-generation workloads.
    - On **NVIDIA A100-SXM4-40GB** and **NVIDIA H100 80GB HBM3** GPUs.
  - Key columns (after parsing):
    - `Model`, `GPU`
    - `Energy/video (J)`, `Energy/image (J)`, `Energy/req (J)`
    - `Batch latency (s)`, `Batch size`, `Denoising steps`, `Frames`
    - `TP`, `PP`
    - `Avg TPOT (s)`, `Token tput (tok/s)`
    - `Avg Output Tokens`, `Avg BS (reqs)`, `Max BS (reqs)`

- **Anthropic Economic Index**
  - Fetched via `scripts/fetch_anthropic_econ_index.py`.
  - Stored in `data/anthropic_econ_index/processed/` as:
    - `anthropic_econ_index.{split}.parquet`
    - `anthropic_econ_index.{split}.csv`
  - Used primarily in the `anthropic-preprocessing-*.ipynb` notebooks for data quality checks and exploratory analysis of interaction/task types.

- **Corporate carbon disclosure PDFs**
  - Fetched via `scripts/fetch_corporate_carbon.py` into `data/corporate_carbon/raw/`.
  - Includes:
    - Google 2024 Environmental Report (PDF).
    - Microsoft 2024 Environmental Sustainability Report (PDF).
    - Microsoft 2024 Environmental Data Fact Sheet (PDF).
  - Currently downloaded and staged, but not yet parsed into structured tables in this repo.

### Preprocessing & Feature Engineering (ML.ENERGY)

1. **Parsing JSON into a DataFrame**
   - A large JSON-like text block of workload records is embedded in `llm-energy-output-modeling.ipynb`.
   - Records are extracted via regex and parsed with `json.loads` into a list of dicts, then converted to `mlenergy_df = pd.DataFrame(records)`.

2. **Missing values**
   - All missing entries are set to zero:
     ```python
     mlenergy_df = mlenergy_df.fillna(0)
     ```

3. **Unified energy metric**
   - Define a single `Energy (J)` per run as the maximum across the available energy fields:
     ```python
     mlenergy_df['Energy (J)'] = mlenergy_df[
         ['Energy/video (J)', 'Energy/image (J)', 'Energy/req (J)']
     ].max(axis=1)
     ```

4. **Log scaling & categorical labels**
   - Log-transform the energy:
     ```python
     mlenergy_df['Energy Log Scaled'] = np.log1p(mlenergy_df['Energy (J)'])
     ```
   - Use tertiles of the log-scaled energy to define **Low / Medium / High** classes:
     ```python
     mlenergy_df['Energy Output Label'] = pd.qcut(
         mlenergy_df['Energy Log Scaled'],
         q=3,
         labels=['Low', 'Medium', 'High']
     )
     ```

5. **Train/test split & features**
   - Drop direct energy columns and text fields; keep configuration/runtime features:
     ```python
     X = mlenergy_df.drop(columns=[
         'Energy (J)', 'Energy Log Scaled', 'Energy Output Label',
         'Energy/video (J)', 'Energy/image (J)', 'Energy/req (J)',
         'Model', 'GPU'
     ])
     y = mlenergy_df['Energy Output Label']

     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=42, stratify=y
     )
     ```

### Modeling

- **Classifier**
  - Multiclass logistic regression:
    ```python
    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    ```
- **Metrics**
  - Accuracy and confusion matrix:
    ```python
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=['Low', 'Medium', 'High'])
    ```

### EDA & Data Quality

- **Anthropic Economic Index**
  - Loaded in `anthropic-preprocessing-*.ipynb` and `data_quality.ipynb`.
  - Basic checks:
    - Shape, column types, `nunique()` per column.
    - Frequency tables over interaction/task types.
    - Distribution summaries (e.g., feedback loops, scores).

- **GPU-specific views (ML.ENERGY)**
  - Split into A100 and H100 subsets and:
    - Plot histograms of `Energy/req (J)` per GPU.
    - Plot correlation heatmaps of numeric features within each GPU subset.

---

## Results & Key Findings

### Dataset Summary (ML.ENERGY subset)

- Total runs: **431**
- GPUs:
  - **NVIDIA H100 80GB HBM3**: 221 runs
  - **NVIDIA A100-SXM4-40GB**: 210 runs

**Overall `Energy (J)` distribution:**

- Mean: ≈ **576 J**
- Std: ≈ **1887 J**
- Min: **5.6 J**
- 25th percentile: **45.9 J**
- Median: **119.0 J**
- 75th percentile: **440.2 J**
- Max: **16,916 J**

**Energy labels (tertiles of log-energy):**

- **Low**: 144 runs
- **Medium**: 143 runs
- **High**: 144 runs

By label:

| Label  | Mean Energy (J) | Median (J) | Max (J)     |
|--------|-----------------|------------|-------------|
| Low    | 28.5            | 23.2       | 54.9        |
| Medium | 123.5           | 119.0      | 245.8       |
| High   | 1574.1          | 711.5      | 16,915.9    |

High-energy workloads are roughly an **order of magnitude** more energy intensive than medium ones, and ~**50×** higher than low-energy runs on average.

### Classifier Performance

- **Task:** predict `Low` / `Medium` / `High` from configuration/runtime features only.
- **Model:** Logistic Regression (balanced, `liblinear`).
- **Test accuracy:** **~81.5%** on a 30% stratified hold-out.
- **Confusion matrix** (rows = true labels, cols = predicted):

|              | Pred: Low | Pred: Medium | Pred: High |
|--------------|-----------|--------------|------------|
| **True Low** | 39        | 4            | 0          |
| **True Medium** | 10     | 28           | 5          |
| **True High** | 0        | 5            | 39         |

- Low and High classes are classified quite cleanly.
- Most errors occur in the **Medium** band, which intuitively overlaps with both low and high workloads.

### Drivers of Energy Use

**Correlation with `Energy (J)` (selected numeric features):**

- Strong positive correlations:
  - `Batch latency (s)` (~0.81)
  - `Frames` (~0.72)
- Moderate positive:
  - `Denoising steps` (~0.33)
- Negative:
  - `Token tput (tok/s)` (~−0.31)
  - `PP` (pipeline parallelism) (~−0.26)
  - `Avg Output Tokens` (~−0.18)
  - `Avg BS (reqs)` and `Max BS (reqs)` (mildly negative)

Rough story:

- **Slow, frame-heavy, many-step workloads** tend to be high-energy.
- **Higher throughput and more parallelism** correlate with **lower energy** per run in this snapshot.
- Batch size and sequence length interact with these in non-trivial ways.

---

## Visualizations

The notebooks include several helpful visualizations:

- **Energy distributions**
  - Histograms of `Energy/req (J)` for:
    - A100-only workloads.
    - H100-only workloads.
  - Used to compare how request-level energy is distributed across GPU types.

- **Correlation heatmaps**
  - Seaborn heatmaps of numeric feature correlations for:
    - The full `mlenergy_df` (excluding labels and text fields).
    - H100-only and A100-only subsets.
  - Reveal clusters of strongly associated features (e.g., latency, frames, steps).

- **Energy label confusion matrix**
  - Matplotlib `ConfusionMatrixDisplay` for the 3-class classifier, showing where misclassifications concentrate (mainly around the Medium class).

- **Economic Index frequency plots**
  - Bar charts of interaction/task type frequencies in the Anthropic Economic Index, giving a high-level picture of the task distribution.

---

## Potential Next Steps

1. **Richer models & targets**
   - Move from logistic regression to tree-based or boosting methods (Random Forest, XGBoost, LightGBM) for non-linear patterns.
   - Try **regression** on `Energy (J)` directly, then bin into categories as needed.
   - Include `Model` and `GPU` as categorical features (one-hot or target encoding) to capture architecture-specific effects.

2. **Cross-dataset integration**
   - Link Economic Index task types to ML.ENERGY entries (where possible) to estimate **energy per economic task / interaction**.
   - Parse the corporate PDFs into structured data and associate energy figures with **scope 2/3 emissions**, enabling CO₂e estimates per workload.

3. **Scenario analysis**
   - Compare energy consumption across:
     - GPUs (A100 vs H100).
     - Model architectures and sizes.
     - Batch sizes and parallelism settings.
   - Simulate policies:
     - “What if all high-energy workloads moved from A100 to H100?”
     - “What if we cap denoising steps for certain deployment settings?”

4. **Productionization & documentation**
   - Turn scripts into a small CLI or pipeline for:
     - Downloading data.
     - Reproducing the modeling notebook.
   - Expand this README with:
     - Environment setup instructions.
     - Clear “How to run” steps for each notebook and script.

---

## Individual Contributions

- **Kai – ML.ENERGY Modeling & Analysis**
  - Parsed and cleaned the ML.ENERGY subset into `mlenergy_df`.
  - Engineered the `Energy (J)` metric and Low/Medium/High labels.
  - Built and evaluated the logistic regression classifier; generated confusion matrices and correlation heatmaps.

- **Jinso/Amanda – Economic Index Data & Baselines**
  - Wrote the preprocessing notebooks for the Anthropic Economic Index.
  - Performed data-quality checks and exploratory plots of interaction/task distributions.
  - Implemented or prototyped baseline models on the Economic Index (if applicable).

