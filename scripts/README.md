# Scripts Folder

This folder contains all scripts used in the EdgeSynth pipeline.

## Files

- `prepare_prompts.py`  
  Generates structured GPT-2 prompts from SCADA CSV data (chunked reading, filtering, sampling).

- `prepare_prompts_v1.py`  
  Early version of prompt-prep (kept for reproducibility reference). Use `prepare_prompts.py` as the main one.

- `real_sample.py`  
  Preprocesses and samples real SCADA data for evaluation and comparison.

- `postprocess.py`  
  Parses GPT-2 raw outputs → structured tabular format. Cleans, clips unrealistic values, removes malformed/duplicate rows.

- `evaluate.py`  
  Synthetic-only evaluation. Plots histograms, scatter plots, boxplots for QA.

- `week4.py`  
  Main evaluation pipeline: distribution overlays (KDE), KS tests, correlation heatmaps, Random Forest regression tests.

- `app_streamlit.py` (in `../app/`)  
  Interactive visualization tool for exploring synthetic vs. real datasets.

## Usage

Scripts are modular. Typical pipeline:

1. Run `prepare_prompts.py` → `batched_prompts.csv`  
2. Generate synthetic data with GPT-2 (external generation step).  
3. Run `postprocess.py` → `synthetic_scada_cleaned.csv`  
4. Evaluate with `week4.py` or `evaluate.py`  
5. Explore interactively with `app/app_streamlit.py`
