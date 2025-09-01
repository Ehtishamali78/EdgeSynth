# Scripts Folder

This folder contains the core Python scripts that make up the **EdgeSynth pipeline** for generating and evaluating synthetic SCADA data.

---

## Files

- **`prepare_promptsV1.py`**  
  Early prototype for preparing prompts from the raw wind turbine dataset. Performs chunked reading, filtering (operational rows only), and outputs structured text prompts.  
  *Kept for reference; superseded by* `real_sample.py` + `prepare_prompts.py`.

- **`real_sample.py`**  
  Samples a subset of real SCADA data from the large raw file (`wind_turbine.csv`).  
  - Filters operational data (`PowerOutput > 0`, `WindSpeed > 2`).  
  - Samples to match the synthetic dataset size (≈17.5k rows).  
  - Produces `real_sample_final.csv`.

- **`prepare_prompts.py`**  
  Main prompt-prep script.  
  - Reads `batched_prompts.csv` created by `prepare_prompts_v1.py`.  
  - Uses Hugging Face GPT-2 pipeline to generate synthetic SCADA logs.  
  - Saves raw generated text to `gpt2_raw_output_week6.txt`.

- **`postprocess.py`**  
  Cleans raw GPT-2 outputs into structured tabular form.  
  - Parses SCADA log entries.  
  - Clips unrealistic values to real turbine ranges.  
  - Normalizes timestamps.  
  - Outputs `synthetic_scada_cleaned_final.csv`.

- **`evaluate.py`**  
  Quick-look QA for synthetic data only.  
  - Histograms (WindSpeed, Power, GeneratorTemp).  
  - Scatter plots (WindSpeed vs. Power, RotorSpeed vs. GeneratorSpeed).  
  - Boxplots for outlier inspection.  

- **`week4.py`**  
  Comprehensive evaluation comparing real vs. synthetic datasets.  
  - KL divergence per feature.  
  - KDE distribution overlays (saved in `results/figures/`).  
  - Correlation heatmaps (real vs. synthetic).  
  - Random Forest regression parity tests (saved in `results/metrics/`).  

- **`../app/app_streamlit.py`**  
  Interactive **EdgeSynth Explorer** prototype.  
  - Dataset preview and summary stats.  
  - KDE overlays, correlation heatmaps.  
  - Privacy heuristic (nearest-neighbor distance).  

---

## Typical Workflow

1. **Sample real data**  
   ```bash
   python scripts/real_sample.py
→ produces data/real_sample_final.csv.

2. **Prepare prompts (optional early step)**
   ```bash
   python scripts/prepare_promptsV1.py
→ produces data/batched_prompts.csv.

3. **Generate synthetic logs with GPT-2**
   ```bash
   python scripts/prepare_prompts.py
→ produces data/gpt2_raw_output_week6.txt.

4. **Post-process generated logs**
   ```bash
   python scripts/postprocess.py
→ produces data/synthetic_scada_cleaned_final.csv.

5. **Evaluate results**

Quick QA: python scripts/evaluate.py
Full analysis: python scripts/week4.py

6. **Explore interactively**
    ```bash
   streamlit run app/app_streamlit.py

Notes

Large raw dataset (wind_turbine.csv) is not committed to GitHub.

Outputs (synthetic_scada_cleaned_final.csv, figures, metrics) are stored under data/ or results/.

Scripts are modular — you can run them individually or chain them as needed.


---

✅ This version:  
- Clarifies **which script is for legacy vs. final use**.  
- Aligns filenames with outputs in your repo (`synthetic_scada_cleaned_final.csv`, `real_sample_final.csv`).  
- Explains *where figures/metrics go* (`results/`).  
- Has a **clean step-by-step pipeline** you can drop directly in GitHub.  

Would you like me to also draft a **`README.md` for the whole repository root** (with project overview + folder structure) so your repo looks professional when published?
