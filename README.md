# EdgeSynth: Leveraging Generative AI for Synthetic Edge Data

EdgeSynth is a framework that uses **transformer-based generative AI (GPT-2)** to synthesize realistic SCADA sensor data for **IoT and edge computing** scenarios.  
It addresses two core challenges in edge AI:
- **Data scarcity** → limited training samples at the device level.  
- **Privacy concerns** → risks of exposing raw operational logs.  

By generating high-fidelity synthetic datasets, EdgeSynth enables **data augmentation, privacy-preserving sharing, and robust model training** at the network edge.

---

## 🚀 Features
- GPT-2–based **synthetic SCADA log generation** with prompt engineering.
- Post-processing pipeline for cleaning, scaling, and operational plausibility.
- Evaluation suite:  
  - KDE distribution overlays  
  - Correlation heatmaps  
  - KS statistical tests  
  - ML downstream utility tests (Random Forest regression).
- **Streamlit-based prototype app** for dataset preview, distribution inspection, and privacy heuristics.
- Lightweight deployment feasible on edge gateways with optimizations.

---

## 📂 Repository Structure

```plaintext
EdgeSynth/
├── data/
│   ├── batched_prompts.csv           # Prompts generated for GPT-2
│   ├── synthetic_scada_cleaned.csv   # Final cleaned synthetic dataset
│   └── README.md
│
├── scripts/
│   ├── prepare_prompts.py            # Prompt creation from raw CSV (final)
│   ├── prepare_prompts_v1.py         # Early variant (kept for reference)
│   ├── real_sample.py                # Real-data preprocessing/sampling
│   ├── postprocess.py                # Parse/clean GPT-2 outputs → tabular
│   ├── evaluate.py                   # Synthetic-only QA plots (KDE, scatter, box)
│   ├── week4.py                      # Main evaluation: KDE, KS, corr, RF regression
│   └── README.md
│
├── app/
│   ├── app_streamlit.py              # “EdgeSynth Explorer” prototype
│   └── README.md
│
├── results/
│   ├── figures/                      # Saved plots (KDEs, heatmaps, screenshots, etc.)
│   └── README.md
│
├── docs/
│   ├── report.pdf                    # IEEE paper (optional placeholder)
│   ├── supporting_material.pdf       # Supplement (optional placeholder)
│   ├── presentation.pptx             # Slides (optional placeholder)
│   └── README.md
│
├── env/
│   └── requirements.txt              # Python dependencies
│
└── README.md                         # Project overview (this file)



---



\## ⚙️ Setup \& Installation



\### 1. Clone repository

```bash

git clone https://github.com/Ehtishamali78/EdgeSynth.git 

cd EdgeSynth



2\. Create environment \& install dependencies

python -m venv env

source env/bin/activate   # (Linux/Mac)

env\\Scripts\\activate      # (Windows)



pip install -r env/requirements.txt



## ▶️ Usage

Run the pipeline step by step:

**Step 1 – Sample real data**  
```bash
python scripts/real_sample.py
→ Produces data/real_sample_final.csv (matched in size to synthetic dataset).

Step 2 – (Optional) Prepare prompts from raw CSV
```bash
python scripts/prepare_promptsV1.py
→ Produces data/batched_prompts.csv (structured GPT-2 prompts).
⚠️ Note: A pre-generated version of this file is already included in data/.
Run this step only if you want to regenerate prompts yourself.

Step 3 – (Optional) Generate synthetic SCADA logs
```bash
python scripts/prepare_prompts.py
→ Uses GPT-2 to generate synthetic records. Raw text saved to data/gpt2_raw_output_week6.txt.
⚠️ Note: A cleaned synthetic dataset is already provided (data/synthetic_scada_cleaned_final.csv).

Step 4 – (Optional) Post-process synthetic data
```bash
python scripts/postprocess.py
→ Cleans/parses raw logs into data/synthetic_scada_cleaned_final.csv.
⚠️ Already included in the repo; only run this if you regenerate data.

Step 5 – Evaluate real vs. synthetic data
Quick QA:
```bash
python scripts/evaluate.py

Full analysis (KDEs, correlations, ML utility)::
```bash
python scripts/week4.py

Step 6 – Launch interactive prototype
```bash
streamlit run app/app_streamlit.py
→ Opens the EdgeSynth Explorer (KDE overlays, correlation heatmaps, privacy checks).


📊 Results Summary

Distributional fidelity: KDE plots and KS tests show strong similarity between real and synthetic data.

Correlation preservation: Multivariate structures (e.g., RotorSpeed ↔ GeneratorSpeed correlation of 0.99) retained.

Predictive utility:

Real data → MAE = 0.04, R² = 1.00

Synthetic data → MAE = 0.03, R² = 1.00

Efficiency: ~200 synthetic samples/minute on CPU (Intel i7, 32GB RAM).

📘 Publications & References

This repository supports the IEEE research paper:
"EdgeSynth: Leveraging Generative AI for Synthetic Edge Data"

For extended details, see:

docs/supporting_material.pdf – Extended literature, lifecycle, ethical considerations.

docs/slides/ – Project presentation.

📜 License

This project is released under the MIT License. See LICENSE for details.

🙌 Acknowledgements

Hugging Face Transformers [19]

PyTorch

Streamlit [20]

Synthetic Data Vault (SDV) [21]

Wind turbine SCADA dataset (public domain)

---

✅ This gives you a polished repo overview:  
- **Why** EdgeSynth exists  
- **How** to install and run  
- **Where** everything is located  
- **What results** to expect  
