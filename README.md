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

EdgeSynth/

│

├── data/ # Raw and processed datasets

│ ├── raw/ # Original wind turbine SCADA dataset

│ ├── processed/ # Cleaned training/evaluation datasets

│ └── synthetic/ # Generated synthetic datasets

│

├── scripts/ # Core pipeline scripts

│ ├── prepare\_prompts.py # Converts SCADA logs → GPT-2 prompts

│ ├── prepare\_prompts\_v1.py # Initial version of prompt preparation

│ ├── postprocess.py # Cleans GPT-2 raw outputs → structured dataset

│ ├── evaluate.py # Synthetic-only histograms, scatter plots

│ ├── week4.py # Main evaluation: distributions, KS tests, ML utility

│ └── real\_sample.py # Extracts real data samples for evaluation

│

├── app/

│ └── app\_streamlit.py # Streamlit web interface (prototype)

│

├── results/ # Evaluation outputs \& visualizations

│ ├── distributions/ # KDE plots

│ ├── correlations/ # Heatmaps

│ ├── metrics/ # KS stats, regression results (CSV)

│ └── screenshots/ # Streamlit prototype views

│

├── env/

│ └── requirements.txt # Python dependencies

│

├── docs/ # Supporting material

│ ├── supporting\_material.pdf

│ └── slides/ # Presentation slides

│

└── README.md # Project overview (this file)



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



▶️ Usage

Step 1: Prepare prompts
python scripts/prepare_prompts.py



Step 2: Generate synthetic SCADA logs (via GPT-2 pipeline)
\# Run your GPT-2 generation script (Jupyter or Python-based)



Step 3: Post-process generated data
python scripts/postprocess.py

Step 4: Evaluate synthetic vs. real data
python scripts/week4.py

Step 5: Launch interactive prototype
streamlit run app/app_streamlit.py

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
