# Data Folder

This folder holds **datasets** used in EdgeSynth.

## Files

- `wind_turbine.csv`  
  Raw SCADA dataset (⚠️ not included in repo — too large, proprietary). 
  Download it from this link : https://www.kaggle.com/datasets/pythonafroz/wind-turbine-scada-data#:~:text=Wind%20Turbine%20SCADA%20data%20,in%20C%2C%20Windspeed%20in%20m%2Fs 
  Place it here before running `prepare_prompts.py`.

- `batched_prompts.csv`  
  Generated GPT-2 prompts, created by `scripts/prepare_prompts.py`.  
  Used as input for synthetic data generation.

- `synthetic_scada_cleaned.csv`  
  Final cleaned synthetic dataset, ready for evaluation and downstream ML tasks.

## Notes

- Do **not** commit raw datasets (`wind_turbine.csv`).  
- Keep synthetic outputs (`synthetic_scada_cleaned.csv`) if you want reproducibility.  
- Any intermediate raw generations (e.g., `synthetic_raw.txt`) should also be ignored in `.gitignore`.
