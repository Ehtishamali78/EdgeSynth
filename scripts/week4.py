import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Paths (relative to this file)
# -----------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
RESULTS_DIR = (HERE / ".." / "results").resolve()
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

REAL_CSV = DATA_DIR / "real_sample_final.csv"
SYN_CSV = DATA_DIR / "synthetic_scada_cleaned.csv"  # canonical final name

# -----------------------------
# Load datasets
# -----------------------------
if not REAL_CSV.exists():
    raise FileNotFoundError(f"Missing real dataset: {REAL_CSV}")
if not SYN_CSV.exists():
    raise FileNotFoundError(f"Missing synthetic dataset: {SYN_CSV}")

real_df = pd.read_csv(REAL_CSV)
synthetic_df = pd.read_csv(SYN_CSV)

# -----------------------------
# Standardize column names
# Accepts either GeneratorTemp/GeneratorTemperature and Power/PowerOutput
# -----------------------------
rename_map = {
    "Power": "PowerOutput",
    "GeneratorTemp": "GeneratorTemperature"
}
real_df = real_df.rename(columns=rename_map)
synthetic_df = synthetic_df.rename(columns=rename_map)

# Required columns
common_cols = ["WindSpeed", "RotorSpeed", "GeneratorSpeed", "PowerOutput", "GeneratorTemperature"]
missing_real = [c for c in common_cols if c not in real_df.columns]
missing_syn = [c for c in common_cols if c not in synthetic_df.columns]
if missing_real:
    raise KeyError(f"Real dataset missing columns: {missing_real}")
if missing_syn:
    raise KeyError(f"Synthetic dataset missing columns: {missing_syn}")

# Keep only required columns
real_df = real_df[common_cols].copy()
synthetic_df = synthetic_df[common_cols].copy()

# -----------------------------
# KL Divergence per feature
# -----------------------------
def kl_divergence(p, q, bins=50):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    # add small epsilon to avoid zeros
    p_hist = p_hist + 1e-12
    q_hist = q_hist + 1e-12
    return float(entropy(p_hist, q_hist))

kl_results = {col: kl_divergence(real_df[col].values, synthetic_df[col].values) for col in common_cols}
pd.DataFrame.from_dict(kl_results, orient="index", columns=["KL_Divergence"])\
  .to_csv(METRICS_DIR / "kl_divergence.csv")

# -----------------------------
# KDE Plots (real vs synthetic)
# -----------------------------
sns.set(style="whitegrid")
for col in common_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(real_df[col], label="Real", fill=True, common_norm=False)
    sns.kdeplot(synthetic_df[col], label="Synthetic", fill=True, common_norm=False)
    plt.title(f"Distribution Comparison: {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"dist_{col}.png", dpi=300)
    plt.close()

# -----------------------------
# Correlation Heatmaps
# -----------------------------
plt.figure(figsize=(7.5, 5.5))
sns.heatmap(real_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap — Real Data")
plt.tight_layout()
plt.savefig(FIG_DIR / "real_corr.png", dpi=300)
plt.close()

plt.figure(figsize=(7.5, 5.5))
sns.heatmap(synthetic_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap — Synthetic Data")
plt.tight_layout()
plt.savefig(FIG_DIR / "synthetic_corr.png", dpi=300)
plt.close()

# -----------------------------
# ML Utility Test (Random Forest)
# -----------------------------
def run_model(df, label=""):
    X = df.drop("PowerOutput", axis=1)
    y = df["PowerOutput"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{label} — MAE: {mae:.4f}, R²: {r2:.4f}")
    return {"Dataset": label, "MAE": mae, "R2": r2}

real_metrics = run_model(real_df, "Real")
synthetic_metrics = run_model(synthetic_df, "Synthetic")

# Save metrics table
metrics_df = pd.DataFrame([real_metrics, synthetic_metrics])
metrics_df.to_csv(METRICS_DIR / "regression_results.csv", index=False)

print("✅ Outputs saved:")
print(f"  • KL divergence  → {METRICS_DIR / 'kl_divergence.csv'}")
print(f"  • RF metrics     → {METRICS_DIR / 'regression_results.csv'}")
print(f"  • Figures        → {FIG_DIR}")
