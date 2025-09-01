import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Load datasets
real_df = pd.read_csv("../data/real_sample_final.csv")
synthetic_df = pd.read_csv("../data/synthetic_scada_cleaned_final.csv")

# Standardize column names
real_df = real_df.rename(columns={"GeneratorTemperature": "GeneratorTemp"})
synthetic_df = synthetic_df.rename(columns={
    "GeneratorTemperature": "GeneratorTemp",
    "Power": "PowerOutput"
})

# Define common columns
common_cols = ['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemp']

# Filter to shared structure
real_df = real_df[common_cols]
synthetic_df = synthetic_df[common_cols]

# === Step 2: KL Divergence ===
def kl_divergence(p, q, bins=50):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist += 1e-10
    q_hist += 1e-10
    return entropy(p_hist, q_hist)

kl_results = {col: kl_divergence(real_df[col], synthetic_df[col]) for col in common_cols}

# === Step 3: KDE Plots ===
os.makedirs("../data/outputs", exist_ok=True)
for col in common_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(real_df[col], label="Real", fill=True)
    sns.kdeplot(synthetic_df[col], label="Synthetic", fill=True)
    plt.title(f"Distribution Comparison: {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../data/outputs/dist_{col}.png")
    plt.close()

# === Step 4: Correlation Heatmaps ===
plt.figure(figsize=(8, 5))
sns.heatmap(real_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap – Real Data")
plt.tight_layout()
plt.savefig("../data/outputs/real_corr.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.heatmap(synthetic_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap – Synthetic Data")
plt.tight_layout()
plt.savefig("../data/outputs/synth_corr.png")
plt.close()

# === Step 5: ML Utility Test ===
def run_model(df, label=""):
    X = df.drop("PowerOutput", axis=1)
    y = df["PowerOutput"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{label} – MAE: {mae:.2f}, R²: {r2:.2f}")
    return {"MAE": mae, "R2": r2}

real_metrics = run_model(real_df, "Real")
synthetic_metrics = run_model(synthetic_df, "Synthetic")

# ✅ Final Outputs
kl_results, real_metrics, synthetic_metrics
