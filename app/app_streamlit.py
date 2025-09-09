# app/app_streamlit.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ----------------- PAGE & STYLE -----------------
st.set_page_config(page_title="EdgeSynth", page_icon="🌀", layout="wide")
sns.set_theme(style="whitegrid")

# ----------------- PATHS -----------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA_DIR = ROOT / "data"

DATA_REAL   = DATA_DIR / "real_sample_final.csv"
DATA_SYN    = DATA_DIR / "synthetic_scada_cleaned.csv"
DATA_SYN_V2 = DATA_DIR / "synthetic_scada_cleaned_v2.csv"   # optional refined set

# ----------------- HELPERS -----------------
CANON_COLS = ["WindSpeed", "RotorSpeed", "GeneratorSpeed", "PowerOutput", "GeneratorTemperature"]
RENAME_MAP = {
    "Power": "PowerOutput",
    "GeneratorTemp": "GeneratorTemperature",
}

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize expected alternate names
    df = df.rename(columns=RENAME_MAP)
    return df

@st.cache_data
def get_available_sources():
    sources = {}
    if DATA_REAL.exists():   sources["Real"] = str(DATA_REAL)
    if DATA_SYN_V2.exists(): sources["Synthetic (v2 refined)"] = str(DATA_SYN_V2)
    if DATA_SYN.exists():    sources["Synthetic (v1)"] = str(DATA_SYN)
    return sources

@st.cache_data
def to_common(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in CANON_COLS if c in df.columns]
    return df[keep].dropna()

# ----------------- APP -----------------
st.title("EdgeSynth – SCADA Data Explorer & Privacy Check")

sources = get_available_sources()
if not sources:
    st.error(
        "No datasets found in ./data.\n\n"
        "Expected at least one of:\n"
        "• data/real_sample_final.csv\n"
        "• data/synthetic_scada_cleaned.csv\n"
        "• data/synthetic_scada_cleaned_v2.csv"
    )
    st.stop()

left, right = st.columns([1, 1])
with left:
    dataset_name = st.selectbox("Dataset to preview", list(sources.keys()), index=0)
    df = to_common(load_csv(Path(sources[dataset_name])))

with right:
    compare_choices = [k for k in sources.keys() if k != dataset_name]
    if not compare_choices:
        st.warning("Only one dataset available. Add another file in ./data to enable comparison.")
        st.stop()
    compare_with = st.selectbox("Compare against (for plots/privacy)", compare_choices, index=0)
    df_cmp = to_common(load_csv(Path(sources[compare_with])))

# ----------------- PREVIEW + DOWNLOAD -----------------
st.subheader("📋 Preview")
n_rows = st.slider("Rows to show", min_value=10, max_value=len(df), value=min(len(df), 200), step=10)
st.dataframe(df.head(n_rows), use_container_width=True)

csv_bytes = df.head(n_rows).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download selection as CSV", csv_bytes, "edgesynth_preview.csv", "text/csv")

# ----------------- SUMMARY STATS -----------------
st.subheader("📊 Summary statistics")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**{dataset_name}**")
    st.dataframe(df.describe().T, use_container_width=True)
with c2:
    st.markdown(f"**{compare_with}**")
    st.dataframe(df_cmp.describe().T, use_container_width=True)

# ----------------- DISTRIBUTION COMPARISON -----------------
st.subheader("📈 Distribution comparison (KDE)")
feature = st.selectbox("Feature", df.columns)

# Center the plot
c1, c2, c3 = st.columns([1, 3, 1])
with c2:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.kdeplot(df[feature], label=dataset_name, fill=True, ax=ax)
    sns.kdeplot(df_cmp[feature], label=compare_with, fill=True, ax=ax)
    ax.set_title(f"Distribution: {feature}", fontsize=10)
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

# ----------------- CORRELATION HEATMAPS -----------------
st.subheader("🔗 Correlation heatmaps")
h1, h2 = st.columns(2)

with h1:
    fig1, ax1 = plt.subplots(figsize=(3, 4))
    sns.heatmap(
        df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax1,
        annot_kws={"size": 6}, cbar=False, square=True, fmt=".2f"
    )
    ax1.set_title(f"Correlation – {dataset_name}", fontsize=8)
    ax1.tick_params(axis='both', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=False)

with h2:
    fig2, ax2 = plt.subplots(figsize=(3, 4))
    sns.heatmap(
        df_cmp.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2,
        annot_kws={"size": 6}, cbar=False, square=True, fmt=".2f"
    )
    ax2.set_title(f"Correlation – {compare_with}", fontsize=8)
    ax2.tick_params(axis='both', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=False)

# ----------------- PRIVACY CHECK -----------------
st.subheader("🛡️ Privacy check (nearest-neighbor distance)")
st.caption(
    "We z-score features and compute each target row’s nearest-neighbor distance in the reference dataset. "
    "Very small distances can indicate potential memorization. This is a heuristic, not a formal guarantee."
)

target_df = df.copy()
ref_df = df_cmp.copy()

# Guard against degenerate cases
if target_df.empty or ref_df.empty:
    st.warning("One of the datasets is empty after filtering. Cannot run privacy check.")
else:
    # Standardize
    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref_df.values)
    X_tgt = scaler.transform(target_df.values)

    # Nearest neighbor distance to reference
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_ref)
    dists, _ = nbrs.kneighbors(X_tgt)
    dists = dists.flatten()

    # Threshold heuristic (z-space): <= 0.25 = suspiciously close
    threshold = st.slider("Flag distance threshold (z-space)", 0.05, 1.0, 0.25, 0.05)
    pct_flag = (dists <= threshold).mean() * 100
    st.metric("Share of rows flagged as 'too close'", f"{pct_flag:.2f}%")
    st.caption(
        f"Min distance: {dists.min():.3f} | Median: {np.median(dists):.3f} | 95th %ile: {np.percentile(dists, 95):.3f}"
    )

    with st.expander("Show flagged rows (first 50)"):
        flagged_idx = np.where(dists <= threshold)[0][:50]
        st.dataframe(target_df.iloc[flagged_idx], use_container_width=True)

st.markdown("---")
st.caption(
    "EdgeSynth – MSc Project. This privacy check is indicative only; formal guarantees require DP or attack-based evaluations."
)
