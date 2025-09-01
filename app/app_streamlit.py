# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
#
# # Load cleaned synthetic data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("data/synthetic_scada_cleaned.csv")
#     return df
#
# df = load_data()
# st.title("EdgeSynth – Synthetic SCADA Data Generator")
#
# # Sidebar settings
# st.sidebar.header("Configuration")
# n_rows = st.sidebar.slider("Number of synthetic entries to display:", 10, len(df), 50, step=10)
# sample_df = df.head(n_rows)
#
# st.subheader("📋 Preview of Synthetic SCADA Entries")
# st.dataframe(sample_df)
#
# # Download
# st.subheader("💾 Download Synthetic Data")
# csv = sample_df.to_csv(index=False).encode("utf-8")
# st.download_button("Download CSV", csv, "synthetic_preview.csv", "text/csv")
#
# # Visualization
# st.subheader("📈 Feature Distribution Plots")
# feature = st.selectbox("Select a feature to plot:", df.columns)
#
# fig, ax = plt.subplots(figsize=(6, 4))
# sns.histplot(df[feature], kde=True, bins=30, ax=ax)
# st.pyplot(fig)
#
# st.markdown("---")
# st.markdown("🧠 Built as part of MSc Project: *EdgeSynth – Leveraging Generative AI for Synthetic Edge Data*")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="EdgeSynth", page_icon="🌀", layout="wide")
sns.set_theme(style="whitegrid")

DATA_REAL  = "data/real_sample_final.csv"
DATA_SYN   = "data/synthetic_scada_cleaned.csv"
DATA_SYN_V2= "data/synthetic_scada_cleaned_v2.csv"   # produced by Week 6 refinement

# ----------------- LOADERS -----------------
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"GeneratorTemperature":"GeneratorTemp","Power":"PowerOutput"})
    return df

@st.cache_data
def get_available_sources():
    sources = {}
    if os.path.exists(DATA_REAL):   sources["Real"] = DATA_REAL
    if os.path.exists(DATA_SYN_V2): sources["Synthetic (v2 refined)"] = DATA_SYN_V2
    if os.path.exists(DATA_SYN):    sources["Synthetic (v1)"] = DATA_SYN
    return sources

@st.cache_data
def commonize(df):
    # focus on modeling columns
    cols = ['WindSpeed','RotorSpeed','GeneratorSpeed','PowerOutput','GeneratorTemp']
    keep = [c for c in cols if c in df.columns]
    return df[keep].dropna()

# ----------------- APP -----------------
st.title("EdgeSynth – SCADA Data Explorer & Privacy Check")

sources = get_available_sources()
if not sources:
    st.error("No datasets found in ./data. Please add real_sample.csv and synthetic_scada_cleaned.csv")
    st.stop()

left, right = st.columns([1,1])
with left:
    dataset_name = st.selectbox("Dataset to preview", list(sources.keys()), index=0)
    df = commonize(load_csv(sources[dataset_name]))
with right:
    compare_with = st.selectbox("Compare against (for plots/privacy)", [k for k in sources if k!=dataset_name])
    df_cmp = commonize(load_csv(sources[compare_with]))

# ----------------- PREVIEW + DOWNLOAD -----------------
st.subheader("📋 Preview")
n_rows = st.slider("Rows to show", 10, len(df), len(df), step=10)
st.dataframe(df.head(n_rows), use_container_width=True)

csv = df.head(n_rows).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download selection as CSV", csv, "edgesynth_preview.csv", "text/csv")

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

# Create 3 columns: left spacer, plot, right spacer
c1, c2, c3 = st.columns([1, 3, 1])
with c2:
    fig, ax = plt.subplots(figsize=(4, 3))  # smaller size
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
    fig1, ax1 = plt.subplots(figsize=(3, 4))  # smaller figure
    sns.heatmap(
        df.corr(), annot=True, cmap="coolwarm", ax=ax1,
        annot_kws={"size": 6},  # smaller numbers
        cbar=False, square=True
    )
    ax1.set_title(f"Correlation – {dataset_name}", fontsize=8)
    ax1.tick_params(axis='both', labelsize=7)  # smaller axis labels
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=False)

with h2:
    fig2, ax2 = plt.subplots(figsize=(3, 4))  # smaller figure
    sns.heatmap(
        df_cmp.corr(), annot=True, cmap="coolwarm", ax=ax2,
        annot_kws={"size": 6},
        cbar=False, square=True
    )
    ax2.set_title(f"Correlation – {compare_with}", fontsize=8)
    ax2.tick_params(axis='both', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=False)


# ----------------- PRIVACY CHECK -----------------
st.subheader("🛡️ Privacy check (nearest-neighbor distance)")
st.caption("We z-score features and compute each synthetic/target row's nearest neighbor distance in the reference dataset. "
           "Very small distances can indicate potential memorization. This is a heuristic, not a formal guarantee.")

target_df = df.copy()
ref_df    = df_cmp.copy()

# Standardize
scaler = StandardScaler()
X_ref = scaler.fit_transform(ref_df.values)
X_tgt = scaler.transform(target_df.values)

# Nearest neighbor distance to reference
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_ref)
dists, _ = nbrs.kneighbors(X_tgt)
dists = dists.flatten()

# Threshold heuristic (z-space): <=0.25 = suspiciously close
threshold = st.slider("Flag distance threshold (z-space)", 0.05, 1.0, 0.25, 0.05)
pct_flag = (dists <= threshold).mean()*100
st.metric("Share of rows flagged as 'too close'", f"{pct_flag:.2f}%")
st.caption(f"Min distance: {dists.min():.3f} | Median: {np.median(dists):.3f} | 95th %ile: {np.percentile(dists,95):.3f}")

with st.expander("Show flagged rows (first 50)"):
    flagged_idx = np.where(dists <= threshold)[0][:50]
    st.dataframe(target_df.iloc[flagged_idx], use_container_width=True)

st.markdown("---")
st.caption("EdgeSynth – MSc Project. This privacy check is indicative only; formal guarantees require DP or attack-based evaluations.")
