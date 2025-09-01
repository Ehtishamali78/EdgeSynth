# scripts/real_sample.py
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths (relative to this file)
# -----------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
RAW_CSV = DATA_DIR / "wind_turbine.csv"
OUT_CSV = DATA_DIR / "real_sample_final.csv"
SYNTH_CSV = DATA_DIR / "synthetic_scada_cleaned.csv"  # if present, used to match sample size

# -----------------------------
# Settings
# -----------------------------
CHUNK_SIZE = 100_000
DEFAULT_SAMPLE_SIZE = 17_550     # fallback if synthetic file not found
MAX_PER_CHUNK = 1_000            # cap per chunk for diversity
SEED = 42

# Required columns (canonical)
REQUIRED = ["WindSpeed", "RotorSpeed", "GeneratorSpeed", "PowerOutput", "GeneratorTemperature"]
RENAME_MAP = {
    "Power": "PowerOutput",
    "GeneratorTemp": "GeneratorTemperature",
}

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw dataset: {RAW_CSV}")

    # Determine desired sample size
    if SYNTH_CSV.exists():
        try:
            synth_n = sum(1 for _ in open(SYNTH_CSV, "r", encoding="utf-8")) - 1  # minus header
            sample_size = max(1, synth_n)
        except Exception:
            sample_size = DEFAULT_SAMPLE_SIZE
    else:
        sample_size = DEFAULT_SAMPLE_SIZE

    samples = []
    total_collected = 0
    print(f"📦 Sampling from {RAW_CSV.name} with operational filters… target rows: {sample_size}")

    # Read in chunks (read all columns to allow renaming reliably)
    for chunk in pd.read_csv(RAW_CSV, chunksize=CHUNK_SIZE):
        # Normalize potential alternative column names
        chunk = chunk.rename(columns=RENAME_MAP)

        # Skip chunks missing required columns
        if any(c not in chunk.columns for c in REQUIRED):
            # Try to continue; your source may have varying schemas
            available = [c for c in REQUIRED if c in chunk.columns]
            if not available:
                continue
            # Reduce to only available req columns to avoid memory bloat
            chunk = chunk[available]

        # Drop missing on the columns we will use
        present_required = [c for c in REQUIRED if c in chunk.columns]
        chunk = chunk.dropna(subset=present_required)

        # Filter meaningful operational data
        if "PowerOutput" in chunk.columns and "WindSpeed" in chunk.columns:
            q = (chunk["PowerOutput"] > 0) & (chunk["WindSpeed"] > 2.0)
            chunk = chunk.loc[q]

        # If not all required columns are present, skip this chunk
        if any(c not in chunk.columns for c in REQUIRED):
            continue

        # Determine how many rows to sample from this chunk
        remaining = sample_size - total_collected
        if remaining <= 0:
            break
        n = min(remaining, len(chunk), MAX_PER_CHUNK)
        if n <= 0:
            continue

        samples.append(chunk.sample(n=n, random_state=SEED))
        total_collected += n

        if total_collected >= sample_size:
            break

    if not samples:
        raise RuntimeError("No samples collected. Check column names and filters in the raw dataset.")

    df_sample = pd.concat(samples, ignore_index=True)

    # Ensure exact size (trim if we overshot)
    if len(df_sample) > sample_size:
        df_sample = df_sample.head(sample_size)

    # Reorder columns to canonical order
    df_sample = df_sample[REQUIRED]

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved {len(df_sample)} rows → {OUT_CSV}")

if __name__ == "__main__":
    main()
