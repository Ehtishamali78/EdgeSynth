# scripts/prepare_prompts_v1.py
import pandas as pd
from pathlib import Path

# ---- Paths ----
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
RAW_CSV = DATA_DIR / "wind_turbine.csv"
PROMPTS_CSV = DATA_DIR / "batched_prompts.csv"

CHUNKSIZE = 100_000
SAMPLE_CAP = 50
SEED = 42

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw dataset: {RAW_CSV}")

    filtered_rows = []

    # Stream-read for memory safety
    for chunk in pd.read_csv(RAW_CSV, chunksize=CHUNKSIZE):
        # Normalize potential alternative column names
        chunk = chunk.rename(columns={
            "Power": "PowerOutput",
            "GeneratorTemp": "GeneratorTemperature",
            "Timestamp": "Datetime"
        })

        required = ["Datetime", "WindSpeed", "RotorSpeed", "GeneratorSpeed",
                    "PowerOutput", "GeneratorTemperature"]
        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise KeyError(f"Missing required columns in chunk: {missing}")

        # Filter based on meaningful operational data
        q = (chunk["PowerOutput"] > 0) & (chunk["WindSpeed"] > 2.0)
        chunk = chunk.loc[q]

        # Sample up to SAMPLE_CAP rows per chunk (if available)
        n = min(SAMPLE_CAP, len(chunk))
        if n > 0:
            filtered_rows.append(chunk.sample(n, random_state=SEED))

    if not filtered_rows:
        raise RuntimeError("No rows passed the filter — adjust thresholds or check data.")

    df = pd.concat(filtered_rows, ignore_index=True)

    # Prompt formatting
    def to_prompt(row):
        return (
            "SCADA Log Entry:\n"
            f"Timestamp: {row['Datetime']}\n"
            f"WindSpeed: {row['WindSpeed']} m/s\n"
            f"RotorSpeed: {row['RotorSpeed']} rpm\n"
            f"GeneratorSpeed: {row['GeneratorSpeed']} rpm\n"
            f"Power: {row['PowerOutput']} kW\n"
            f"GeneratorTemp: {row['GeneratorTemperature']} °C\n"
            "---"
        )

    df["Prompt"] = df.apply(to_prompt, axis=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df[["Prompt"]].to_csv(PROMPTS_CSV, index=False, header=False)
    print(f"✅ Wrote prompts → {PROMPTS_CSV} (rows={len(df)})")

if __name__ == "__main__":
    main()
