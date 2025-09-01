import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def parse_entries(text):
    entries = []
    blocks = text.split("SCADA Log Entry:")
    for block in blocks[1:]:
        # Only take the header lines before the first '---'
        header_lines = []
        for line in block.strip().splitlines():
            if line.strip().startswith("---"):
                break
            if line.strip():
                header_lines.append(line.strip())

        header_text = "\n".join(header_lines)

        entry = {}
        try:
            entry["Timestamp"] = re.search(r"Timestamp:\s*(.*)", header_text).group(1).strip()
            entry["WindSpeed"] = float(re.search(r"WindSpeed:\s*([\d.]+)", header_text).group(1))
            entry["RotorSpeed"] = float(re.search(r"RotorSpeed:\s*([\d.]+)", header_text).group(1))
            entry["GeneratorSpeed"] = float(re.search(r"GeneratorSpeed:\s*([\d.]+)", header_text).group(1))
            entry["Power"] = float(re.search(r"Power:\s*([\d.]+)", header_text).group(1))
            entry["GeneratorTemp"] = float(re.search(r"GeneratorTemp:\s*([\d.]+)", header_text).group(1))
            entries.append(entry)
        except Exception:
            continue
    return pd.DataFrame(entries)

def fix_timestamps(df):
    """
    Fix timestamps to be evenly spaced, matching the average gap in the data if parseable.
    Falls back to 1-minute spacing if needed.
    """
    try:
        parsed_times = pd.to_datetime(df["Timestamp"], errors="coerce")
        parsed_times = parsed_times.dropna().reset_index(drop=True)

        if len(parsed_times) >= 2:
            avg_gap = (parsed_times.iloc[1] - parsed_times.iloc[0]).total_seconds()
            if avg_gap <= 0 or avg_gap > 3600:  # sanity check
                avg_gap = 60
        else:
            avg_gap = 60

        start_time = parsed_times.iloc[0] if len(parsed_times) > 0 else datetime(2022, 1, 1)
    except Exception:
        avg_gap = 60
        start_time = datetime(2022, 1, 1)

    new_times = [start_time + timedelta(seconds=i * avg_gap) for i in range(len(df))]
    df["Timestamp"] = new_times
    return df

# ---- Load raw GPT-2 output ----
with open("../data/gpt2_raw_output_week6.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---- Parse raw text ----
df = parse_entries(text)
df.to_csv("../data/synthetic_scada_final.csv", index=False)
print(f"✅ Parsed and saved {len(df)} entries to synthetic_scada_final.csv")

# ---- Drop incomplete rows ----
df_clean = df.dropna()

# ---- Clip values to realistic turbine ranges ----
df_clean["WindSpeed"] = df_clean["WindSpeed"].clip(0, 30)
df_clean["RotorSpeed"] = df_clean["RotorSpeed"].clip(0, 80)
df_clean["GeneratorSpeed"] = df_clean["GeneratorSpeed"].clip(0, 900)
df_clean["Power"] = df_clean["Power"].clip(0, 10)  # kW scale in your sample
df_clean["GeneratorTemp"] = df_clean["GeneratorTemp"].clip(0, 120)

# ---- Fix timestamps using real average gap ----
df_clean = fix_timestamps(df_clean)

# ---- Save cleaned file ----
df_clean.to_csv("../data/synthetic_scada_cleaned_final.csv", index=False)
print(f"✅ Cleaned synthetic dataset saved with {len(df_clean)} rows and fixed timestamps")

