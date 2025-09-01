# scripts/postprocess.py
import re
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Paths (no helper needed; resolve relative to this file) ---
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()

RAW_TXT = DATA_DIR / "synthetic_raw.txt"                 # <-- put your raw GPT-2 generations here
PARSED_CSV = DATA_DIR / "synthetic_scada_final.csv"      # intermediate (parsed, before cleaning)
CLEAN_CSV = DATA_DIR / "synthetic_scada_cleaned.csv"     # final canonical filename

# --- Column names (canonical) ---
COL_DATETIME = "Datetime"
COL_WS = "WindSpeed"
COL_RS = "RotorSpeed"
COL_GS = "GeneratorSpeed"
COL_PO = "PowerOutput"
COL_GT = "GeneratorTemperature"


def parse_entries(text: str) -> pd.DataFrame:
    """
    Parse raw GPT-2 text containing repeated blocks like:

    SCADA Log Entry:
    Timestamp: 2022-03-01 12:00:00
    WindSpeed: 5.2 m/s
    RotorSpeed: 12.3 rpm
    GeneratorSpeed: 980.0 rpm
    Power: 1.7 kW
    GeneratorTemp: 54.2 °C
    ---
    """
    entries = []
    blocks = text.split("SCADA Log Entry:")
    for block in blocks[1:]:
        # Collect header lines up to the first '---'
        header_lines = []
        for line in block.strip().splitlines():
            if line.strip().startswith("---"):
                break
            if line.strip():
                header_lines.append(line.strip())
        header_text = "\n".join(header_lines)

        entry = {}
        try:
            # Timestamp (string; fixed later)
            m = re.search(r"Timestamp:\s*(.*)", header_text)
            entry[COL_DATETIME] = m.group(1).strip() if m else ""

            # Numeric fields (strip units)
            m = re.search(r"WindSpeed:\s*([\d.]+)", header_text)
            entry[COL_WS] = float(m.group(1)) if m else None

            m = re.search(r"RotorSpeed:\s*([\d.]+)", header_text)
            entry[COL_RS] = float(m.group(1)) if m else None

            m = re.search(r"GeneratorSpeed:\s*([\d.]+)", header_text)
            entry[COL_GS] = float(m.group(1)) if m else None

            m = re.search(r"Power:\s*([\d.]+)", header_text)
            entry[COL_PO] = float(m.group(1)) if m else None

            m = re.search(r"GeneratorTemp:\s*([\d.]+)", header_text)
            entry[COL_GT] = float(m.group(1)) if m else None

            entries.append(entry)
        except Exception:
            # Skip malformed block
            continue

    return pd.DataFrame(entries)


def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make timestamps evenly spaced using the average gap of parseable times.
    Falls back to 60s spacing from a default start if needed.
    """
    try:
        parsed = pd.to_datetime(df[COL_DATETIME], errors="coerce")
        parsed = parsed.dropna().reset_index(drop=True)

        if len(parsed) >= 2:
            avg_gap = (parsed.iloc[1] - parsed.iloc[0]).total_seconds()
            if avg_gap <= 0 or avg_gap > 3600:  # sanity bound: 0 < gap <= 1h
                avg_gap = 60
        else:
            avg_gap = 60

        start_time = parsed.iloc[0] if len(parsed) > 0 else datetime(2022, 1, 1)
    except Exception:
        avg_gap = 60
        start_time = datetime(2022, 1, 1)

    new_times = [start_time + timedelta(seconds=i * avg_gap) for i in range(len(df))]
    df[COL_DATETIME] = new_times
    return df


def main():
    if not RAW_TXT.exists():
        raise FileNotFoundError(
            f"Raw GPT-2 output not found: {RAW_TXT}\n"
            f"Place your raw generations in this file before running postprocess."
        )

    # ---- Load raw GPT-2 output ----
    text = RAW_TXT.read_text(encoding="utf-8")

    # ---- Parse raw text ----
    df = parse_entries(text)
    PARSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PARSED_CSV, index=False)
    print(f"✅ Parsed and saved {len(df)} rows → {PARSED_CSV.name}")

    # ---- Drop incomplete rows ----
    df_clean = df.dropna().copy()

    # ---- Clip to realistic operational ranges (align with report) ----
    # WindSpeed (m/s): 0–25
    if COL_WS in df_clean.columns:
        df_clean[COL_WS] = df_clean[COL_WS].clip(0, 25)

    # RotorSpeed (rpm): typical large turbine ~0–80 rpm (adjust if your turbine differs)
    if COL_RS in df_clean.columns:
        df_clean[COL_RS] = df_clean[COL_RS].clip(0, 80)

    # GeneratorSpeed (rpm): up to 1500 rpm per your text
    if COL_GS in df_clean.columns:
        df_clean[COL_GS] = df_clean[COL_GS].clip(0, 900)

    # PowerOutput (kW): your plots show near-rated ~7 kW; keep a safe cap at 10 kW
    if COL_PO in df_clean.columns:
        df_clean[COL_PO] = df_clean[COL_PO].clip(0, 10)

    # GeneratorTemperature (°C): 0–120
    if COL_GT in df_clean.columns:
        df_clean[COL_GT] = df_clean[COL_GT].clip(0, 120)

    # ---- Fix timestamps (even spacing) ----
    df_clean = fix_timestamps(df_clean)

    # ---- Save cleaned file (canonical name used across repo) ----
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"✅ Cleaned synthetic dataset saved → {CLEAN_CSV.name} (rows={len(df_clean)})")


if __name__ == "__main__":
    main()
