import pandas as pd

# === Configuration ===
file_path = "../data/wind_turbine.csv"         # Full dataset
output_path = "../data/real_sample_final.csv"        # Sample destination
sample_size = 17_551                            # Match synthetic data
chunk_size = 100_000                            # Tune this based on RAM

# Columns we care about
selected_cols = ['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']

samples = []
total_collected = 0

print("📦 Sampling from large file with operational filters...")

# Process in chunks
for chunk in pd.read_csv(file_path, usecols=selected_cols, chunksize=chunk_size):
    # Drop missing data
    chunk = chunk.dropna()

    # ✅ Filter meaningful operational data
    chunk = chunk[(chunk['PowerOutput'] > 0) & (chunk['WindSpeed'] > 2.0)]

    # How many rows still needed
    remaining = sample_size - total_collected
    n = min(remaining, len(chunk), 1000)

    if n > 0:
        sampled_chunk = chunk.sample(n=n, random_state=42)
        samples.append(sampled_chunk)
        total_collected += len(sampled_chunk)

    if total_collected >= sample_size:
        break

# Concatenate and trim to exact size
df_sample = pd.concat(samples).reset_index(drop=True).head(sample_size)

# Save
df_sample.to_csv(output_path, index=False)
print(f"✅ Saved {len(df_sample)} rows to {output_path}")
