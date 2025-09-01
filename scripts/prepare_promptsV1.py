import pandas as pd

chunks = pd.read_csv("../data/wind_turbine.csv", chunksize=100000)
filtered_rows = []

for chunk in chunks:

    # Filter based on meaningful operational data
    chunk = chunk[(chunk['PowerOutput'] > 0) & (chunk['WindSpeed'] > 2.0)]

    # Sample 10–50 rows per chunk
    filtered_rows.append(chunk.sample(min(50, len(chunk)), random_state=42))

# Combine samples from all chunks
df = pd.concat(filtered_rows, ignore_index=True)

# Format to prompt
df['Prompt'] = df.apply(
    lambda row: f"""SCADA Log Entry:
Timestamp: {row['Datetime']}
WindSpeed: {row['WindSpeed']} m/s
RotorSpeed: {row['RotorSpeed']} rpm
GeneratorSpeed: {row['GeneratorSpeed']} rpm
Power: {row['PowerOutput']} kW
GeneratorTemp: {row['GeneratorTemperature']} °C
---""", axis=1
)

# Save prompts for generation
df[['Prompt']].to_csv("../data/batched_prompts.csv", index=False, header=False)