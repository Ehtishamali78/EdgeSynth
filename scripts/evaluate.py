# # import pandas as pd
# # import re
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # # Load generated text output
# # with open("../data/generated_gpt2_output.txt", "r") as file:
# #     lines = file.readlines()
# #
# # # Clean and filter valid rows
# # cleaned = []
# # pattern = r"Timestamp: (.*?) \| WindSpeed: ([\d\.]+) m/s \| RotorSpeed: ([\d\.]+) rpm \| GeneratorSpeed: ([\d\.]+) rpm \| GeneratorTemp: ([\d\.]+) °C \| Power: ([\d\.]+) kW"
# #
# # for line in lines:
# #     match = re.match(pattern, line.strip())
# #     if match:
# #         cleaned.append({
# #             "Timestamp": match.group(1),
# #             "WindSpeed": float(match.group(2)),
# #             "RotorSpeed": float(match.group(3)),
# #             "GeneratorSpeed": float(match.group(4)),
# #             "GeneratorTemp": float(match.group(5)),
# #             "PowerOutput": float(match.group(6))
# #         })
# #
# # # Convert to DataFrame
# # synthetic_df = pd.DataFrame(cleaned)
# # print("✅ Parsed synthetic samples:", len(synthetic_df))
# # print(synthetic_df.head())
# #
# # # Load real data
# # real_df = pd.read_csv("../data/wind_turbine.csv")
# #
# # real_df = real_df[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'GeneratorTemperature', 'PowerOutput']]
# # real_df = real_df.dropna().reset_index(drop=True)
# # real_df = real_df.head(len(synthetic_df))  # Match length for comparison
# #
# # # Rename for consistency
# # real_df = real_df.rename(columns={"GeneratorTemperature": "GeneratorTemp"})
# #
# # # Compare statistics
# # print("\n📊 Mean Comparison:\n")
# # print(pd.DataFrame({
# #     "Real Mean": real_df.mean(),
# #     "Synthetic Mean": synthetic_df.drop(columns=["Timestamp"]).mean()
# # }))
# #
# # print("\n📊 Std Dev Comparison:\n")
# # print(pd.DataFrame({
# #     "Real Std": real_df.std(),
# #     "Synthetic Std": synthetic_df.drop(columns=["Timestamp"]).std()
# # }))
# #
# # # Plot histogram comparison
# # for col in synthetic_df.columns[1:]:
# #     plt.figure(figsize=(6, 4))
# #     sns.kdeplot(real_df[col], label='Real', fill=True)
# #     sns.kdeplot(synthetic_df[col], label='Synthetic', fill=True)
# #     plt.title(f'Distribution: {col}')
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.show()
#
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
# from datetime import datetime, timedelta
# import numpy as np
#
# # --- Load Real Data for Comparison ---
# real_df = pd.read_csv("../data/wind_turbine.csv")
# real_df = real_df[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'GeneratorTemperature', 'PowerOutput']]
# real_df = real_df.fillna(0)
#
# # --- Load Synthetic GPT-2 Generated Samples ---
# with open("../data/generated_gpt2_output.txt", "r") as f:
#     lines = f.readlines()
#
# synthetic_data = []
#
# for line in lines:
#     try:
#         # Extract values using regex
#         timestamp_match = re.search(r"Timestamp: (.+?) \|", line)
#         wind_match = re.search(r"WindSpeed: ([\d\.]+)", line)
#         rotor_match = re.search(r"RotorSpeed: ([\d\.]+)", line)
#         gen_speed_match = re.search(r"GeneratorSpeed: ([\d\.]+)", line)
#         temp_match = re.search(r"Temp: ([\d\.]+)", line)
#         power_match = re.search(r"Power: ([\d\.]+)", line)
#
#         if all([timestamp_match, wind_match, rotor_match, gen_speed_match, temp_match, power_match]):
#             row = {
#                 "Timestamp": timestamp_match.group(1),
#                 "WindSpeed": float(wind_match.group(1)),
#                 "RotorSpeed": float(rotor_match.group(1)),
#                 "GeneratorSpeed": float(gen_speed_match.group(1)),
#                 "GeneratorTemp": float(temp_match.group(1)),
#                 "PowerOutput": float(power_match.group(1))
#             }
#             synthetic_data.append(row)
#     except Exception as e:
#         print(f"Error parsing line: {line}")
#         continue
#
# synthetic_df = pd.DataFrame(synthetic_data)
# synthetic_df["Timestamp"] = pd.to_datetime(synthetic_df["Timestamp"], errors="coerce")
#
# print(f"✅ Parsed synthetic samples: {len(synthetic_df)}")
# print(synthetic_df.head())
#
# # --- Mean Comparison ---
# print("\n📊 Mean Comparison:\n")
# print(pd.DataFrame({
#     "Real Mean": real_df.mean(),
#     "Synthetic Mean": synthetic_df.mean(numeric_only=True)
# }))
#
# # --- Std Dev Comparison ---
# print("\n📊 Std Dev Comparison:\n")
# print(pd.DataFrame({
#     "Real Std": real_df.std(),
#     "Synthetic Std": synthetic_df.std(numeric_only=True)
# }))
#
# # --- Distribution Plot ---
# features = ['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'GeneratorTemperature', 'PowerOutput']
#
# for col in features:
#     plt.figure(figsize=(6, 4))
#     try:
#         # KDE Plot (if real has variance)
#         if real_df[col].std() > 0:
#             sns.kdeplot(real_df[col], label='Real', fill=True, color="blue")
#         sns.kdeplot(synthetic_df[col], label='Synthetic', fill=True, color="darkorange")
#
#         plt.title(f"Distribution: {col}")
#         plt.xlabel(col)
#         plt.ylabel("Density")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f"../report/distribution_{col}.png")  # Optional: save plots
#         plt.show()
#     except Exception as e:
#         print(f"⚠️ Failed to plot {col}: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned synthetic dataset
df = pd.read_csv("../data/synthetic_scada_cleaned.csv")

# Set visual style
sns.set(style="whitegrid")

# Histogram: WindSpeed
plt.figure()
sns.histplot(df["WindSpeed"], bins=30, kde=True)
plt.title("WindSpeed Distribution")
plt.xlabel("WindSpeed (m/s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Histogram: Power
plt.figure()
sns.histplot(df["Power"], bins=30, kde=True)
plt.title("Power Output Distribution")
plt.xlabel("Power (kW)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Histogram: GeneratorTemp
plt.figure()
sns.histplot(df["GeneratorTemp"], bins=30, kde=True)
plt.title("Generator Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter: WindSpeed vs Power
plt.figure()
sns.scatterplot(data=df, x="WindSpeed", y="Power")
plt.title("WindSpeed vs Power Output")
plt.xlabel("WindSpeed (m/s)")
plt.ylabel("Power (kW)")
plt.tight_layout()
plt.show()

# Scatter: RotorSpeed vs GeneratorSpeed
plt.figure()
sns.scatterplot(data=df, x="RotorSpeed", y="GeneratorSpeed")
plt.title("RotorSpeed vs GeneratorSpeed")
plt.xlabel("RotorSpeed (rpm)")
plt.ylabel("GeneratorSpeed (rpm)")
plt.tight_layout()
plt.show()

# Optional Boxplot for outliers
plt.figure()
sns.boxplot(data=df[["WindSpeed", "Power", "GeneratorTemp"]])
plt.title("Boxplot of Key Variables")
plt.tight_layout()
plt.show()
