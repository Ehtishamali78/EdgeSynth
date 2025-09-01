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
