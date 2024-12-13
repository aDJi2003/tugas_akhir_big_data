import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest
import numpy as np
import scipy.stats as stats

file_path = "crop_dataset.csv"
df = pd.read_csv(file_path)

df['Season'] = df['Season'].str.strip()
df['Crop'] = df['Crop'].str.strip()

df_rice = df[(df["Crop"] == "Rice") & (df["Season"].isin(["Summer", "Winter"]))]

summer = df_rice[df_rice["Season"] == "Summer"]["Production"]
winter = df_rice[df_rice["Season"] == "Winter"]["Production"]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(summer, kde=True, color="blue", bins=20)
plt.title("Distribusi Hasil Panen Musim Summer")

plt.subplot(1, 2, 2)
sns.histplot(winter, kde=True, color="green", bins=20)
plt.title("Distribusi Hasil Panen Musim Winter")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
stats.probplot(summer, dist="norm", plot=plt)
plt.title("Q-Q Plot Musim Summer")

plt.subplot(1, 2, 2)
stats.probplot(winter, dist="norm", plot=plt)
plt.title("Q-Q Plot Musim Winter")

plt.tight_layout()
plt.show()

shapiro_summer = shapiro(summer)
shapiro_winter = shapiro(winter)

print(f"Shapiro-Wilk Test untuk Musim Summer: Statistik = {shapiro_summer.statistic}, p-value = {shapiro_summer.pvalue}")
print(f"Shapiro-Wilk Test untuk Musim Winter: Statistik = {shapiro_winter.statistic}, p-value = {shapiro_winter.pvalue}")
