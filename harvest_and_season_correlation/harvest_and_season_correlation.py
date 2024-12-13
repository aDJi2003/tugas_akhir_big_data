import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

file_path = "crop_dataset.csv"
df = pd.read_csv(file_path)

df['Season'] = df['Season'].str.strip()
df['Crop'] = df['Crop'].str.strip()

df_rice = df[(df["Crop"] == "Rice") & (df["Season"].isin(["Summer", "Winter"]))]

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()

df_rice_cleaned = remove_outliers(df_rice, "Production")

print(f"Jumlah data sebelum penghapusan outlier: {len(df_rice)}")
print(f"Jumlah data setelah penghapusan outlier: {len(df_rice_cleaned)}")

season_group = df_rice_cleaned.groupby("Season")["Production"].mean()
print("\nRata-rata hasil panen per musim (setelah outlier dihapus):\n", season_group)

sns.boxplot(x="Season", y="Production", data=df_rice_cleaned, hue="Season", palette="Set2", dodge=False)
for season, avg in season_group.items():
    plt.text(season_group.index.tolist().index(season), avg, f"{avg:.0f}", 
             horizontalalignment='center', color='blue')
plt.title("Distribusi Hasil Panen Padi Berdasarkan Musim")
plt.xlabel("Musim")
plt.ylabel("Hasil Panen (Production)")
plt.show()

df_rice_cleaned['Season_numeric'] = df_rice_cleaned['Season'].map({'Summer': 0, 'Winter': 1})

corr, p_value = spearmanr(df_rice_cleaned['Season_numeric'], df_rice_cleaned['Production'])
print(f"Spearman's Rank Correlation: {corr:.4f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Ada korelasi signifikan antara musim dan hasil panen.")
    if corr > 0:
        print("Korelasi bersifat positif: produksi meningkat dengan musim Winter.")
    elif corr < 0:
        print("Korelasi bersifat negatif: produksi menurun dengan musim Winter.")
else:
    print("Tidak ada korelasi signifikan antara musim dan hasil panen.")
