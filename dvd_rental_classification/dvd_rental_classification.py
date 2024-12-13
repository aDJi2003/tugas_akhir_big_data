import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import hdbscan

data = pd.read_csv("dvd_rental_dataset.csv")

data['rental_date'] = pd.to_datetime(data['rental_date'])
data['return_date'] = pd.to_datetime(data['return_date'])
data['rental_duration'] = (data['return_date'] - data['rental_date']).dt.days

data = data[data['rental_duration'] > 0]

features = data[['amount', 'rental_duration']]

features = features.fillna(features.median())

Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1

features_no_outliers = features[~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)]

data_no_outliers = data.loc[features_no_outliers.index]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_no_outliers)

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

hdbscan_model = hdbscan.HDBSCAN(min_samples=150, min_cluster_size=300)
labels_hdbscan = hdbscan_model.fit_predict(features_scaled)

plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels_hdbscan, cmap='viridis', alpha=0.7)
plt.colorbar(label='Cluster')
plt.title('Clustering Results Using HDBSCAN')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

data_no_outliers['cluster'] = labels_hdbscan

unique_clusters = np.unique(labels_hdbscan)
for cluster in unique_clusters:
    if cluster == -1:
        print("\nCluster: Noise")
        cluster_data = data_no_outliers[data_no_outliers['cluster'] == cluster]
    else:
        print(f"\nCluster: {cluster}")
        cluster_data = data_no_outliers[data_no_outliers['cluster'] == cluster]

    print(cluster_data[['amount', 'rental_duration']].describe())

    print("Possible reasons for cluster formation:")
    if cluster_data['amount'].std() > cluster_data['rental_duration'].std():
        print("- Likely based on significant variation in 'amount'.")
    elif cluster_data['rental_duration'].std() > cluster_data['amount'].std():
        print("- Likely based on significant variation in 'rental_duration'.")
    else:
        print("- Similar variation in 'amount' and 'rental_duration'.")

valid_indices = labels_hdbscan != -1
features_valid = features_scaled[valid_indices]
labels_valid = labels_hdbscan[valid_indices]

if len(np.unique(labels_valid)) > 1:
    silhouette_hdbscan = silhouette_score(features_valid, labels_valid, metric='euclidean')
    print(f"\nSilhouette Score (HDBSCAN): {silhouette_hdbscan:.2f}")
else:
    print("\nSilhouette score tidak dapat dihitung karena hanya ada satu cluster valid.")
