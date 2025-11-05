'''
Program to implement k-means clustering technique using any standard dataset available in the public domain.

List of Activities:

Activity 1: Import Modules and Read Data
Activity 2: Data Cleaning
Activity 3: Find Optimal Value of K
Activity 4: Plot Silhouette Scores
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Activity 1: Import and read data
df = pd.read_csv("https://raw.githubusercontent.com/jiss-sngce/CO_3/main/jkcars.csv")
print(df.head())

# Activity 2: Data info and cleaning
print("Shape of the DataFrame:", df.shape)
df.info()
print("\nMissing values per column:\n", df.isnull().sum())

# Activity 3: Find Optimal Value of K
data_3d = df[['Volume', 'Weight', 'CO2']]
print(data_3d.head(3))

sil_scores = []
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data_3d)
    score = silhouette_score(data_3d, model.labels_)
    sil_scores.append(score)

sildataframe = pd.DataFrame({'K': range(2, 11), 'Silhouette Score': sil_scores})
print(sildataframe)

# Plot silhouette scores vs number of clusters
plt.figure(figsize=(10, 6))
plt.plot(sildataframe['K'], sildataframe['Silhouette Score'], marker='o')
plt.title('Silhouette Scores vs Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

# Find optimal K
optimal_k = sil_scores.index(max(sil_scores)) + 2
print("Optimal K based on highest silhouette score:", optimal_k)

# Activity 4: Fit and visualize clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=10, n_init=10)
df['cluster'] = kmeans.fit_predict(data_3d)

print(df.head())

# Optional: visualize clusters
plt.figure(figsize=(8,6))
plt.scatter(df['Weight'], df['CO2'], c=df['cluster'], cmap='rainbow')
plt.title(f'K-Means Clustering (K={optimal_k})')
plt.xlabel('Weight')
plt.ylabel('CO2')
plt.show()
