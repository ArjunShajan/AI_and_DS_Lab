import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Original Data:")
print(df.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

cov_matrix = np.cov(scaled_data.T)
print("\nCovariance Matrix (Before PCA):")
print(cov_matrix)

pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

print("\nExplained Variance Ratio (for 3 components):")
print(pca.explained_variance_ratio_)

# Bar chart for explained variance ratio
plt.figure(figsize=(7,5))
components = [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]
plt.bar(components, pca.explained_variance_ratio_, color='skyblue')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Covariance Matrix Heatmap (Before PCA)")
plt.show()

cov_pca = np.cov(pca_data.T)
print("\nCovariance Matrix (After PCA):")
print(cov_pca)

plt.figure(figsize=(6,5))
sns.heatmap(cov_pca, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Covariance Matrix Heatmap (After PCA)")
plt.show()
