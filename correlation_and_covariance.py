import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

correlation_matrix = data.corr()
covariance_matrix = data.cov()

print("Correlation Matrix:")
print(correlation_matrix)
print("\nCovariance Matrix:")
print(covariance_matrix)

with open("iris_correlation_covariance.csv", "w") as f:
    f.write("Correlation Matrix\n")
    correlation_matrix.to_csv(f)
   
    f.write("\nCovariance Matrix\n")
    covariance_matrix.to_csv(f)

print("\nBoth correlation and covariance matrices saved in 'iris_correlation_covariance.csv'.")
