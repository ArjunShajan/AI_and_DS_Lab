from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
import matplotlib.pyplot as plt

import numpy as np
# Load iris dataset (multiclass)
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=62)

# Train Naive Bayes classifier
model= LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# Get absolute value of coefficients and average across classes
importance = np.mean(np.abs(model.coef_), axis=0)

# Print feature importance values
for feature_name, importance_value in zip(data.feature_names, importance):
    print(f"{feature_name}: {importance_value:.4f}")
