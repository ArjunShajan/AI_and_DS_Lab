import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # <-- Added

# Load data
data = pd.read_csv('height_weight.csv')

# Feature and target
X = data['Height'].values.reshape(-1, 1)  
y = data['Weight'].values  

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
slope = model.coef_[0]
intercept = model.intercept_
print(f"Linear regression equation: Weight = {slope:.2f} * Height + {intercept:.2f}")

# Make predictions
y_pred = model.predict(X_test)

# Combine results
results = pd.DataFrame({
    'Height': X_test.flatten(),
    'Actual Weight': y_test,
    'Predicted Weight': y_pred
})
print("\nActual vs Predicted Weights on Test Set:")
print(results)

# Error Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, color='red', label='Regression line')

plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs Weight Linear Regression')
plt.legend()
plt.show()

# Print test data
print("X Test:\n")
print(X_test)
print("y Test:\n")
print(y_test)