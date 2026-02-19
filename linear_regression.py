import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def simple_linear_regression(X, y):
    n = len(X)
    
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    
    nu = np.sum((X - mean_x) * (y - mean_y))
    de = np.sum((X - mean_x) ** 2)
    
    m = nu / de
    b = mean_y - m * mean_x
    
    return m, b

def predict(X, m, b):
    return m * X + b

# Generate Data
np.random.seed(42)
X = np.random.rand(100) * 10
y = 3 * X + 7 + np.random.rand(100) * 2

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
m, b = simple_linear_regression(X_train, y_train)

# Predict
y_pred = predict(X_test, m, b)

# Evaluate
mse = mean_squared_error(y_test, y_pred)

print(f"Estimated Slope (m): {m}")
print(f"Estimated Intercept (b): {b}")
print(f"Mean Squared Error: {mse}")

# Plot Results
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.title("Simple Linear Regression")
plt.show()
