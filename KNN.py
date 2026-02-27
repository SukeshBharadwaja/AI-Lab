import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# Generate a small dataset
np.random.seed(42)
X = np.sort(5 * np.random.rand(20, 1), axis=0) # Feature
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0]) # Target with noise
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Implement KNN Regression
k = 3 # Number of neighbors
knn_regressor = KNeighborsRegressor(n_neighbors=k, weights='uniform')
knn_regressor.fit(X_train, y_train)
# Predict on test set
y_pred = knn_regressor.predict(X_test)
# Compute Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
# Plot results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, 'ro', label='Predictions')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title(f"KNN Regression (k={k})")
plt.legend()
plt.show()