from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize k-NN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print correct predictions
print("\nCorrect Predictions:")
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f'Index {i}: Predicted: {iris.target_names[y_pred[i]]}, '
              f'Actual: {iris.target_names[y_test[i]]}')

# Print incorrect predictions
print("\nIncorrect Predictions:")
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        print(f'Index {i}: Predicted: {iris.target_names[y_pred[i]]}, '
              f'Actual: {iris.target_names[y_test[i]]}')