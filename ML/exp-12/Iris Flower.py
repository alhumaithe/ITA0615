from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN model
model = KNeighborsClassifier(n_neighbors=5)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Iris Flower Classification")
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new flower
result = model.predict([[5.1, 3.5, 1.4, 0.2]])
print("New Flower:", iris.target_names[result[0]])