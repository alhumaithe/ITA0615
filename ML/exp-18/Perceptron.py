from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Use first two flower classes
X = iris.data[:100, :2]
y = iris.target[:100]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Perceptron model
model = Perceptron()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Predicted Values:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))