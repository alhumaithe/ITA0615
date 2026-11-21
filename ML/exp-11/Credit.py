from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate sample credit score data
X, y = make_classification(
    n_samples=200,
    n_features=4,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Credit Score Classification")
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))