import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Dataset: RAM, Battery, Internal Storage
X = np.array([
    [2, 3000, 32],
    [3, 3500, 64],
    [4, 4000, 64],
    [6, 4500, 128],
    [8, 5000, 256],
    [12, 6000, 256]
])

# Price categories
y = np.array([
    "Low", "Low", "Medium",
    "Medium", "High", "High"
])

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X, y)

# Predict new mobile price category
result = model.predict([[8, 5000, 128]])

print("Mobile Price Category:", result[0])