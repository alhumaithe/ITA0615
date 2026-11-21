import numpy as np
from sklearn.mixture import GaussianMixture

# Dataset
X = np.array([
    [1], [2], [3], [8], [9], [10]
])

# Create EM model
model = GaussianMixture(n_components=2, random_state=42)

# Train model
model.fit(X)

# Predict clusters
y_pred = model.predict(X)

print("Data:", X.flatten())
print("Cluster:", y_pred)
print("Cluster Means:")
print(model.means_)