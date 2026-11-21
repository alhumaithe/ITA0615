import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

print("Predicted Values:", y_pred)
print("Mean Squared Error:",
      mean_squared_error(y, y_pred))

print("Prediction for 6:", model.predict([[6]]))