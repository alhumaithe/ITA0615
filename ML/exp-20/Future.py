import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Monthly sales dataset
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

y = np.array([
    100, 120, 140, 160, 180,
    200, 220, 240, 260, 280
])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict future sales
future_months = np.array([[11], [12], [13]])

predictions = model.predict(future_months)

print("Future Sales Prediction:")

for i in range(len(predictions)):
    print("Month", i + 11, "Sales:", predictions[i])

# Plot graph
plt.scatter(X, y, color='red')
plt.plot(X, model.predict(X), color='blue')

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction")
plt.show()