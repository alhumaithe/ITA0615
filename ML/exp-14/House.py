import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset: House area and price
X = np.array([
    [500], [800], [1000], [1200], [1500]
])

y = np.array([
    2000000, 3200000, 4000000, 4800000, 6000000
])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict house price
area = 1000
price = model.predict([[area]])

print("House Area:", area)
print("Predicted House Price:", price[0])