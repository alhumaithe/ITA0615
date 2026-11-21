import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 8, 16, 25])

# Linear Regression
linear = LinearRegression()
linear.fit(X, y)
linear_pred = linear.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
poly_pred = model.predict(X_poly)

# Display results
print("Linear Prediction:", linear_pred)
print("Polynomial Prediction:", poly_pred)

# Plot graph
plt.scatter(X, y, color='red', label='Actual')
plt.plot(X, linear_pred, label='Linear')
plt.plot(X, poly_pred, label='Polynomial')

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.savefig("linear_vs_polynomial.png")