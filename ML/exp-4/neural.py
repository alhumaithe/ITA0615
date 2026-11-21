import numpy as np

# Input data (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights
np.random.seed(1)
w1 = np.random.randn(2, 4)
w2 = np.random.randn(4, 1)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training
for i in range(10000):
    # Forward propagation
    h = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(h, w2))

    # Backpropagation
    error = y - output
    d_output = error * output * (1 - output)

    error_hidden = d_output.dot(w2.T)
    d_hidden = error_hidden * h * (1 - h)

    # Update weights
    w2 += h.T.dot(d_output) * 0.5
    w1 += X.T.dot(d_hidden) * 0.5

print("Predicted Output:")
print(np.round(output))