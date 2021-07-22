"""
Tutorial Perceptron Simples
Referencia: 
Data: 17-07-2021
"""

# Libs
import numpy as np  
import matplotlib.pyplot as plt

# Constants
N = 50
# Generate random data
np.random.seed(1)

c0 = np.random.randn(n, 2) + np.array([-1,0])
c1 = np.random.randn(n, 2) + np.array([3, 5])

X = np.vstack((c0,c1))
y = np.vstack((np.zeros((N,1)), np.ones((N,1))))

# Plot data 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ind = y.ravel() == 1
plt.plot(X[ind, 0], X[ind, 1], 'go')
plt.plot(X[~ind, 0], X[~ind, 1], 'ro')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
xlim = (-10, 10)
ylim = (-10, 10)

def predict(w, x):
    return int(np.dot(w.T, x) >= 0)
def get_line(w):
    x1 = np.linspace(xlim[0], xlim[1], 2)
    x2 = (-w[0] - w[1]*x1)/w[2]
    return x1, x2
def get_decision_boundary()
w = np.array([0,10,1])
x1, x2 = get_line(w)
plt.plot(x1, x2, '-k')
plt.show()


# Training Perceptron
w - np.random.random(3,1)
epochs = range(200)
eta = 0.02
for epoch in epochs:
    for i in range(X.shape[0]):
        x = np.hstack([1, X[i, :]]).reshape((3,1))
        y_get = y[i]
        y_hat = predict(w, x)
        w = w + eta * (y_get - y_hat) * x