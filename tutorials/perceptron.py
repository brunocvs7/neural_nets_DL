"""
Tutorial Perceptron Simples
Referencia: 
Data: 17-07-2021
"""

# Libs
import numpy as np  
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(1)
n = 50
c0 = np.random.randn(n, 2) + np.array([-1,0])
c1 = np.random.randn(n, 2) + np.array([3, 5])

X = np.vstack((c0,c1))
y = np.vstack((np.zeros((n,1)), np.ones((n,1))))

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

w = np.array([0,10,1])
x1, x2 = get_line(w)
plt.plot(x1, x2, '-k')
plt.show()
