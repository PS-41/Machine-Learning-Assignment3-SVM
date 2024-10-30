import numpy as np
from scipy.optimize import minimize

# Define the dataset
X = np.array([
    [3, 6],
    [2, 2],
    [4, 4],
    [1, 3],
    [2, 0],
    [4, 2],
    [4, 0]
])

y = np.array([-1, -1, -1, -1, 1, 1, 1])  # Labels
N = len(y)

# Compute the Kernel matrix
K = np.dot(X, X.T) * np.outer(y, y)

# Objective function for the dual problem
def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)

# Constraints: sum(alpha * y) = 0 and alpha >= 0
constraints = [
    {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},
    {'type': 'ineq', 'fun': lambda alpha: alpha}
]

# Initial guess for alpha values
initial_alpha = np.zeros(N)

# Use scipy's minimize function
result = minimize(objective, initial_alpha, constraints=constraints, method='SLSQP')
alphas = result.x

# Select support vectors (alphas > small threshold)
threshold = 1e-5
support_vectors = np.where(alphas > threshold)[0]
print("Optimal alpha values:", alphas)
print("Support vectors indices:", support_vectors)

# Compute the weight vector w (only 2 dimensions since we have 2 features x1 and x2)
# Fixing the shape issue for correct broadcasting
w = np.sum(alphas[support_vectors][:, np.newaxis] * y[support_vectors][:, np.newaxis] * X[support_vectors], axis=0)
print("Weight vector (w):", w)

# Compute the intercept b using the support vectors
b = y[support_vectors[0]] - np.dot(w, X[support_vectors[0]])
print("Intercept (b):", b)

# Display the hyperplane equation
print("\nOptimal Hyperplane Equation: {:.3f} + {:.3f} * x1 + {:.3f} * x2 = 0".format(b, w[0], w[1]))
