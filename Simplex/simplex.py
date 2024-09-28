import numpy as np
import pandas as pd

def revised_simplex(c, A, b):
    m, n = A.shape

    basis = list(range(n - m, n)) 
    non_basis = list(range(n-m))

    B = A[:, basis]
    B_inv = np.linalg.inv(B)
    x_B = np.dot(B_inv, b)
    count = 0
    while True:

        # Determine the Entering Variable
        c_B = c[basis]
        y = np.dot(B_inv.T, c_B)

        p_j = A[:, non_basis]

        reduced_cost = c[non_basis] - np.dot(y.T, p_j)

        if np.all(reduced_cost <= 0):
            # Optimal Solution Found
            x = np.zeros(n)
            x[basis] = x_B
            return x, np.dot(c, x), count
        
        # choose j_star
        j_star = np.argmax(reduced_cost)
        entering = non_basis[j_star]

        # Determine the leaving variable
        p_j_star = A[:, entering]
        alpha_j_star = np.dot(B_inv, p_j_star)

        if np.all(alpha_j_star <= 0):
            raise ValueError("Problem is Unbounded")
        
        positive_indices = alpha_j_star > 0
        ratios = np.divide(x_B[positive_indices], alpha_j_star[positive_indices])

        if ratios.size == 0:
            raise ValueError("Problem is Unbounded")
        
        r_star = np.argmin(ratios)

        leaving = basis[np.where(positive_indices)[0][r_star]]

        # Do the Swap
        leaving_index = basis.index(leaving)
        entering_index = non_basis.index(entering)
        basis[leaving_index] = entering
        non_basis[entering_index] = leaving

        basis.sort()
        non_basis.sort()

        B = A[:, basis]
        B_inv = np.linalg.inv(B)

        x_B = np.dot(B_inv, b)
        count += 1

# Coefficients of the objective function (c)
c = np.array([20, 30, 50, 0, 0,0])

# Coefficients matrix for constraints (A)
A = np.array([
    [5, 8, 15, 1, 0, 0],
    [3, 4, 8, 0, 1, 0],
    [1, 2, 4, 0, 0, 1]
])

# Right-hand side of constraints (b)
b = np.array([300, 200, 30])

print(revised_simplex(c, A, b))