import numpy as np
import itertools
from scipy.optimize import linprog
import sys
sys.path.append('../Youtube/Simplex')

from simplex import *
from electricity_utils import generate_nodes, generate_demand

# Initialize Set
size = 30
G = range(3)
N = range(size)

# Generate data 
distance_matrix, coordinates, generators = generate_nodes(size)
lower_demand = generate_demand(size)
upper_demand = generate_demand(size) + lower_demand

# Generator capacity and cost data
minimum = [2000, 500, 300]
maximum = [8000, 3000, 5000]
cost = [5, 0.1, 10]

# Decision variable indices
X_indices = {i: idx for idx, i in enumerate(N)}  # Indices for generation variables X[i]
Y_indices = {i: idx + len(X_indices) for idx, i in enumerate(N)}  # Indices for consumption Y[i]
W_indices = {(i, j): idx + len(X_indices) + len(Y_indices) for idx, (i, j) in enumerate([(i, j) for i in N for j in N if distance_matrix[i][j] != np.inf and distance_matrix[i][j] != 0])}  # Indices for flow W[i,j]

# Total number of decision variables
num_vars = len(X_indices) + len(Y_indices) + len(W_indices)

# Estimated number of constraints (slack variable tracking)
num_constraints = 2 * len(list(itertools.chain.from_iterable(generators))) + 2 * len(N) + len(N)
slack_counter = 0  # Track slack variable indexing

# Initialize matrices A, b, and c
A = []
b = []
c = np.zeros(num_vars + num_constraints)  # Extend c for slack variables

# Objective coefficients for Y[i] (profit) and X[i] (generation cost)
for i in N:
    c[Y_indices[i]] = 13  # Profit for consumption Y[i]

for g in G:
    for i in generators[g]:
        c[X_indices[i]] = -cost[g]  # Cost for generation X[i]

# (1) Generator capacity constraints (minimum and maximum generation)
for g in G:
    for i in generators[g]:
        # Maximum capacity constraint: X[i] + slack == max_g
        A_max = np.zeros(num_vars + num_constraints)  # Space for slack variable
        A_max[X_indices[i]] = 1
        A_max[num_vars + slack_counter] = 1  # Slack variable for max constraint
        A.append(A_max)
        b.append(maximum[g])
        slack_counter += 1

        # Minimum capacity constraint: X[i] - slack >= min_g
        A_min = np.zeros(num_vars + num_constraints)  # Space for slack variable
        A_min[X_indices[i]] = -1
        A_min[num_vars + slack_counter] = 1  # Slack variable for min constraint
        A.append(A_min)
        b.append(-minimum[g])
        slack_counter += 1

# (2) Not Generator constraints (restrict generation at non-generator nodes)
for i in N:
    if i not in itertools.chain.from_iterable(generators):
        # Not Generator: X[i] should be 0
        A_not_gen = np.zeros(num_vars + num_constraints)  # Space for slack variable
        A_not_gen[X_indices[i]] = 1  # X[i] should be <= 0
        A.append(A_not_gen)
        b.append(0)  # X[i] <= 0
        slack_counter += 1

# (3) Demand constraints (lower and upper bounds for consumption)
for i in N:
    if i not in itertools.chain.from_iterable(generators):
        # Lower demand: Y[i] - slack >= lower_demand[i]
        A_lower = np.zeros(num_vars + num_constraints)  # Space for slack variable
        A_lower[Y_indices[i]] = -1
        A_lower[num_vars + slack_counter] = 1  # Slack variable for lower bound
        A.append(A_lower)
        b.append(-lower_demand[i])
        slack_counter += 1

        # Upper demand: Y[i] + slack <= upper_demand[i]
        A_upper = np.zeros(num_vars + num_constraints)  # Space for slack variable
        A_upper[Y_indices[i]] = 1
        A_upper[num_vars + slack_counter] = 1  # Slack variable for upper bound
        A.append(A_upper)
        b.append(upper_demand[i])
        slack_counter += 1
    else:
        # Add a constraint for generator nodes: Y[i] == 0 (no consumption at generator nodes)
        A_gen_demand = np.zeros(num_vars + num_constraints)
        A_gen_demand[Y_indices[i]] = 1  # Y[i] should be 0
        A.append(A_gen_demand)
        b.append(0)  # Set Y[i] to 0

# (4) Flow constraints (Conservation of flow)
for i in N:
    A_flow = np.zeros(num_vars + num_constraints)  # Space for slack variable
    A_flow[X_indices[i]] = 1  # Generation at node i (inflow)

    # Inflow from other nodes (account for distance loss)
    for (j, k) in W_indices:
        if k == i:
            A_flow[W_indices[(j, k)]] += (1 - 0.002 * distance_matrix[j, i])  # Accumulate inflow

    # Outflow to other nodes (from i to j)
    for (k, j) in W_indices:
        if k == i:
            A_flow[W_indices[(k, j)]] -= 1  # Account for outflow

    # Include energy consumption at node i
    A_flow[Y_indices[i]] = -1  # Consumption at node i (outflow)

    A.append(A_flow)
    b.append(0)  # Net inflow equals outflow

# Convert A and b to numpy arrays
A = np.array(A)
b = np.array(b)

# result = revised_simplex(c, A, b)







# tries = 0
# while True:
#     slacks = np.ones(slack_counter, dtype=int)

#     arr = np.zeros(len(c) - slack_counter, dtype=int)

#     arr[np.random.choice(len(c) - slack_counter, A.shape[0] - slack_counter, replace=False)] = 1

#     basis = np.concatenate((arr, slacks))


#     result = None
#     try:
#         result = revised_simplex(c, A, b, init_basis = basis)
#     except Exception as e:
#         tries += 1
#         print(tries)
    
#     if result != None:
#         print(result)
#         break


# # Phase I: Add artificial variables and solve the auxiliary problem
# A_phase1, b_phase1, c_phase1, num_artificial_vars = add_artificial_variables(A, b)

# # Use Simplex method to solve Phase I (finding feasible basis)
# result_phase1 = revised_simplex(c_phase1, A_phase1, b_phase1)

# print(result_phase1)

# # If feasible solution found (i.e., artificial variables driven to zero)
# if result_phase1:
#     print("Feasible solution found in Phase I")
    
#     # Proceed to Phase II: Solve the original problem with feasible basis
#     basis_phase1 = []
#     for i in result_phase1[0]:
#         if i > 0.1:
#             basis_phase1.append(1)
#         else:
#             basis_phase1.append(0)
#     print(len(basis_phase1))
#     print(len(c))
#     result_phase2 = phase_ii(c, A, b, basis=basis_phase1)
#     print("Optimal solution found in Phase II:", result_phase2.x)
# else:
#     print("Problem is infeasible in Phase I")


bounds = [(0, None) for _ in range(num_vars + num_constraints)]

res = linprog(-c, A_eq=A, b_eq=b, method='Simplex')

if res.success:
    print("\nOptimal solution found:")
    print("Objective value (minimized cost):", res.fun)
    print("Optimal decision variables (X, Y, W, and slack):\n", res.x)
else:
    print("Optimization failed:", res.message)