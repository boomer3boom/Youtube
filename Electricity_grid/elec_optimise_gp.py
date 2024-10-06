from electricity_utils import *
import gurobipy as gp
import numpy as np
import itertools

size = 30
m = gp.Model('Electric Grid')

#### Sets ####
N = range(size)
G = range(3)


#### Data ####
distance_matrix, coordinates, generators = generate_nodes(size)

minimum = [2000, 500, 300]
maximum = [8000, 3000, 5000]

cost = [5, 0.1, 10]

lower_demand = generate_demand(size)

upper_demand = generate_demand(size) + lower_demand

#### Variable ####
# How much to generate at node n
X = {
    i: m.addVar()
    for i in N
}

# How much to consume at node n
Y = {
    i: m.addVar()
    for i in N
}

# How much electricity to send through on the edge connecting i and j
W = {
    (i, j): m.addVar()
    for i in N for j in N if distance_matrix[i][j] != np.inf and distance_matrix[i][j] != 0
}

#### Objective ####
m.setObjective(
    13*gp.quicksum(Y[i] for i in N) - gp.quicksum(cost[g] * X[i] for g in G for i in generators[g]),
    gp.GRB.MAXIMIZE
)


#### Constraints ####
# (1): Respect the capacity of each generator
MaxCapacity = {}
MinCapacity = {}
NotGenerator = {}
for i in N:
    for g in G:
        if i in generators[g]:
            MaxCapacity[i] = m.addConstr(X[i] <= maximum[g])
            

            MinCapacity[i] = m.addConstr(X[i] >= minimum[g])
        elif i not in list(itertools.chain.from_iterable(generators)):
            NotGenerator[i] = m.addConstr(X[i] <= 0)

# (2): Ensure consumption is within lower and upper demand bound
MinDemand = {}
MaxDemand = {}
GeneratorDemand = {}
for i in N:
    if i not in list(itertools.chain.from_iterable(generators)):
        MinDemand[i] = m.addConstr(Y[i] >= lower_demand[i])
        MaxDemand[i] = m.addConstr(Y[i] <= upper_demand[i])
    else:
        for g in G:
            if i in generators[g]:
                GeneratorDemand[i] = m.addConstr(Y[i] == 0)

# (3): Conservation of flow
# Inflow == Outflow
ConservationOfFlow = {}
for i in N:
    ConservationOfFlow[i] = m.addConstr(X[i] + gp.quicksum((1-0.002*distance_matrix[j, i])*W[j, ii] for (j, ii) in W if ii == i) == Y[i] + gp.quicksum(W[i, j] for (ii, j) in W if ii == i))

m.optimize()

# Insight and Analysis
plot_electric_grid(coordinates, distance_matrix, generators, X, W)