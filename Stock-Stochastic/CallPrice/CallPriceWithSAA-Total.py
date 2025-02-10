import gurobipy as gp
import numpy as np
import math

# Sets
W = range(52) # 52 weeks
S = range(500) 
H = range(4) # 1st quater, 2nd 3rd and 4th
#P = range(5000, 10000)
A = range(50)

def f(Bought_Amount, Price):

    # Initialize total return
    total_spending = 0
    stocks_bought = 0
  
    last_price_range = None
    # Loop over the weekly prices
    for p_week in Price:
        # round p_week to the nearest hundred
        rounded_price = math.floor(p_week / 100) * 100

        if Bought_Amount[rounded_price] > 0.8 and rounded_price != last_price_range:
            last_price_range=rounded_price
            stocks_bought += Bought_Amount[rounded_price]
            total_spending += Bought_Amount[rounded_price] * p_week 
    
    invalids = []
    # If the ending price for this scenario leans toward a cheaper price, then I want to buy more stock since i trust the s and p 500?
    if stocks_bought < 30:
        invalids.append("NeedMoreStock")
    
    if total_spending > Capital:
        invalids.append("SpendTooMuch")

    result = {}

    result['cost'] = total_spending
    result['stocks'] = stocks_bought
    result['assets'] = stocks_bought*Price[51]

    if len(invalids) > 0:
        result["invalids"] = invalids
    else:
        result['ROI'] = stocks_bought*Price[51] - total_spending

    return result
    
# Constants
Prices = np.load('Stock-Stochastic/CallPrice/simulated_prices.npy')
Capital = 3000000

# Price ranges from 5000 to 8000 with 100 dollar increment
P = range(2000, 4500, 100)

# Master Problem
master = gp.Model("MasterProblem")

# Variables
Z = {s: master.addVar(vtype=gp.GRB.CONTINUOUS) for s in S} # ROI for scenario s

BA = {(p, a): master.addVar(vtype=gp.GRB.BINARY)for p in P for a in A}

Asset = {s: master.addVar(vtype=gp.GRB.CONTINUOUS, ub = (Capital/min(Prices[s])) * Prices[s][51]) for s in S}

Cost = {s: master.addVar(vtype=gp.GRB.CONTINUOUS, lb=min(Prices[s]*30)) for s in S}

# Objective: Maximize Z
master.setObjective(gp.quicksum(Z[s] for s in S)/len(S), gp.GRB.MAXIMIZE)

OneAPerB = {
    p: master.addConstr(gp.quicksum(BA[p,a] for a in A) == 1)
    for p in P
}

# Initial Z estimation constraint: Prices[s][51]*no_of_stock/cost[s]
InitZ = {
    s: master.addConstr(Z[s] <= Asset[s]-Cost[s])
    for s in S
}

SpendTooMuch = 0
SpendTooLittle = 0
NotEnough = 0
def Callback(model,where):
    global SpendTooMuch, SpendTooLittle, NotEnough

    if where==gp.GRB.Callback.MIPSOL:
        BAV = model.cbGetSolution(BA)
        Current = {(p,a) for p in P for a in A if BAV[p,a] >= 0.5}

        Dict_BA = {}

        for (p,a) in Current:
            Dict_BA[p] = a
        
        for p in P:
            if p not in Dict_BA:
                Dict_BA[p] = 0

        for s in S:
            result = f(Dict_BA, Prices[s])
            if 'invalids' in result:
                model.cbLazy(gp.quicksum(BA[p,a] for (p,a) in Current) <= len(Current) - 1)

                if "SpendTooMuch" in result['invalids']:
                    SpendTooMuch += 1
                    # Feasibility Cut: Any amount ad that is more than the current solution must be cut off.
                    model.cbLazy(gp.quicksum(BA[p,ad] for (p,a) in Current for ad in A if a < ad) <= len(Current) - 1)
        
                if "SpendTooLittle" in result['invalids']:
                    SpendTooLittle += 1
                    # Feasibility Cut: Any amount ad that is less than the current solution must be cut off.
                    model.cbLazy(gp.quicksum(BA[p,ad] for (p,a) in Current for ad in A if a > ad) <= len(Current) - 1)
            else:
                model.cbLazy(Z[s] <= result['ROI'] + (500000 * gp.quicksum(1-BA[p, a] for (p,a) in Current)))
            
# Solve the Master Problem
master.setParam('LazyConstraints',1)
master.setParam('MIPFocus', 1)
master.setParam('Heuristics', 0.9)
master.setParam('Presolve', 2)

# Set the branch priority based on the price moving toward the average
for p in P:
    for a in A:
        # Calculate the distance from the average price
        price_distance = abs(p - 3400)

        # Calculate the priority based on the price's proximity to the average
        price_priority = 10 / (1 + price_distance)  # Higher priority for prices closer to the average

        # Calculate the branch priority as an integer
        branch_priority = int(price_priority + (10 - abs(a - 25)))  # Cast to integer
        print(branch_priority)

        # Set the branch priority for each BA[p, a] variable
        BA[p, a].setAttr('BranchPriority', branch_priority)

master.optimize(Callback)
Current = {(p,a) for p in P for a in A if BA[p,a].x >= 0.5}

Dict_BA = {}

for (p,a) in Current:
    Dict_BA[p] = a

print(Dict_BA)