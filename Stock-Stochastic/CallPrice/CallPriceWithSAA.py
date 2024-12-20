import gurobipy as gp
import numpy as np
import math

# Do it for this first and get a couple of choices. Then test this on 10 seperate new problems? 
# Can we do something about this with benders/callback?

# Sets
W = range(52) # 52 weeks
S = range(100) 
H = range(4) # 1st quater, 2nd 3rd and 4th
#P = range(5000, 10000)
A = range(30)

# Data
def brokerage(x):
    if x <= 100000:
        return 500
    elif x > 100000 and x <= 300000:
        return 1000
    elif x > 300000 and x <= 1000000:
        return 2000
    elif x > 1000000 and x <= 2500000:
        return 3000
    else:
        return 0.12 * x

# def splits(h):
#     if h == 0:
#         return range(0, 13)
#     elif h == 1:
#         return range(13, 26)
#     elif h == 2:
#         return range(26, 39)
#     elif h == 3:
#         return range(39, 52)

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

    if len(invalids) > 0:
        result["invalids"] = invalids
    else:
        result['ROI'] = stocks_bought*Price[51]/total_spending

    return result
    
# Constants
Prices = np.load('Stock-Stochastic/CallPrice/simulated_prices.npy')
Capital = 10000000

big_M = 10000  # Big-M constant

# Price ranges from 5000 to 8000 with 100 dollar increment
P = range(5000, 7500, 100)

# Master Problem
master = gp.Model("MasterProblem")

# Variables
Z = {s: master.addVar(vtype=gp.GRB.CONTINUOUS) for s in S} # ROI for scenario s

BA = {(p, a): master.addVar(vtype=gp.GRB.BINARY)for p in P for a in A}

ratio = {s: master.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=100) for s in S}

# Objective: Maximize Z
master.setObjective(gp.quicksum(Z[s] for s in S)/len(S), gp.GRB.MAXIMIZE)

OneAPerB = {
    p: master.addConstr(gp.quicksum(BA[p,a] for a in A) == 1)
    for p in P
}

# Initial Z estimation constraint: Prices[s][51]*no_of_stock/cost[s]
InitZ = {
    s: master.addConstr(Z[s] <= Prices[s][51] * ratio[s])
    for s in S
}
# InitZ = {
#     s: master.addConstr(Z[s] <= 2)
#     for s in S
# }

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

        #function_result = {}
        #Invalidated = False
        for s in S:
            result = f(Dict_BA, Prices[s])
            if 'invalids' in result:
                #Invalidated = True
                if "SpendTooMuch" in result['invalids']:
                    SpendTooMuch += 1
                    # Feasibility Cut: Cut the current solution off. 
                    model.cbLazy(gp.quicksum(BA[p,a] for (p,a) in Current) <= len(Current) - 1)
                    # Feasibility Cut: Any amount ad that is more than the current solution must be cut off.
                    model.cbLazy(gp.quicksum(BA[p,ad] for (p,a) in Current for ad in A if a < ad) <= len(Current) - 1)
        
                if "SpendTooLittle" in result['invalids']:
                    SpendTooLittle += 1
                    # Feasibility Cut: Cut the current solution off. 
                    model.cbLazy(gp.quicksum(BA[p,a] for (p,a) in Current) <= len(Current) - 1)
                    # Feasibility Cut: Any amount ad that is less than the current solution must be cut off.
                    model.cbLazy(gp.quicksum(BA[p,ad] for (p,a) in Current for ad in A if a > ad) <= len(Current) - 1)
                    
                if "NeedMoreStock" in result['invalids']:
                    NotEnough += 1
                    # Feasibility Cut: Choose a higher 'a' value for one of the ps
                    model.cbLazy(gp.quicksum(BA[p,a] for (p,a) in Current) <= len(Current) - 1)
            else:
                # Is there any way to consider risk?
                #model.cbLazy(Z[s] <= Prices[s][51] * ratio[s])
                model.cbLazy(Z[s] <= result['ROI'] + (0.5 * gp.quicksum(1-BA[p, a] for (p,a) in Current)))
            
            # Update the true value of cost and num_stocks
            if result['cost'] >= 0.01:
                model.cbLazy(ratio[s] <= result['stocks']/result['cost'] + (0.5 * gp.quicksum(1-BA[p, a] for (p,a) in Current)))
                
            #function_result[s] = result
        
        # print("SpendTooMuch:", SpendTooMuch)
        # print("SpendTooLittle:", SpendTooLittle)
        # print("NotEnough:", NotEnough)
        
        # # If we got to this point, then just solve the sub problem and add optimality cut
        # if not Invalidated:
        #     # Subproblem
        #     subproblem = gp.Model("SubProblem")

        #     # Subproblem Variables
        #     Beta = {s: subproblem.addVar() for s in S} 

        #     Betam = {s: subproblem.addVar() for s in S}

        #     Var = subproblem.addVar()  # Value-at-Risk
        #     CVar = subproblem.addVar()  # Conditional Value-at-Risk

        #     # Objective: Maximize (lambda * sum(beta_s))/len(S) - ((1 - lambda) * CVaR)
        #     subproblem.setObjective(
        #         lam * (gp.quicksum(Beta[s] for s in S)/len(S)) - (1 - lam) * CVar, gp.GRB.MAXIMIZE
        #     )

        #     # Constraints for subproblem
        #     # Beta is suppose to be a ratio not exact values
        #     Beta_Constraints = {
        #         s: subproblem.addConstr(Beta[s] == function_result[s])
        #         for s in S
        #     }

        #     Betam_Constraints = {
        #         s: subproblem.addConstr(Beta[s] + Betam[s] >= Var)
        #         for s in S
        #     }

        #     # CVaR definition: CVaR = VaR - (1 / (alpha * |S|)) * sum(beta_m_s)
        #     CVar_Constraint = subproblem.addConstr(
        #         CVar == Var - (1 / (alpha * len(S))) * gp.quicksum(Betam[s] for s in S),
        #     )

        #     subproblem.setParam('OutputFlag', 0)
        #     # Solve the Subproblem
        #     subproblem.optimize()

        #     # If optimization is successful, apply optimality cuts
        #     if subproblem.status == gp.GRB.OPTIMAL:
        #         # Apply Benders optimality cut
        #         cuts_added = False
            
        #         # Check if a cut is needed based on subproblem dual values
        #         if subproblem.objVal:  # Check if a cut is needed
        #             cuts_added = True
        #             #subproblem_obj_val = subproblem.objVal
                    
        #             # Retrieve dual values for Beta and Betam constraints
        #             dual_beta_s = Beta_Constraints[s].pi  # Dual value for Beta constraint

        #             # Generate Benders cut using duals of subproblem constraints
        #             master.cbLazy(
        #                 Z <= gp.quicksum(dual_beta_s * function_result[s] for s in S) * (1 + gp.quicksum(100000*(1-BA[p, a]) for (p,a) in Current))
        #             )
                  
                
        #         if not cuts_added:
        #             print("No new cuts added. Convergence reached.")
        #             return

# Solve the Master Problem

master.setParam('LazyConstraints',1)
master.setParam('MIPFocus', 1)


#master.setParam('NodeMethod', 0)  # Hybrid search to emphasize breadth.
#master.setParam('VarBranch', 2)  # Max infeasibility branching for balanced exploration.
#master.setParam('Cuts', 0)       # Reduce cuts to prevent aggressive pruning.
#master.setParam('Presolve', 0)   # Disable presolve (optional, for exhaustive exploration).
#master.optimize()
master.optimize(Callback)
Current = {(p,a) for p in P for a in A if BA[p,a].x >= 0.5}

Dict_BA = {}

for (p,a) in Current:
    Dict_BA[p] = a

print(Dict_BA)