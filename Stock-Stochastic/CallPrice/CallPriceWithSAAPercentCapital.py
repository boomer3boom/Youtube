import gurobipy as gp
import numpy as np
import math

# Sets
W = range(52)  # 52 weeks
S = range(500)  # Scenarios

def f(Bought_Amount, Price, Capital):

    # Initialize total return
    total_spending = 0
    stocks_bought = 0
    used = set()

    # Loop over the weekly prices
    for p_week in Price:
        # round p_week to the nearest hundred
        rounded_price = math.floor(p_week / 100) * 100

        if Bought_Amount[rounded_price] > 0.0001 and rounded_price not in used:
            #Bought_Amount[rounded_price] = 0
            Spending = Bought_Amount[rounded_price]/100 * Capital
            stocks_bought += Spending/p_week
            total_spending += Spending
            used.add(rounded_price)
            #total_spending += Bought_Amount[rounded_price] * p_week + brokerage(Bought_Amount[rounded_price] * p_week)
    
    invalids = []
    
    if total_spending > Capital:
        invalids.append("SpendTooMuch")

    if total_spending < 0.7*Capital:
        invalids.append("SpendTooLittle")

    result = {}

    result['cost'] = total_spending
    result['stocks'] = stocks_bought
    result['used'] = used

    if len(invalids) > 0:
        result["invalids"] = invalids
        # ie stay neutral in terms of ROI
        result['ROI'] = 1
    else:
        result['ROI'] = stocks_bought*Price[51]/total_spending

    return result

def get_prices_in_scenario(Prices):
    Scenario_Price_List = {}
    for s in S:
        price_list = set()
        for price in Prices[s]:
            if price not in price_list:
                rounded_price = math.floor(price / 100) * 100
                price_list.add(rounded_price)
        
        Scenario_Price_List[s] = price_list
    
    return Scenario_Price_List

Prices = np.load('Stock-Stochastic/CallPrice/simulated_prices.npy')
big_M = 10000  # Big-M constant
Capital = 3000000
Scenario_Price_List = get_prices_in_scenario(Prices)

P = range(2000, 4500, 100)

# represent % of cash to put in to purchase stock
A = range(0, 51, 2)

# Master Problem
master = gp.Model("MasterProblem")

# Variables
Z = {s: master.addVar(vtype=gp.GRB.CONTINUOUS) for s in S} # ROI for scenario s

BA = {(p,a): master.addVar(vtype=gp.GRB.BINARY) for p in P for a in A}

ratio = {s: master.addVar(vtype=gp.GRB.CONTINUOUS, lb=-1, ub=1) for s in S}

master.setObjective(
    gp.quicksum(Z[s] for s in S) / len(S),
    gp.GRB.MAXIMIZE
)
OneAPerB = {
    p: master.addConstr(gp.quicksum(BA[p,a] for a in A) == 1)
    for p in P
}

# Initial Z estimation constraint: Prices[s][51]*no_of_stock/cost[s]
InitZ = {
    s: master.addConstr(Z[s] <= Prices[s][51] * ratio[s])
    for s in S
}

# max_weight = len(P) * 6  # Adjust based on your problem scale
# master.addConstr(
#     gp.quicksum(a * BA[p, a] for p in P for a in A) <= max_weight

#print(Scenario_Price_List)
LimitScenario = {
    s: master.addConstr(gp.quicksum(BA[p,a]*a for p in Scenario_Price_List[s] for a in A) <= 130)
    for s in S
}

LimitScenario = {
    s: master.addConstr(gp.quicksum(BA[p,a]*a for p in Scenario_Price_List[s] for a in A) >= 40)
    for s in S
}
# Get all the Prices that exist in each scenario.
# Then just say 
# for each scenario
# sum(BA[p,a]*a for p in P for a in A if p exist in scenario) <= 1.2

SpendTooMuch = 0
SpendTooLittle = 0
def Callback(model,where):
    global SpendTooMuch, SpendTooLittle, NotEnough

    if where == gp.GRB.Callback.MIPSOL:
        BAV = model.cbGetSolution(BA)
        Current = {(p,a) for p in P for a in A if BAV[p,a] >= 0.5}

        Dict_BA = {}

        for (p,a) in Current:
            Dict_BA[p] = a

        count = 0

        OverUsed = set()
        UnderUsed = set()

        Scenario_Result = {}
        for s in S:
            result = f(Dict_BA, Prices[s], Capital)
            Scenario_Result[s] = result
            if 'invalids' in result:
                count += 1
                if "SpendTooMuch" in result['invalids']:
                    OverUsed.update(result['used'])    
            
                if "SpendTooLittle" in result['invalids']:
                    UnderUsed.update(result['used'])
            
            if result['cost'] >= 0.01:
                model.cbLazy(ratio[s] <= result['stocks']/result['cost'] + (0.4 * gp.quicksum(1-BA[p, a] for (p,a) in Current)))

        
        if count >= 25:
            if OverUsed:
                SpendTooMuch += 1
                model.cbLazy(gp.quicksum(BA[p,a] for p in OverUsed for a in A if a >= Dict_BA[p]) <= len(OverUsed) - 1)
            
            if UnderUsed:
                SpendTooLittle += 1
                model.cbLazy(gp.quicksum(BA[p,a] for p in UnderUsed for a in A if a <= Dict_BA[p]) <= len(UnderUsed) - 1)
            
            # ensure this entire solution is forbidden
            model.cbLazy(gp.quicksum(BA[p,a] for (p,a) in Current) <= len(Current) - 1)
        else:
            #returnsum = 0
            for s in S:
                result = Scenario_Result[s]
                #returnsum += result['ROI']
                model.cbLazy(Z[s] <= result['ROI'] + (0.4 * gp.quicksum(1-BA[p, a] for (p,a) in Current)))
            
            #print(Current)
            #print(returnsum/len(S))

master.setParam('LazyConstraints',1)
master.setParam('MIPFocus', 1)

middle_value = 10  # Encourage smaller values
for p in P:
    for a in A:
        distance_from_middle = abs(a - middle_value)
        priority = max(1, len(A) - distance_from_middle)  # Larger priorities for smaller values
        BA[p, a].setAttr('BranchPriority', priority)

master.optimize(Callback)
print(SpendTooMuch)
print(SpendTooLittle)

Current = {(p,a) for p in P for a in A if BA[p, a].x >= 1e-6}

#Dict_BA = {}

# for p in P:
#     print(BA[p].x)
    #Dict_BA[p] = BA[p].x

#print(Dict_BA)
print(Current)