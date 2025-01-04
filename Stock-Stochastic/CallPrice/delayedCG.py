import gurobipy as gp
import numpy as np
import math

# Sets
W = range(52)  # 52 weeks
S = range(500)  # Scenarios
A = range(0, 70)
P = range(2500, 4500, 100)

Prices = np.load('Stock-Stochastic/CallPrice/simulated_prices.npy')
Capital = 3000000

# Start with column that is just 1 long. (ie ((p,a))).
column = set()
for p in P:
    for a in A:
        set.add((p,a))


def f(Bought_Amount, Price, Capital):
    total_spending = 0
    stocks_bought = 0
    used = set()
    
    last_price_range = None
    for p_week in Price:
        rounded_price = math.floor(p_week / 100) * 100

        if Bought_Amount.get(rounded_price) > 0.000001 and rounded_price != last_price_range:
            last_price_range = rounded_price
            Spending = Bought_Amount[rounded_price] / 10000 * Capital
            stocks_bought += Spending / p_week
            total_spending += Spending
            used.add(rounded_price)
    
    invalids = []
    
    # Relax spending conditions for the pricing phase if necessary
    if total_spending > Capital:
        invalids.append("SpendTooMuch")
    
    # Allow for more flexibility in initial iterations
    if total_spending < 0.7 * Capital:
        invalids.append("SpendTooLittle")

    result = {'cost': total_spending, 'stocks': stocks_bought, 'used': used}
    
    if invalids:
        result["invalids"] = invalids
    else:
        result['ROI'] = stocks_bought * Price[51] / total_spending
    
    return result

# Master Problem setup (same as before but starting with a smaller subset of variables)
master = gp.Model("MasterProblem")
# Variables

# ROI for scenario s
Z = {s: master.addVar(vtype=gp.GRB.CONTINUOUS) for s in S}

# Which column to pick as our solution
C = {c: master.addVar(vtype=gp.GRB.BINARY) for c in column}

# Objective: Maximize Z
master.setObjective(gp.quicksum(Z[s] for s in S)/len(S), gp.GRB.MAXIMIZE)


# Choose Exactly 1 column
ChooseOneColumn = master.addConstr(gp.quicksum(C[c] for c in C) == 1)

# Initial Z estimation constraint: Prices[s][51]*no_of_stock/cost[s]
InitZ = {
    s: master.addConstr(Z[s] <= Prices[s][51] * ratio[s])
    for s in S
}





### Process of Coumn Generation
while True:
    # Step 1: Solve the master problem with all the simple basic columns. Get all solutions that are feasible.
    master.optimize()

    # Step 2: Add onto the feasible columns from step 1 by extending it with new P and A. (a) Make sure these new column 
    #         are feasible as well. (b)) Also need to make sure the reduced cost is negative. If (a) and (b) are not met
    #         then don't add this column.

    # Step 3: If P and A was not in step 2, then terminate and collect the optimal solution from the columns.

    # Step 4: If P and A was indeed extended, then add it onto the master problems variable.











def f(Bought_Amount, Price, Capital):
    total_spending = 0
    stocks_bought = 0
    used = set()
    
    last_price_range = None
    for p_week in Price:
        rounded_price = math.floor(p_week / 100) * 100

        if Bought_Amount.get(rounded_price) > 0.000001 and rounded_price != last_price_range:
            last_price_range = rounded_price
            Spending = Bought_Amount[rounded_price] / 10000 * Capital
            stocks_bought += Spending / p_week
            total_spending += Spending
            used.add(rounded_price)
    
    invalids = []
    
    # Relax spending conditions for the pricing phase if necessary
    if total_spending > Capital:
        invalids.append("SpendTooMuch")
    
    # Allow for more flexibility in initial iterations
    if total_spending < 0.7 * Capital:
        invalids.append("SpendTooLittle")

    result = {'cost': total_spending, 'stocks': stocks_bought, 'used': used}
    
    if invalids:
        result["invalids"] = invalids
    else:
        result['ROI'] = stocks_bought * Price[51] / total_spending
    
    return result

def Callback(model,where):
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
                model.cbLazy(ratio[s] <= result['stocks']/result['cost'] + (0.6 * gp.quicksum(1-BA[p, a] for (p,a) in Current)))

        
        if count >= 50:
            if OverUsed:
                SpendTooMuch += 1
                model.cbLazy(gp.quicksum(BA[p,a] for p in OverUsed for a in A if a >= Dict_BA[p]) <= len(OverUsed) - 1)
            
            if UnderUsed:
                SpendTooLittle += 1
                model.cbLazy(gp.quicksum(BA[p,a] for p in UnderUsed for a in A if a <= Dict_BA[p]) <= len(UnderUsed) - 1)
            
            # ensure this entire solution is forbidden
            model.cbLazy(gp.quicksum(BA[p,a] for (p,a) in Current) <= len(Current) - 1)
        else:
            for s in S:
                result = Scenario_Result[s]
                model.cbLazy(Z[s] <= result['ROI'] + (0.6 * gp.quicksum(1-BA[p, a] for (p,a) in Current)))


def true_cost_of_pa(p, a, Prices):
    count = 0
    results = []
    for s in S:
        total_spending = 0
        stocks_bought = 0
        
        last_price_range = None
        for p_week in Prices[s]:
            rounded_price = math.floor(p_week / 100) * 100
            last_price_range = rounded_price

            if rounded_price==p and rounded_price != last_price_range and total_spending < Capital:
                Spending = a / 10000 * Capital
                stocks_bought += Spending / p_week
                total_spending += Spending
            
        
        invalids = []
        if total_spending > Capital:
            count += 1
        
        if total_spending < 0.7*Capital:
            count += 1
        
        results.append(stocks_bought*Prices[s][51]/total_spending)

    if count >= 25:
        return "Invalid"
    else:
        # return average result
        average_result = sum(results) / len(results)
        return average_result


def pricing_problem(current_P, current_A, full_P, full_A, Prices, Capital, dual_vars):
    """
    Pricing problem to identify new variables with negative reduced cost.

    Args:
        current_P: Current set of P values in the master problem.
        current_A: Current set of A values in the master problem.
        full_P: Full range of possible P values (e.g., range(2500, 3500, 100)).
        full_A: Full range of possible A values (e.g., range(4000, 6000, 5)).
        Prices: Price data for all scenarios.
        Capital: Available capital for investment.
        dual_vars: Dual variables from the master problem, used for reduced cost computation.
    
    Returns:
        List of new columns (p, a) to add to the master problem.
    """
    # Track potential new variables (p, a) and their reduced costs
    reduced_cost = {}

    # Explore new combinations of P and A
    for p in full_P:
        if p in current_P:  # Skip variables already in the master problem
            continue
        for a in full_A:
            if a in current_A:  # Skip variables already in the master problem
                continue
            
            # Evaluate the cost and constraints of the new (p, a)
            cost = true_cost_of_pa(p, a, Prices)

            if cost != "Invalid":
                # Compute reduced cost: objective contribution - dual variable adjustment
                # Placeholder logic: Replace `dual_vars[p]` with correct dual variable logic
                reduced_cost[(p, a)] = cost - dual_vars.get(p, 0)  # Adjust based on master problem structure

    # Find new variables with negative reduced cost
    new_columns = [var for var, cost in reduced_cost.items() if cost < 0]
    
    return new_columns


def column_generation():
    # Start with the initial set of variables
    Dict_BA = {}
    
    while True:
        # Step 1: Solve the master problem with the current set of variables until feasbility reached
        master.setParam('LazyConstraints',1)
        master.setParam('MIPFocus', 1)
        master.optimize(Callback)
        
        # Step 2: Get the current solution for BA (binary assignments)
        BAV = master.cbGetSolution(BA)
        for (p, a) in BA:
            if BAV[p, a] >= 0.5:
                Dict_BA[p] = a

        # Step 3: Solve the pricing problem to find negative reduced cost columns
        new_columns = pricing_problem(Dict_BA, Prices, Capital)
        
        # If no new columns are found, terminate
        if not new_columns:
            break
        
        # Step 4: Add new columns (variables) to the master problem
        for (p, a) in new_columns:
            BA[p, a] = master.addVar(vtype=gp.GRB.BINARY)
        
        # Re-solve the master problem with the new columns added
        master.optimize()


# Start the column generation process
column_generation()
