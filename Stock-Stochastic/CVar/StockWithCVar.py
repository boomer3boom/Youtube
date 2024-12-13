# Suppose a company needs to invest in talents (ie employees) and they preform differently everyday ish..

import gurobipy as gp
import numpy
import csv
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# List of stocks/ETFs
tickers = ["WEB.AX", "BOQ.AX", "CBA.AX", "NAB.AX", "WBC.AX", "ORCL", "SEMI.AX", "MSFT", "AMZN", "TEAM", "GOOG", "META", "AAPL", "AMD", "INTC", "TSMC34.SA", "DTL.AX", "NDQ.AX", "IBM", "WDAY", "HACK.AX", "ADBE", "RMD.AX", "EDOC", "ACDC.AX", "SAP", "FLT.AX", "ORG.AX", "SNOW", "IOZ.AX", "CRED.AX", "SPY", "BTC-USD", "ETHI.AX", "SYI.AX", "IEM.AX", "QTUM", "ITA"]  # Replace with your stock/ETF list
etfs = ["NDQ.AX", "HACK.AX", "EDOC", "ACDC.AX", "IOZ.AX", "CRED.AX", "SPY", "SYI.AX", "IEM.AX", "QTUM", "ITA", "SEMI.AX"]
must_have = ["NDQ.AX", "HACK.AX", "SPY", "SYI.AX", "QTUM", "IEM.AX"]
tickers = sorted(tickers)

# Define the time range
start_date = '1980-01-01'  # Adjust as needed
end_date = '2024-12-13'    # Adjust as needed

# Retrieve data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Ensure data is in chronological order
data = data.sort_index()

# Compute 6-month percentage change
returns_6m = data.pct_change(periods=126)  # ~126 trading days ≈ 6 months

# Calculate the mean 6-month change
mean_6m_change = (1 + returns_6m).mean()  # Add 1 to pct_change to get the ratio

print(mean_6m_change)

# Compute the covariance matrix
cov_matrix = returns_6m.cov()

R = mean_6m_change.tolist()
W = cov_matrix.values.tolist()
N = range(len(tickers))

for i in N:
    W[i][i] += 0.005
m = gp.Model('SAA CVar')
alpha = 0.05
Lambda = 0.80

S = range(100000)

RS = numpy.random.multivariate_normal(R, W, len(S)).tolist()

X = {i: m.addVar() for i in N}
Beta = {s: m.addVar() for s in S}
Betam = {s: m.addVar() for s in S}
Var = m.addVar()
CVar = m.addVar()


m.addConstr(gp.quicksum(X[i] for i in N)==1)

m.addConstr(gp.quicksum(X[i] for i in N if tickers[i] not in etfs)<=0.4)

NotTooMuch = {
    i: m.addConstr(X[i] <= 0.4)
    for i in N
}

MustHaveActive = {
    i: m.addConstr(X[i] >= 0.05)
    for i in N if tickers[i] in must_have
}

SetBeta = {s: m.addConstr(Beta[s] == gp.quicksum(RS[s][i]*X[i] for i in N))
            for s in S}

Setbetam = {s: m.addConstr(Beta[s]+Betam[s] >= Var) for s in S}

m.addConstr(CVar == Var - gp.quicksum(Betam[s] for s in S)/(alpha*len(S)))

#m.setParam('OutputFlag', 0)
Ret = []
CV = []
LV = []
m.setObjective(Lambda*gp.quicksum(Beta[s] for s in S)/len(S)+(1-Lambda)*CVar, gp.GRB.MAXIMIZE)
m.optimize()
# for l in range(1, 100):
#     Lambda = 0.01 * l

#     m.setObjective(Lambda*gp.quicksum(Beta[s] for s in S)/len(S)+(1-Lambda)*CVar, gp.GRB.MAXIMIZE)

#     m.optimize()
#     Ret.append(sum(Beta[s].x for s in S)/len(S))
#     CV.append(CVar.x)
#     LV.append(Lambda)

# plt.plot(LV, Ret)
# plt.plot(LV, CV)
# plt.show()
# Buy Goverment Bonds or keep in bank with interest
# RiskFree=True
# if RiskFree:
#     R.append(1.0275)
#     for i in N:
#         W[i].append(0.0)
#     N = range(len(tickers)+1)
#     W.append([0.0 for i in N])

for i in N:
    # if i == N[-1]:
    #     print("Bank", X[i].x)
    if X[i].x > 0.0001:
        print(tickers[i], X[i].x)