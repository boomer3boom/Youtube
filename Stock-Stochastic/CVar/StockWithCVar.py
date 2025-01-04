# Suppose a company needs to invest in talents (ie employees) and they preform differently everyday ish..

# Each industry can only have X amount
# No single stock (Including those from etf) can carry over 30% of my portfolio.
# Consider the PE to be between 
import gurobipy as gp
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import pandas as pd
from sklearn.covariance import ledoit_wolf_shrinkage, EmpiricalCovariance


############ Collecting All The Data ############

start_date = '1980-01-01'
end_date = '2024-12-13'

# List of tickers
tickers = [
    "SEMI.AX", "NDQ.AX", "HACK.AX", "EDOC", "ACDC.AX", 
    "IOZ.AX", "SPY", "BTC-USD", "ETHI.AX",
    "SYI.AX", "IEM.AX", "QTUM", "IOO.AX", "DHHF.AX"
]

theme_based_etf = ["SEMI.AX", "HACK.AX", "EDOC", "ACDC.AX", "QTUM"]

TL = {
    "SEMI.AX": {"USA": 0.662, "Taiwan": 0.134, "Netherlands": 0.109, "Japan": 0.048, "Korea": 0.019, "Ireland": 0.013, "Germany": 0.013},
    "NDQ.AX": {"USA": 1.0},  # Assuming US-focused Nasdaq index exposure
    "HACK.AX": {"USA": 0.69, "Israel": 0.153, "Japan": 0.117, "Korea": 0.043}, # Using Bugg here as this is what I would buy.
    "EDOC": {"USA": 0.811, "China": 0.098, "Japan": 0.045, "Netherlands": 0.035},  # Example of multiregion distribution
    "ACDC.AX": {"Japan": 0.244, "USA": 0.228, "Germany": 0.119, "Korea": 0.110, "Australia": 0.052, "France": 0.035, "Taiwan": 0.034, "Canada": 0.034, "China": 0.031, "Switzerland": 0.031, "Finland": 0.029},
    "IOZ.AX": {"Australia": 1.0},
    "SPY": {"USA": 1.0}, 
    "BTC-USD": {"Global": 1.0},
    "ETHI.AX": {"USA": 0.817, "Japan": 0.069, "Netherlands": 0.03, "Denmark": 0.017, "Britain": 0.017, "Canada": 0.014, "Finland": 0.01},
    "SYI.AX": {"Australia": 1.0},
    "IEM.AX": {"China": 0.2766, "Taiwan": 0.1969, "India": 0.1939, "Korea": 0.09, "Saudi Arabia": 0.04, "Brazil": 0.04, 'South Africa': 0.0295, 'Mexico': 0.0179, 'Malaysia': 0.0148, 'Indonesia': 0.0146, 'Thailand': 0.0142},  # Broad emerging market exposure
    "QTUM": {"USA": 0.565, "Japan": 0.124, "Netherlands": 0.072, "Taiwan": 0.056, "Island": 0.047, "Ireland": 0.018, "India": 0.017, "Finland": 0.017, "Israel": 0.017, "France": 0.016, 'Italy': 0.015, 'Switzerland': 0.015, 'Germany': 0.012, 'Norway': 0.011},
    "IOO.AX": {"USA": 0.826, "UK": 0.0387, "Switzerland": 0.0282, 'France': 0.0249, 'Japan': 0.0219, 'Germany': 0.0209, "China": 0.0113, "Netherlands": 0.0113},
    "DHHF.AX": {"USA": 0.424, "Australia": 0.37, "Japan": 0.038, "China": 0.018, "Canada": 0.018, "Britain": 0.016, "India": 0.014, "Taiwan": 0.014, "Germany": 0.011}
}

# Extract all unique countries from the TL dictionary
unique_countries = set()

for regions in TL.values():
    unique_countries.update(regions.keys())

# Store the unique countries in the variable "Geography"
Geography = sorted(unique_countries)  # Sorting for better readability

# Transpose the dictionary to have countries as keys and tickets as values
transposed_TL = {}

for ticket, countries in TL.items():
    for country, weight in countries.items():
        if country not in transposed_TL:
            transposed_TL[country] = {}
        transposed_TL[country][ticket] = weight

# Assume IT to incorporate everythin
TT = {
    "SEMI.AX": {"Semiconductor": 1.0},
    "NDQ.AX": {"IT": 0.5558, "Semiconductor": 0.185, "Cyber": 0.023, "Consumer":0.1037, "Health": 0.0494, "Industrial": 0.0457, "Materials": 0.0125, "Utilities": 0.0123, "Financials": 0.0054, "Energy": 0.0054, "Real": 0.0018},
    "HACK.AX": {"Cyber": 1.0},
    "EDOC": {"Health": 1.0},
    "ACDC.AX": {"Energy": 1.0},
    "IOZ.AX": {"Financials": 0.3362, "Materials": 0.186, "Health": 0.0967, "Consumer": 0.1511, "Industrial": 0.0718, "Real": 0.069, "Energy": 0.0384, "IT": 0.0323, "Utilities": 0.0142}, 
    "SPY": {"IT": 0.2083, "Semiconductor": 0.11, "Cyber": 0.008, "Financials": 0.1344, "Consumer": 0.1979, "Health": 0.1058, "Industrial": 0.0769, "Energy": 0.0344, "Utilities": 0.0264, "Real": 0.0224, "Materials": 0.0185},
    "BTC-USD": {"Crypto": 1.0},
    "ETHI.AX": {"IT": 0.229, "Semiconductor": 0.08, "Financials": 0.249, "Health": 0.147, "Consumer": 0.186, "Industrial": 0.071, "Real": 0.031, "Utilities": 0.004, "Materials": 0.003},
    "SYI.AX": {"Financials": 0.4467, "Materials": 0.1273, "Energy": 0.1179, "Consumer": 0.1945, "Industrial": 0.0518, "Utilities": 0.0411, "Health": 0.0184, "IT": 0.24},
    "IEM.AX": {"IT": 0.1316, "Semiconductor": 0.11, "Financials": 0.2355, "Consumer": 0.2732, "Industrial": 0.0659, "Materials": 0.0572, "Energy": 0.049, "Health": 0.038, "Utilities": 0.0266, "Real": 0.0164}, 
    "QTUM": {"Quantum": 0.42, "IT": 0.430, "Semiconductor": 0.119, "Crypto":0.0345},
    "IOO.AX": {"IT": 0.2965, "Semiconductor": 0.1533, "Consumer": 0.2611, "Financials": 0.093, "Health": 0.0904, "Industrial": 0.0431, "Energy": 0.0367, "Materials": 0.0169, "Utilities": 0.0044, "Real": 0.0029}, 
    "DHHF.AX": {"IT": 0.1684, "Consumer": 0.1922, "Industrial": 0.1141, "Financials": 0.2188, "Health": 0.09719, "Energy": 0.04037, "Real": 0.04160, "Utilities": 0.01818, "Materials": 0.08854}
}

# Extract all unique countries from the TL dictionary
unique_themes = set()

for themes in TT.values():
    unique_themes.update(themes.keys())

# Store the unique countries in the variable "Geography"
Themes = sorted(unique_themes)  # Sorting for better readability

# Transpose the dictionary to have sectors as keys and tickets as values
transposed_TT = {}

for ticket, sectors in TT.items():
    for sector, weight in sectors.items():
        if sector not in transposed_TT:
            transposed_TT[sector] = {}
        transposed_TT[sector][ticket] = weight


TechTheme = ["Cyber", "Quantum", "Semiconductor"]

# Dictionary to hold the historical data for each ticker
historical_data = {}

# Loop through the tickers and retrieve the historical data
for ticker in tickers:
    data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d',
                                      auto_adjust=False)
    historical_data[ticker] = data['Adj Close']

# Convert the 'Adj Close' data into a pandas DataFrame
adj_close_df = pd.DataFrame(historical_data)

# Convert the index (dates) to a uniform timezone (UTC) to handle different time zones
adj_close_df.index = adj_close_df.index.tz_convert('UTC')

# Extract the date from the timestamp and reset the index so that 'Date' becomes a column
adj_close_df['Date'] = adj_close_df.index.date

# Reset the index to avoid ambiguity and group by 'Date'
adj_close_df_reset = adj_close_df.reset_index(drop=True)

# Group by the Date and aggregate, taking the first non-null value for each date
merged_df = adj_close_df_reset.groupby('Date', as_index=False).first()

# Convert 'Date' to datetime format and set it as the index
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.set_index('Date')

# Resample the data by 126 days (approximately 6 months), taking the first value in each period
resampled_df = merged_df.resample('126D').first()

# Calculate the percentage change in the adjusted closing price for each stock (as decimals)
# We use `pct_change()` here to calculate the percentage change over the 126-day periods
percentage_change_df = resampled_df.pct_change().dropna(how='all')

# Calculate the average 6-month percentage change across the entire DataFrame
mean_6m_change = 1 + percentage_change_df.mean(axis=0, skipna=True)

#print(mean_6m_change)
# Calculate the covariance matrix of the percentage changes
cov_matrix = percentage_change_df.cov()

#print(percentage_change_df)

#print(cov_matrix)

#eig_values = np.linalg.eigvals(cov_matrix)
#print(eig_values)

# Dictionary to hold the trailing P/E ratios for each ticker
pe_ratios = {}

# Loop through the tickers and retrieve the P/E ratio
for ticker in tickers:
    try:
        info = yf.Ticker(ticker).info
        # Extract trailing P/E ratio
        pe_ratio = info.get('trailingPE', None)
        
        # Store the data in the dictionary
        pe_ratios[ticker] = {'P/E': pe_ratio}
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        pe_ratios[ticker] = {'P/E': None}

# Convert the dictionary to a DataFrame for better visualization
pe_df = pd.DataFrame(pe_ratios).T  # Transpose to have tickers as rows
pe_df = pe_df.where(pd.notna(pe_df), None)

pe_df.loc['BTC-USD', 'P/E'] = 0
pe_df.loc['HACK.AX', 'P/E'] = 31.05

print(pe_df)

############ Start Actual Optimisation Model ############
m = gp.Model('SAA CVar')
#### Sets ####
R = mean_6m_change.tolist()
W = cov_matrix.values.tolist()
N = range(len(tickers))
S = range(200000)
G = range(len(Geography))
T = range(len(Themes))

#### Data ####
# # Increase covariance to turn into semi definite positive
# for i in N:
#     W[i][i] += 0.005

# The lowest alpha % that we care about.
alpha = 0.05

# How much I care about the expected value.
Lambda = 0.8

# Pe penalty
PE_Penalty = 0.01

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(W)

# Regularize negative eigenvalues (set them to a small positive value)
epsilon = 1e-6  # Small positive value for regularization
eigenvalues = np.maximum(eigenvalues, epsilon)  # Ensure all eigenvalues are non-negative

# Reconstruct the matrix from the adjusted eigenvalues and eigenvectors
W = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# The return for each scenario s
RS = np.random.multivariate_normal(R, W, len(S)).tolist()

#### Variables ####
X = {i: m.addVar() for i in N}
Beta = {s: m.addVar() for s in S}
Betam = {s: m.addVar() for s in S}
Var = m.addVar()
CVar = m.addVar()

Geo = {g: m.addVar() for g in G}
SufGeo = {g: m.addVar(vtype=gp.GRB.BINARY) for g in G}
SmallGeo = {g: m.addVar(vtype=gp.GRB.BINARY) for g in G}

The  = {t: m.addVar() for t in T}
SufThe = {t: m.addVar(vtype=gp.GRB.BINARY) for t in T}

#### Constraints ####

m.addConstr(gp.quicksum(X[i] for i in N)==1)

SetBeta = {s: m.addConstr(Beta[s] == gp.quicksum(RS[s][i]*X[i] for i in N))
            for s in S}

Setbetam = {s: m.addConstr(Beta[s]+Betam[s] >= Var) for s in S}

m.addConstr(CVar == Var - gp.quicksum(Betam[s] for s in S)/(alpha*len(S)))


## Geography Constraints ##

# Total geography should be 1
m.addConstr(gp.quicksum(Geo[g] for g in G)==1)

# Set the true percentage of portfolio for each geography (ie Geo[g])
GeoAndX = {
    g: m.addConstr(Geo[g] == gp.quicksum(X[i] * transposed_TL[Geography[g]][tickers[i]] 
                                         for i in N if tickers[i] in transposed_TL[Geography[g]]))
    for g in G
}

# America can only have at most 70% of my stock portfolio.
AmericaCap = {
    g: m.addConstr(Geo[g] <= 0.7)
    for g in G if Geography[g] == "USA"
}

# We consider a Geography (ie SufGeo[g]) to be active if it takes up at least 6% of my portfolio
GeoAndSufGeoOne = {
    g: m.addConstr(Geo[g] >= 0.05 * SufGeo[g])
    for g in G
}

# Need at least 4 "representing" countries in our portfolio
SufficientCountries = m.addConstr(gp.quicksum(SufGeo[g] for g in G) >= 4)


# Set when to set small geo
SmallGeoActivate1 = {
    g: m.addConstr(Geo[g] >= 0.008 * SmallGeo[g])
    for g in G
}

# Need at least 3 small countries that represent our portfolio
SufficientSmallCountries = m.addConstr(gp.quicksum(SmallGeo[g] for g in G) >= 9)


## Theme Constraints ##

# Total theme should be 1
m.addConstr(gp.quicksum(The[t] for t in T)==1)

# Set the true percentage of portfolio for each theme (ie The[t])
TheAndX = {
    t: m.addConstr(The[t] == gp.quicksum(X[i] * transposed_TT[Themes[t]][tickers[i]] 
                                         for i in N if tickers[i] in transposed_TT[Themes[t]]))
    for t in T
}

# We consider a Theme (ie SufThe[t]) to be active if it takes up at least 5% of my portfolio
TheAndSufThe = {
    t: m.addConstr(The[t] >= 0.05 * SufThe[t])
    for t in T
}

#  Need at least 5 themes with at least 5% holding in our portfolio
SufficientTheme = m.addConstr(gp.quicksum(SufThe[t] for t in T) >= 5)

# Not too much in theme based etf.
TechCap = m.addConstr(gp.quicksum(X[i] for i in N if tickers[i] in theme_based_etf) <= 0.4)


# The net PE ratio should be below 30 ideally.
m.addConstr(gp.quicksum(pe_df['P/E'][tickers[i]] * X[i] for i in N) <= 30)


NotTooMuch = {
    i: m.addConstr(X[i] <= 0.45)
    for i in N
}

LimitBTC = m.addConstr(gp.quicksum(The[t] for t in T if Themes[t] == "Crypto") <= 0.1)

# NDQHACK = m.addConstr(gp.quicksum(X[i] for i in N if tickers[i] == "NDQ.AX") >= gp.quicksum(X[i] for i in N if tickers[i] == "HACK.AX"))
# HACKSEMI = m.addConstr(gp.quicksum(X[i] for i in N if tickers[i] == "HACK.AX") >= gp.quicksum(X[i] for i in N if tickers[i] == "SEMI.AX"))
# HACKQTUM = m.addConstr(gp.quicksum(X[i] for i in N if tickers[i] == "HACK.AX") >= gp.quicksum(X[i] for i in N if tickers[i] == "QTUM"))

#### Objective ####
m.setObjective(
    Lambda * gp.quicksum(Beta[s] for s in S) / len(S) + (1 - Lambda) * CVar,
    gp.GRB.MAXIMIZE
)
m.optimize()

for i in N:
    if X[i].x > 0.0001:
        print(tickers[i], X[i].x)

total_sum = 0
for s in S:
    total_sum += Beta[s].x

print(total_sum/len(S))