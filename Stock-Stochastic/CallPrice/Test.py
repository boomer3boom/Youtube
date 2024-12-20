import numpy as np
import math
import yfinance as yf
import pandas as pd

# Define the ticker symbol for S&P 500
ticker = "^GSPC"

# Set the date range
start_date = "2020-01-01"
end_date = "2021-01-01"

# Download the data with weekly frequency
sp500_data = yf.download(ticker, start=start_date, end=end_date, interval="1wk")

# Extract the closing prices into a list
prices = sp500_data['Close'].tolist()

Rule = {2800: 23, 2900: 9, 2000: 18, 2700: 24, 3000: 29, 3800: 0, 4100: 5, 4300: 14, 3300: 18, 2400: 21, 2100: 7, 3700: 4, 4200: 0, 3500: 22, 4000: 6, 4400: 28, 2200: 7, 2500: 10, 2600: 12, 2300: 25, 3400: 0, 3600: 4, 3200: 21, 3100: 10, 3900: 24}

Monthly_Income = 250000

Capital = 3000000

def month_buy(prices):    
    # Take prices every 4 weeks
    four_week_prices = prices[::4]
    
    # Initialize variables
    shares_owned = 0
    total_spent = 0
    
    # Buy shares every 4 weeks
    for price in four_week_prices:
        if pd.notna(price):  # Ensure price is not NaN
            max_shares = Monthly_Income // price  # Calculate max shares you can buy
            total_cost = max_shares * price
            
            # Update shares owned and total spent
            shares_owned += max_shares
            total_spent += total_cost
    
    # Calculate final value of stocks owned at the last price
    final_price = prices[51]
    if pd.notna(final_price):
        final_value = shares_owned * final_price
    else:
        final_value = 0
    
    # Calculate ROI
    roi = shares_owned * final_price / total_spent 
    net_income = final_value - total_spent

    return roi, net_income

print(month_buy(prices))

def f(Bought_Amount, Price):

    # Initialize total return
    total_spending = 0
    stocks_bought = 0
    total_money = 0
  
    last_price_range = None
    # Loop over the weekly prices
    for p_week in Price:
        # round p_week to the nearest hundred
        rounded_price = math.floor(p_week / 100) * 100

        if Bought_Amount[rounded_price] > 0.8 and rounded_price != last_price_range:
            last_price_range=rounded_price
            stocks_bought += Bought_Amount[rounded_price]
            total_spending += Bought_Amount[rounded_price] * p_week 
            #total_spending += Bought_Amount[rounded_price] * p_week + brokerage(Bought_Amount[rounded_price] * p_week)
    
    if total_spending > Capital:
        print("Over Budget")


    roi = stocks_bought*Price[51]/total_spending
    net_income = stocks_bought*Price[51] - total_spending

    return roi, net_income

print(f(Rule, prices))