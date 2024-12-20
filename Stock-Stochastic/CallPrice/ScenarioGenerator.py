import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch historical data
ticker = '^GSPC'
start_date = '1980-01-01'
end_date = '2024-12-01'
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Resample to weekly data and calculate weekly log returns
data_weekly = data['Adj Close'].resample('W').last()
weekly_returns = np.log(data_weekly / data_weekly.shift(1)).dropna()

# Compute rolling weekly volatility (standard deviation) over a short window
window = 4  # Rolling window in weeks
weekly_volatility = weekly_returns.rolling(window).std().dropna()

# Combine into a dataframe
historical_stats = pd.DataFrame({
    'mu': weekly_returns,
    'sigma': weekly_volatility
}).dropna()

# Fit mu (weekly returns) to a normal distribution
mu_mean, mu_std = norm.fit(historical_stats['mu'])

# Define mu bands with closed range from -infinity to infinity
num_bands = 8  # Number of bands

# Manually define the quantiles (from -infinity to infinity)
quantiles = pd.qcut(historical_stats['mu'], num_bands, labels=[f"Band {i}" for i in range(num_bands)], retbins=True)[1]

# Adjust the first and last band ranges to include -infinity and +infinity
quantiles[0] = -float('inf')  # Set the lower bound of the first band to -infinity
quantiles[-1] = float('inf')  # Set the upper bound of the last band to +infinity

# Create a new 'mu_band' column based on the quantile ranges
historical_stats['mu_band'] = pd.cut(historical_stats['mu'], bins=quantiles, labels=[f"Band {i}" for i in range(num_bands)])

# Add variance column
historical_stats['variance'] = historical_stats['sigma'] ** 2

# Generate histograms for each mu band
# for i in range(num_bands):
#     band_data = historical_stats[historical_stats['mu_band'] == f"Band {i}"]
#     plt.figure(figsize=(6, 4))
#     plt.hist(band_data['variance'], bins=20, color='skyblue', edgecolor='black')
#     plt.title(f'Variance Distribution for Mu Band {i}', fontsize=14)
#     plt.xlabel('Variance (\u03C3²)', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.tight_layout()
#     plt.show()

# Print the mu range for each band
for i in range(num_bands):
    lower_bound = quantiles[i]
    upper_bound = quantiles[i + 1]
    print(f"Mu range for Band {i}: ({lower_bound}, {upper_bound})")

# Precompute the grouped variances for each mu band
variance_by_band = {
    band: group['sigma'].values
    for band, group in historical_stats.groupby('mu_band')
}

# Example of how to sample for the S&P 500 simulation:
num_scenarios = 500
num_weeks = 52
#initial_price = 6050
initial_price = 3260
target_price = 3750
#target_price = 6600  # Set your target price
alpha_max = 0.1
simulated_prices = np.zeros((num_scenarios, num_weeks + 1))

for s in range(num_scenarios):
    valid_scenario = False  # Flag to check if the scenario is valid
    while not valid_scenario:  # Repeat until a valid scenario is generated
        prices = [initial_price]
        valid_scenario = True  # Assume scenario is valid until proven otherwise
        for t in range(1, num_weeks + 1):
            # Sample mu from the fitted normal distribution
            mu_sample = np.random.choice(historical_stats['mu'])

            # Identify the band for the sampled mu by comparing it with the predefined quantiles
            for i in range(num_bands):
                if quantiles[i] < mu_sample <= quantiles[i + 1]:
                    mu_band = f"Band {i}"
                    break

            # Sample sigma from the corresponding band
            if mu_band in variance_by_band:
                sigma_sample = np.random.choice(variance_by_band[mu_band])
            else:
                valid_scenario = False  # Invalidate if band not found
                break

            # Adjust price to ensure that it is either 0 or above when mu is positive, or 0 or below when mu is negative
            if mu_sample > 0:
                next_price = prices[-1]*(1+mu_sample)  # Ensure price is non-negative if mu is positive
            elif mu_sample < 0:
                next_price = prices[-1]*(1+mu_sample)   # Ensure price is non-positive if mu is negative
            
            alpha_t = alpha_max * (t / num_weeks)
            # Use the sampled mu and sigma to compute the next price
            Z = np.random.normal(0, 1)  # Standard normal random variable
            next_price = prices[-1] * np.exp((mu_sample - 0.5 * sigma_sample**2) + sigma_sample * Z)
            next_price += alpha_t * (target_price - next_price)

            next_price = round(next_price)
            prices.append(next_price)

            # Check if the price is out of bounds
            if next_price < 2000 or next_price > 4500:
               valid_scenario = False  # Invalidate the scenario
               break  # Exit the loop and restart the scenario

        if valid_scenario:  # Only save the scenario if it's valid
            simulated_prices[s, :] = prices

# Plot the simulated scenarios
plt.figure(figsize=(10, 6))
for s in range(num_scenarios):
    plt.plot(range(num_weeks + 1), simulated_prices[s, :], label=f"Scenario {s+1}")
plt.axhline(y=target_price, color='r', linestyle='--', label="Target Price")
plt.xlabel("Week")
plt.ylabel("Price")
plt.title("Simulated S&P 500 Scenarios")
plt.show()

# Save the simulated prices to a .npy file
np.save('simulated_prices.npy', simulated_prices)