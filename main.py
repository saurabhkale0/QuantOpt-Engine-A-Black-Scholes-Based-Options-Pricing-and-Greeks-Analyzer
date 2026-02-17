import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import (simulate_gbm_paths_antithetic, monte_carlo_option_price,
                   black_scholes_price, black_scholes_delta, black_scholes_gamma)

# Set matplotlib dark theme
def set_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = '#222222'
    plt.rcParams['axes.facecolor'] = '#222222'
    plt.rcParams['savefig.facecolor'] = '#222222'

# Parameters
S0 = 100      # Initial stock price
K = 100       # Strike price
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility
T = 1.0       # Time to maturity (years)
steps = 252   # Steps per path (daily)
n_paths = 5000

set_dark_theme()

# Simulate GBM paths with antithetic variates (variance reduction)
t, paths = simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths)

# Standard Monte Carlo pricing
mc_call = monte_carlo_option_price(paths, K, r, T, option_type='call')
mc_put = monte_carlo_option_price(paths, K, r, T, option_type='put')

# Black-Scholes pricing (analytical)
bs_call = black_scholes_price(S0, K, r, sigma, T, option_type='call')
bs_put = black_scholes_price(S0, K, r, sigma, T, option_type='put')

# Calculate Greeks using Black-Scholes
bs_call_delta = black_scholes_delta(S0, K, r, sigma, T, option_type='call')
bs_put_delta = black_scholes_delta(S0, K, r, sigma, T, option_type='put')
bs_gamma = black_scholes_gamma(S0, K, r, sigma, T)

# Price differences
diff_call = mc_call - bs_call
diff_put = mc_put - bs_put

# Prepare DataFrame for display
df_prices = pd.DataFrame({
    'Method': ['Monte Carlo (Antithetic)', 'Black-Scholes'],
    'Call Price': [mc_call, bs_call],
    'Put Price': [mc_put, bs_put]
})

df_greeks = pd.DataFrame({
    'Greek': ['Delta (Call)', 'Delta (Put)', 'Gamma'],
    'Black-Scholes': [bs_call_delta, bs_put_delta, bs_gamma]
})

# Animation setup
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Monte Carlo Option Pricing with Greeks - Hedge Fund Grade Analysis', fontsize=16, color='white', weight='bold')

# 1. Stock Price Paths
ax1 = axes[0, 0]
ax1.set_title('GBM Stock Price Paths (Antithetic Variates)', fontsize=12, color='white')
ax1.set_xlabel('Time (years)')
ax1.set_ylabel('Stock Price')

# Label parameters
params_text = f"S0: {S0}\nK: {K}\nσ: {sigma}\nr: {r}\nT: {T}\nN: {n_paths}"
param_box = ax1.text(0.02, 0.95, params_text, transform=ax1.transAxes, fontsize=10, color='white', va='top', bbox=dict(facecolor='#333333', alpha=0.7))

# Option price display
price_box = ax1.text(0.98, 0.95, '', transform=ax1.transAxes, fontsize=9, color='white', va='top', ha='right', bbox=dict(facecolor='#333333', alpha=0.7))

# Plot lines for paths
lines = [ax1.plot([], [], lw=0.8, alpha=0.4)[0] for _ in range(20)]

ax1.set_xlim(0, T)
ax1.set_ylim(np.min(paths), np.max(paths))
ax1.grid(alpha=0.2)

# 2. Greeks - Delta
ax2 = axes[0, 1]
ax2.set_title('Delta (Hedging Ratio)', fontsize=12, color='white')
ax2.axhline(y=bs_call_delta, color='cyan', linestyle='--', linewidth=2, label=f'BS Call: {bs_call_delta:.4f}')
ax2.axhline(y=bs_put_delta, color='magenta', linestyle='--', linewidth=2, label=f'BS Put: {bs_put_delta:.4f}')
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('Stock Price Movement')
ax2.set_ylabel('Delta Value')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(alpha=0.2)

# 3. Greeks - Gamma
ax3 = axes[1, 0]
ax3.set_title('Gamma (Hedging Convexity Risk)', fontsize=12, color='white')
stock_range = np.linspace(S0 * 0.7, S0 * 1.3, 50)
gamma_range = [black_scholes_gamma(s, K, r, sigma, T) for s in stock_range]
ax3.plot(stock_range, gamma_range, color='yellow', linewidth=2, label='BS Gamma')
ax3.axvline(x=S0, color='red', linestyle='--', alpha=0.5)
ax3.axhline(y=bs_gamma, color='orange', linestyle=':', linewidth=2, label=f'Current: {bs_gamma:.4f}')
ax3.set_xlabel('Stock Price')
ax3.set_ylabel('Gamma Value')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.2)

# 4. Pricing Comparison - Variance Reduction Impact
ax4 = axes[1, 1]
methods = ['Monte Carlo\n(Antithetic)', 'Black-Scholes']
call_prices = [mc_call, bs_call]
put_prices = [mc_put, bs_put]

x_pos = np.arange(len(methods))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, call_prices, width, label='Call', color='cyan', alpha=0.7)
bars2 = ax4.bar(x_pos + width/2, put_prices, width, label='Put', color='magenta', alpha=0.7)

ax4.set_title('Pricing Comparison', fontsize=12, color='white')
ax4.set_ylabel('Option Price')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods, fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.2, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='white')

# Animation function for stock price paths
def animate(i):
    for j, line in enumerate(lines):
        line.set_data(t[:i], paths[j, :i])
    # Update price box dynamically
    price_box.set_text(f"MC Call: {mc_call:.4f}\nBS Call: {bs_call:.4f}\nError: {diff_call:.4f} ({100*diff_call/bs_call:.2f}%)\n\nMC Put: {mc_put:.4f}\nBS Put: {bs_put:.4f}\nError: {diff_put:.4f} ({100*diff_put/bs_put:.2f}%)")
    return lines + [price_box]

ani = animation.FuncAnimation(fig, animate, frames=steps+1, interval=20, blit=True, repeat=False)

plt.tight_layout()
plt.show()

# Print comprehensive analysis
print("=" * 80)
print("MONTE CARLO OPTION PRICING SIMULATOR - ANALYSIS")
print("=" * 80)

print("\nOPTION PRICE COMPARISON:")
print(df_prices.to_string(index=False))

print("\nPRICING ACCURACY:")
print(f"Call Price Error: {diff_call:.6f} ({100*diff_call/bs_call:.2f}%)")
print(f"Put Price Error: {diff_put:.6f} ({100*diff_put/bs_put:.2f}%)")
print(f"Normal MC accuracy: 1-2% error is expected and correct!")

print("\n\nGREEKS - HEDGING ANALYSIS:")
print(df_greeks.to_string(index=False))

print("\nINTERPRETATION:")
print(f"Delta (Call): {bs_call_delta:.4f}")
print(f"  To hedge 100 call options, short {100*bs_call_delta:.0f} shares")
print(f"\nDelta (Put): {bs_put_delta:.4f}")
print(f"  To hedge 100 put options, long {100*abs(bs_put_delta):.0f} shares")
print(f"\nGamma: {bs_gamma:.6f}")
print(f"  Delta changes by {bs_gamma:.6f} per $1 move in stock price")
print(f"  Rebalancing frequency: important for hedging")

print("\n" + "=" * 80)
