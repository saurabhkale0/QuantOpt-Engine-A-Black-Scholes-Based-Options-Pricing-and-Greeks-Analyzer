# QuantOpt Engine: A Black-Scholes Based Options Pricing and Greeks Analyzer

A professional-grade simulator for pricing European Call and Put options using Monte Carlo methods with Geometric Brownian Motion (GBM). Includes variance reduction techniques, Greek calculations, and interactive visualizations.

## Overview

This project implements:
- Monte Carlo simulation of stock price paths using GBM
- European option pricing (Call & Put)
- Black-Scholes comparison for validation
- Antithetic variates for variance reduction
- Greeks calculation (Delta, Gamma) for hedging strategies
- Interactive visualization with animated stock paths

## Key Features

- 5,000 Monte Carlo simulations with antithetic variates
- 252 daily time steps (realistic trading days)
- Accurate pricing (typically 1-2% error vs Black-Scholes)
- Greeks analysis for hedging and risk management
- Real-time animated plots showing stock evolution
- Professional dark theme visualization

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Simulator
```bash
python main.py
```

This will:
1. Generate 5,000 GBM stock price paths
2. Calculate option prices using Monte Carlo
3. Compare with Black-Scholes analytical prices
4. Compute hedging Greeks (Delta, Gamma)
5. Display an interactive 4-panel visualization
6. Print detailed analysis to terminal

## Output

### Visualization (4-Panel Dashboard)
1. Top-Left: Animated GBM stock price paths (20 sample paths)
2. Top-Right: Delta curve (hedging ratio)
3. Bottom-Left: Gamma curve (convexity risk)
4. Bottom-Right: Price comparison (Monte Carlo vs Black-Scholes)

### Terminal Output
```
OPTION PRICE COMPARISON:
          Method  Call Price  Put Price
Monte Carlo       10.574002   5.695695
Black-Scholes     10.450584   5.573526

GREEKS - HEDGING ANALYSIS:
Delta (Call): 0.6368
  → To hedge 100 calls, short 64 shares

Gamma: 0.0188
  → Delta changes by 0.0188 per $1 move
```

## Mathematical Foundation

### Geometric Brownian Motion
S_t = S_0 * exp((r - 0.5*sigma^2)*t + sigma*W_t)

### Option Payoff (at maturity T)
- Call: max(S_T - K, 0)
- Put: max(K - S_T, 0)

### Black-Scholes Formula
- Call: C = S_0 * N(d_1) - K * exp(-r*T) * N(d_2)
- Put: P = K * exp(-r*T) * N(-d_2) - S_0 * N(-d_1)

Where:
- d_1 = (ln(S_0/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
- d_2 = d_1 - sigma*sqrt(T)

### Greeks
- Delta (Call): N(d_1)
- Delta (Put): N(d_1) - 1
- Gamma: N'(d_1) / (S_0 * sigma * sqrt(T))

## Configuration

Edit parameters in main.py:
```python
S0 = 100      # Initial stock price
K = 100       # Strike price
r = 0.05      # Risk-free rate (5%)
sigma = 0.2   # Volatility (20%)
T = 1.0       # Time to maturity (1 year)
steps = 252   # Daily time steps
n_paths = 5000  # Number of simulations
```

## Project Structure

```
├── main.py           # Entry point with visualization & analysis
├── utils.py          # GBM simulation & pricing functions
├── requirements.txt  # Python dependencies
└── README.md         # Documentation
```

## Dependencies

- numpy - Numerical computations
- pandas - Data manipulation
- matplotlib - Visualization
- scipy - Statistical functions

## Understanding the Results

### Why 1-2% Error?
Monte Carlo introduces sampling noise. This is expected and correct:
- Larger n_paths reduces error
- Antithetic variates help reduce variance
- 1-2% error is professional-grade accuracy

### What Does Delta Tell Us?
- Delta = 0.64 means: if stock rises $1, option rises approximately $0.64
- To hedge: Hold 1 call option, short 0.64 shares
- Rebalance as delta changes (hence Gamma matters)

### What Does Gamma Tell Us?
- Gamma = 0.0188 means: Delta changes by 0.0188 per $1 stock move
- At-the-money options have highest Gamma
- Higher Gamma requires more frequent rebalancing

## Example Run

```
MONTE CARLO OPTION PRICING SIMULATOR - ANALYSIS
==================================================

OPTION PRICE COMPARISON:
          Method  Call Price  Put Price
Monte Carlo       10.574002   5.695695
Black-Scholes     10.450584   5.573526

PRICING ACCURACY:
Call Price Error: 0.123418 (1.18%)
Put Price Error: 0.122169 (2.19%)

GREEKS - HEDGING ANALYSIS:
Delta (Call): 0.6368
  → To hedge 100 call options, short 64 shares
Delta (Put): -0.3632
  → To hedge 100 put options, long 36 shares
Gamma: 0.018762
  → Rebalancing frequency: important for hedging
```

## Advanced Usage

### Increase Accuracy
Raise n_paths to 10,000+ (takes longer but more accurate):
```python
n_paths = 10000
```

### Different Option Parameters
```python
S0 = 110       # Out-of-the-money call
K = 100
sigma = 0.3    # More volatile
T = 0.5        # 6 months
```

### Analyze Different Strike Prices
Run with multiple strikes to see Delta/Gamma changes across strikes.

## References

- Hull, J. (2017). Options, Futures, and Other Derivatives
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities

## Disclaimer

This is an educational tool. Not intended for actual trading. Use for learning quantitative finance concepts.

## License

MIT License
