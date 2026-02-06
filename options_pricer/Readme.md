# Options Pricer

A comprehensive Python library for pricing options and computing greeks, built entirely from scratch without the use of quantitative finance libraries. 
This implements three major pricing models: Black-Scholes, Binomial Tree, and Monte Carlo simulation. 

## Features

- **Three Pricing Models:**
  - Black-Scholes pricing
  - Binomial Tree
  - Monte Carlo simulation with variance reduction

- **Complete Greeks calculations:**
  - Delta: Price sensitivity to stock price
  - Gamma: Rate of change of Delta
  - Vega: Sensitivity to volatility
  - Theta: Time decay
  - Rho: Sensitivity to change in risk-free interest rates

- **Unit testing**
  - Greeks validation
  - Model convergence tests
  - Put-Call parity verification
 
- **Interactive Visualisation**
  - Greeks vs Strike price
  - Greeks vs Time to maturity
  - Volatility surface
  - Model convergence


## Future additions

1. Add implied volatility solver
2. Implement dividend handling
3. create a web dashboard
4. add more exotic options
