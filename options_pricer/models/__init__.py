"""
Options Pricer Package

A comprehensive options pricing library implementing:
- Black-Scholes analytical pricing
- Binomial tree (CRR model)
- Monte Carlo simulation

All Greeks computed for risk management.
"""

from .black_scholes import BlackScholes
from .binomial_tree import BinomialTree
from .monte_carlo import MonteCarlo

__version__ = '1.0.0'
__all__ = ['BlackScholes', 'BinomialTree', 'MonteCarlo']
