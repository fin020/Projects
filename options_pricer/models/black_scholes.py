"""
Black Scholes Option Pricing Model

Assumes:
1. Lognormal distribution of stock prices
2. No dividends
3. Constant volatility and interest rates
4. European-style options (exercised only at maturity)
5. No transaction costs
"""

import numpy as np 
from .utils import validate_inputs, standard_normal_cdf, standard_normal_pdf

class BlackScholes:
    """
    Black-Scholes option pricing and greeks calculation.
    """
    def __init__(self, S:float, K:float, T: float, r:float, sigma:float):
        """
        Initialises Black-Scholes pricer.
        
        :param S: Current Stock Price
        :type S: float
        :param K: Strike price
        :type K: float
        :param T: Time till maturity (years)
        :type T: float
        :param r: Risk-free rate of interest (annual)
        :type r: float
        :param sigma: Volatility (annual)
        :type sigma: float
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        self._calculate_d1_d2()
    
    def _calculate_d1_d2(self):
        "Calculates d1 and d2 parameters for Black-Scholes formula"