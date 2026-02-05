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
        """
        Calculates d1 and d2 parameters for Black-Scholes formula
        
        d1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma* sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        These represent moneyness and time decay
        """
        
        self.d1 = (np.log(self.S / self.K) +
                    (self.r + 0.5 * self.sigma **2) * self.T) / \
            (self.sigma * np.sqrt(self.T))
            
        self.d2 = self.d1 - (self.sigma * np.sqrt(self.T))
        
    def call_price(self) -> float:
        """
        Calculates European call option price. 
        
        Formula: C = S*N(d1) - K*exp(-rT)*N(d2)
        
        Where:
        N(x) is the cumulative standard normal distribution
        First term: present value of the expected stock price at expiry if in-the-money.
        Second term: present value of strike price weighted by probability of exercise
        
        Returns:
        float : Call option price
        """
        validate_inputs(self.S, self.K, self.T, self.r, self.sigma, 'call')
        
        Price: float = self.S * standard_normal_cdf(self.d1) - \
                self.K * np.exp(-self.r * self.T) * standard_normal_cdf(self.d2)
        
        return Price
            
    def put_price(self) -> float:
        """
        Calculate European put optioin price.
        
        Formula: P = K*exp(-rT)*N(-d2) - S*N(-d1)
        
        Can also be derived from put-call parity:
        P = C - S + K*exp(-rt)
        
        Returns: 
        float: put option price
        """
        validate_inputs(self.S, self.K, self.T, self.r, self.T, 'put')
        
        Price = self.K * np.exp(-self.r * self.T) * standard_normal_cdf(self.d2) - \
                self.S * standard_normal_cdf(self.d1)
                
        return Price
    
    def delta(self, option_type: str='call'):
        """
        Calculate Delta: dV/dS (rate of change of option price w.r.t stock price)
        
        Call Delta: N(d1)
        Put Deta: N(d1) - 1 = -N(-d1)
        
        Delta ranges:
        Call: [0,1]
        Put: [-1,0]
        
        For a £1 increase in stock price, option price changes by Delta
        
        Params:
        option_type: str 
            'call' or 'put'
            
        Returns:
        float: Delta value
        """ 
        validate_inputs(self.S, self.K, self.T, self.r, self.T, 'put')
        
        if option_type == 'call':
            return standard_normal_cdf(self.d1)
        if option_type == 'put':
            return standard_normal_cdf(self.d1) -1
        
    def gamma(self) -> float:
        """
        Calculates gamma: d^2V/dS^2 (rate of change of Delta w.r.t. stock price)
        
        Formula: Gamma = N'(x) / (S*sigma*sqrt(T))
        
        Where N'(x) is the standard normal PDF.
        
        Gamma is the same for calls and puts.
        
        It measures the convexity of option prices. 
        Higher gamma = Delta changes more rapidly
        Maximum near at-the-money
        
        :param self: Description
        :return: Description
        :rtype: float
        
        Returns:
        float : Gamma value
        """ 
        validate_inputs(self.S, self.K, self.T, self.r, self.T, 'put')
        
        return (standard_normal_pdf(self.d1) /
            (self.S * self.sigma * np.sqrt(self.T)))
        
    def vega(self):
        """
        Calculate Vega: dV/dsigma (rate of change of option price w.r.t. volatility)
        
        Formula: v = S*N'(d1)*sqrt(T)
        
        Vega is the same for calls and puts. 
        
        It is typically expressed per 1% change in volatility, so /100 when interpreting.
        
        For a 1% increase in volatility, option price changes by Vega/100
        Always positive (higher volatility = higher option value)
        Maximum near at-the-money
        
        Returns:
        float: Vega value
        """    
        validate_inputs(self.S, self.K, self.T, self.r, self.T, 'put')
        
        return self.S * standard_normal_pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self, option_type: str='call') -> float:
        """
        Calculate Theta: dV/dT (rate of change of option price w.r.t. time)
        
        This is negative as time to maturity decreases. 
        
        Call theta: -(S*N'(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)
        Put theta: -(S*N'(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)
        
        Represents time decay
        Expressed as change per day: divide by 365
        
        Parameters:
        option_type: str
            'call' or 'put'
        """
        validate_inputs(self.S, self.K, self.T, self.r, self.T, 'put')
        
        term1 = -(self.S * standard_normal_pdf(self.d1)*self.sigma) / \
                (2 * np.sqrt(self.T))
                
        term2 = (self.r * self.K * np.exp(-self.r * self.T) * standard_normal_cdf(self.d2)) 
        
        if option_type == 'call':
            return term1 + term2
        if option_type == 'put':
            return term1 - term2
        else: 
            raise ValueError(f"Invalid option_type '{option_type}'. Expected 'call' or 'put'")
        
    def rho(self, option_type: str='call') -> float:
        """
        Calculate Rho: dV/dr (rate of change of option price w.r.t. interest rate)
    
        Call Rho: K*T*exp(-rT)*N(d2)
        Put Rho: -K*T*exp(-rT)*N(-d2)
        
        For a 1% increase in interest rate, option price change by Rho/100
        Call rho positive, put rho negative
        
        Parameters:
        option_type: str
            'call' or 'put'
        
        Returns:
        Rho value: float
        """  
        validate_inputs(self.S, self.K, self.T, self.r, self.T, 'put')
        
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * \
                standard_normal_cdf(self.d2)
        if option_type == 'put':
            return - self.K * self.T * np.exp(-self.r * self.T) * \
                standard_normal_cdf(-self.d2)
        else:
            raise ValueError(f"Invalid option_type '{option_type}'. Expected 'call' or 'put'")
        
        
        