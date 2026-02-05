"""
Binomial Tree option pricing model.

The binomial model discretises time and stock price movements:
At each step, stock can move up by a factor u or down by d
Uses risk-neutral pricing: expecred return equals risk-free rate
can price American options (early exercise allowed)
"""

import numpy as np
import numpy.typing as npt
from .utils import validate_inputs

class BinomialTree:
    """
    Cox-Ross_rubinstein (CRR) binomial tree for option pricing.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float,
                sigma: float, N: int=100):
        """
        Initialises binomial tree pricer. 
        
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
        self.N = N
        validate_inputs(self.S, self.K, self.T,
                        self.r, self.sigma, 'call')
        
        self.dt = T / N
        
        # Calculate tree paramters using CRR formulas
        # u: up-move factor, d: down-move factor, p: risk-neutral probability
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u # Down factor (ensures recombining tree)
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d) #risk-neutral probability
        
        # Discount factor per step
        self.discount = np.exp(-r * self.dt)
        
    def _build_stock_tree(self) -> npt.NDArray[np.float64]:
        """
        Build the stock price tree.
        
        Tree structure:
        tree[i][j] = stock price at time step i, after j up-moves
        -i ranges from 0 to N (time steps)
        -j ranges from 0 to i (number of up-moves)
        
        Formula: S * u^j * d^(i-j)
        
        Returns:
        np.ndarray : Stock price tree (shape: (N+1, N+1))
        """
        tree = np.zeros((self.N + 1, self.N + 1))
        
        for i in range(self.N + 1):
            for j in range(self.N + 1):
                tree[i][j] = self.S * (self.u ** j) * (self.d ** (i-j))
        
        return tree
    
    def _option_payoff(self, stock_price: float, option_type: str):
        """
        Calculate option payoff at maturity.
        
        Parameters: 
        stock_price : float
            Stock price at maturity
        option_type : str
            'call' or 'put'
            
        Returns: 
        float : option payoff
        """
        if option_type == 'call':
            return max(0, stock_price - self.K)
        else:
            return max(self.K - stock_price, 0)
        
    def price(self, option_type:str ='call', american: bool=False) -> np.float64:
        """
        Price option using backward induction through the tree.
        
        Algorithm: 
        Build stock price tree
        Calculate payoffs at maturity
        Work backward through tree
            European: option_value = discounted expected value
            American: option_value = max(exercises_value, hold_value)
            
        Parameters:
        option_type : str
            'call' or 'put' 
        american : bool
            True for American option, False for European
            
        Returns: 
        Option price : float
        """
        validate_inputs(self.S, self.K, self.T, self.r, self.sigma, option_type)
        
        stock_tree: npt.NDArray[np.float64] = self._build_stock_tree()
        
        option_tree = np.zeros((self.N + 1, self.N + 1))
        
        for j in range(self.N + 1):
            option_tree[self.N, j] = self._option_payoff(
                stock_tree[self.N, j], option_type
            )
            
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                # Expected option value if holding
                hold_value = self.discount * (
                    self.p * option_tree[i + 1, j + 1] +
                    (1 - self.p) * option_tree[i + 1, j]
                )
                
                if american:
                    exercise_value: float = self._option_payoff(
                        stock_tree[i, j], option_type)
                    option_tree[i][j] = max(hold_value, exercise_value)
                else: 
                    option_tree[i, j] = hold_value
        
        return np.float64(option_tree[0, 0])
                
    def delta(self, option_type:str='call', american:bool=False) -> float:
        """
        Calculates delta using finite difference. 
        
        Delta = (V(S + dS) - V(S - dS)) / (2 * dS)
        
        We use the first step of the tree for the calculation
        
        Parameters: 
        
        option type : str
            'call' or 'put' 
        american : bool
            True for American option
        
        Returns: Delta estimate
        """
        stock_tree = self._build_stock_tree()
        
        option_tree = np.zeros((self.N + 1, self.N + 1))
        
        for j in range(self.N + 1):
            option_tree[self.N, j] = self._option_payoff(
                stock_tree[self.N, j], option_type
            )
        
        for i in range(self.N - 1, 0, -1):
            for j in range(i+1):
                hold_value = self.discount * (
                    self.p * option_tree[i+1, j+1] + 
                    (1-self.p) * option_tree[i+1, j]
                )
                if american:
                    exercise_value = self._option_payoff(stock_tree[i,j], option_type)
                    option_tree[i,j] = max(hold_value, exercise_value)
                else:
                    option_tree[i,j] = hold_value
        
        delta = (option_tree[1,1] - option_tree[1,0]) / \
                (stock_tree[1,1] - stock_tree[1,0])
                
        return delta
    
    def gamma(self, option_type:str='call', american:bool=False):
        """
        Calculates Gamma using finite difference. 
        
        Gamma = ((V_uu - v_ud) /(S_uu - S_ud) - (V_ud - V_dd)/(S_ud - S_dd)) /
                ((S_uu - S_dd)/2)
        
        Parameters 
         
        option type : str
            'call' or 'put' 
        american : bool
            True for American option
        
        Returns: Gamma estimate
        """
        if self.N < 2:
            raise ValueError("Need at least 2 time steps to calculate gamma")
        
        stock_tree = self._build_stock_tree()
        
        option_tree = np.zeros((self.N + 1, self.N + 1))
        
        for j in range(self.N + 1):
            option_tree[self.N, j] = self._option_payoff(
                stock_tree[self.N, j], option_type
            )
            
        for i in range(self.N -1, 1, -1):
            for j in range(i+1):
                hold_value = self.discount * (
                    self.p * option_tree[i+1, j+1] + 
                    (1 - self.p) * option_tree[i+1, j]
                )
                if american:
                    exercise_value = self._option_payoff(stock_tree[i,j], option_type)
                    option_tree[i,j] = max(hold_value, exercise_value)
                else: 
                    option_tree[i,j] = hold_value
                
        delta_up = (option_tree[2,2] - option_tree[2,1]) / \
            (stock_tree[2,2] - stock_tree[2,1])
        
        delta_down = (option_tree[2,1] - option_tree[2,0]) / \
            ((stock_tree[2,1] - stock_tree[2,0]) / 2)
            
        gamma = (delta_up - delta_down) / \
            ((stock_tree[2,2] - stock_tree[2,0]) / 2)
            
        return gamma
    
    