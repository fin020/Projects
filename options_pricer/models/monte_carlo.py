"""
Monte Carlo simulatioin for option pricing. 

Monte Carlo method:
Simulates possible future price paths
Caluclates option payoff for each path
Averages payoffs and discounts to present value

This handles complex path-dependent options
Easily extends to multiple assets
It is computationally intensive
Cannot price American options directly (needs regression)
"""
import numpy as np
from .utils import validate_inputs

class MonteCarlo:
    """
    Monte Carlo option pricing using geometric Brownian motion.
    """
    def __init__(self, S: float, K: float, T: float,
                r: float, sigma: float,n_sims: int=10000, seed: int|None=None):
        """
        initialises the Monte Carlo pricer
        
        
        :param S: Current stock price
        :type S: float
        :param K: Strike price
        :type K: float
        :param T: Time to maturity (years)
        :type T: float
        :param r: risk-free rate of interest (annual)
        :type r: float
        :param sigma: Volatility (annual)
        :type sigma: float
        :param n_sims: Number of Monte Carlo simulations
        :type n_sims: float
        :param seed: Random seed for reproducability
        :type seed: int | None
        """
        self.s = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_sims = n_sims
        
        if seed is not None:
            np.random.seed(seed)
            
    def _simulate_paths(self, n_steps: int=1):
        """
        Simulates stock price paths using geometric Brownian motion (GBM).
        
        Stock price follows: dS = mu*S*dt + sigma*S*dW
        Solution: S(T) = S(0) * exp((mu - 0.5 * sigma^2)*T + sigma* sqrt(T) *Z)
        
        Where Z ~ N(0,1)
        
        Under Risk-neutral measure, mu = r. 
        
        :param n_steps: Number of time steps per path
        :type n_steps: int
        
        Returns:
        np.NDArray : simulated terminal stock prices (shape: n_sims)
        """
        dt = self.T / n_steps
        
        if n_steps == 1:
            Z = np.random.standard_normal(self.n_sims)
            
            ST = (self.s * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T +
                    (self.sigma * np.sqrt(self.T) * Z)))
            
            return ST
        
        else: 
            paths = np.zeros((self.n_sims, n_steps + 1))
            paths[:,0] = self.s
            
            for i in range(1, n_steps + 1):
                z = np.random.standard_normal(self.n_sims)
                paths[:,i] = paths[:,i-1] * np.exp(
                    (self.r - 0.5 * self.sigma ** 2) * dt +
                    self.sigma * np.sqrt(dt) * z
                )
            
            return paths
    
    def price(self, option_type: str='call', 
            antithetic:bool=False) -> dict[str, float | tuple[float,float]]:
        """
        Prices european option using monte carlo simulation.
        
        Simulates terminal stock prices
        Calculates payoffs for each simulation
        Averages payoffs
        Discounts to present value
        """
        validate_inputs(self.s, self.K, self.T, self.r, self.sigma, option_type)
        
        if antithetic:
            n_pairs = self.n_sims // 2
            Z = np.random.standard_normal(n_pairs)
            Z_anti = np.concatenate([Z,-Z])
            
            st = (self.s * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T +
                    (self.sigma * np.sqrt(self.T) * Z_anti))
                )
        else: 
            st = self._simulate_paths(n_steps=1)
            
        if option_type == 'call':
            payoffs = np.maximum(st - self.K, 0)
        else:
            payoffs = np.maximum(self.K - st, 0)
            
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        standard_error = np.std(payoffs) / np.sqrt(self.n_sims) * \
                        np.exp(-self.r * self.T)
                        
        return {
            'price': option_price,
            'standard error': standard_error,
            '95%_CI': (option_price - 1.96 * standard_error,
                       option_price + 1.96 * standard_error),
        }
        
    def delta(self, option_type:str = 'call', epsilon:float=0.01) -> float:
            """
            Calculates Delta using finite difference.
            
            Delta = (V(S + epsilon) - V(S-epsilon)) / 2*epsilon
            
            Uses same random numbers for both calculations to reduce variance
            
            Parameters: option_type: str 
                'call' or 'put' 
            epsilon : float 
                Bump size for finite difference 
                
            Return:
            
            Delta: float
            """
            s_original = self.s
            
            bump = epsilon * self.s
            
            seed = np.random.randint(0,1000000)
            
            self.s = s_original + bump
            np.random.seed(seed)
            price_up_dict = self.price(option_type, antithetic=False)
            price_up = price_up_dict['price']
            
            self.s = s_original - bump
            np.random.seed(seed)
            price_down_dict = self.price(option_type, antithetic=False)
            price_down = price_down_dict['price']
            
            self.s = s_original
            
            if not isinstance(price_up, float) or not isinstance(price_down, float):
                raise ValueError("Price calculation returned unexpected type.")
            
            delta = (price_up - price_down) / (2 * bump)
            
            return delta
        
    def gamma(self, option_type:str='call', epsilon:float = 0.01) -> float:
        """
        Calculates Gamma via finite difference.
        
        Gamma = (V(S + epsilon) - 2*V(S) + V(S - epsilon)) / epsilon^2
        
        Parameters: option_type: str 
            'call' or 'put' 
        epsilon : float 
            Bump size for finite difference 
            
        Return:
        
        gamma: float
        """
        s_original = self.s
        bump = self.s * epsilon
        
        seed = np.random.randint(0,1000000)
        
        np.random.seed(seed)
        price_mid_dict = self.price(option_type, antithetic=False)
        price_mid = price_mid_dict['price']
        
        self.s = s_original + bump
        np.random.seed(seed)
        price_up_dict = self.price(option_type, antithetic=False)
        price_up = price_up_dict['price']
        
        np.random.seed(seed)
        self.s = s_original - bump
        price_down_dict = self.price(option_type, antithetic=False)
        price_down = price_down_dict['price']
        
        self.s = s_original
        
        if not isinstance(price_up, float) or not isinstance(price_down, float) or not isinstance(price_mid, float):
            raise ValueError("Price calculation returned unexpected type.")
        
        gamma = (price_up - 2* price_mid + price_down) / (bump**2)
        
        return gamma
        
    def vega(self, option_type:str='call', epsilon:float=0.01):
        """
        Calculating Vega using finite difference.
        
        Vega = (V(sigma + epsilon) - V(sigma - epsilon)) / 2*epsilon
        
        Parameters:
        option_type: str
            'call' or 'put'
        epsilon : float
            bump size for finite difference as absolute change
        Return:
        Vega : float
        """
        sigma_original = self.sigma
        
        seed = np.random.randint(0,1000000)
        
        self.sigma = sigma_original + epsilon
        np.random.seed(seed)
        price_up_dict = self.price(option_type, antithetic=False)
        price_up = price_up_dict['price']
        
        self.sigma = sigma_original - epsilon
        np.random.seed(seed)
        price_down_dict = self.price(option_type, antithetic=False)
        price_down = price_down_dict['price']
        
        if not isinstance(price_up, float) or not isinstance(price_down,float):
            raise ValueError("price calculation returned unexpected type")
        
        vega = (price_up - price_down) / (2 * epsilon)
        
        return vega
            
            

        
        
        
            