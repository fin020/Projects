import numpy as np 
import numpy.typing as npt
from scipy.stats import norm

def validate_inputs(S:float, K:float, T:float, r:float, sigma:float, option_type:str='call'):
    """
    Validate option pricing inputs.
    
    :param S: float
        Current stock price (positive)
    :param K: float
        Strike price (positive)
    :param T: float
        Time to maturinty in years (positive)
    :param r: float
        Risk free rate of interest (annual, decimal)
    :param sigma: float
        Volatility (annual, decimal)
    :param option_type: str
        'call' or 'put'
        
    Raises: ValueError if inputs are invalid
    """
    if S <= 0:
        raise ValueError(f"Stock price must be positive, you entered: {S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive, you entered {K}")
    if T <= 0:
        raise ValueError(f"Time till maturity must be positive, you entered {T}")
    if r < 0 or r > 1:
        raise ValueError(f"Risk free rate of interest must be between 0 and 1, you entered {r}")
    if sigma <= 0:
        raise ValueError(f"Volatility entered must be positive and annual, you entered {sigma}")
    if option_type not in ['call', 'put']:
        raise ValueError(f"Option type must be 'call' or 'put', you entered: {option_type}")
    

def standard_normal_cdf(x:float|npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """
    Cumulative distribution function for standard normal distribution. 
    Uses scipy.stats.norm. 
    
    :param x: float or np.ndarray
        Value(s) at which to evaluae CDF
        
    Returns: 
    
    float or np.ndarry
        CDF value(s)
    """
    return norm.cdf(x)
    

def standard_normal_pdf(x:float|npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """
    Probability density function for standard normal distribution. 
    Formula: (1/sqrt(2*pi) *exp(-x^2/2))
    
    :param x: float or np.ndarray
        Value(s) at which to evaluae CDF
        
    Returns: 
    
    float or np.ndarry
        CDF value(s)
    """
    return norm.pdf(x)


    

   