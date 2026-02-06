"""
Unit tests for Black-Scholes option pricing. 

Tests include:
Put call parity
Greeks calculations
Pricing accuracy against known values
Boundary conditions
"""

import pytest 
import numpy as np 
import sys
sys.path.append("..")

from models.black_scholes import BlackScholes

class TestBlackScholesPricing:
    """
    Test Black-Scholes pricing functionality.
    """
    def test_call_price_basic(self):
        """test call optioin pricing with a known value"""
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        call_price = bs.call_price()
        
        assert 10.4 < call_price < 10.5, f"Call price {call_price} outside of expected range"
        
    def test_put_price_basic(self):
        """tests put option pricing with a known value"""
        
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        put_price = bs.put_price()
        
        assert 5.5 < put_price <5.7, f"Put price {put_price} outside expected range"
        
    def test_deep_in_the_money_call(self):
        """Deep ITM calls should be worth around S - K*e^(-rT)"""
        
        bs=BlackScholes(S=150,K=100,T=1,r=0.05,sigma=0.2)
        call_price = bs.call_price()
        intrinsic_value = 150 - 100 * np.exp(-0.05 * 1.0)
        assert call_price > intrinsic_value, "Call price is below intrinsic value"
        
    def test_deep_out_of_the_money_call(self):
        """Deep OTM calls should be worth close to 0"""

        bs=BlackScholes(S=50,K=100,T=1,r=0.05,sigma=0.2)
        call_price = bs.call_price()
        
        assert call_price < 0.1, f"Deel OTM call price {call_price} too high"
        
    def test_zero_time_to_maturity(self):
        """At maturity, option = max(payoff, 0)."""
        T = 1e-10
        bs=BlackScholes(S=110,K=100,T=T,r=0.05,sigma=0.2)
        
        call_price = bs.call_price()
        expected_payoff = max(110-100,0)
        
        assert abs(call_price - expected_payoff) < 0.01, "Maturity payoff incorrect"
        
    def test_put_call_parity(self):
        """
        Test put-call parity : C - P = S - K*e^(-rt)
        
        This is a fundamnetal relationship that must hold for euro options.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        bs = BlackScholes(S, K, T, r, sigma)
        
        call = bs.call_price()
        put = bs.put_price()
        
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 1e-10, "Put-call parity violated"
        

class TestBlackScholesGreeks:
    """Test Greeks calculations"""
    
    def test_call_delta_range(self):
        """Call delta should be between 0 and 1"""
        
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        delta = bs.delta("call")
        
        assert delta is not None, "Delta should not be None"
        assert 0<= delta <= 1, f"Call delta {delta} out of range [0,1]"
        
    def test_delta_relationship(self):
        """Call delta = Put delta - 1""" 
     
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        call_delta = bs.delta('call')
        put_delta = bs.delta('put')
        
        
        assert call_delta is not None and put_delta is not None, "Delta should not be None"
        assert abs(call_delta - put_delta - 1) < 1e-10, "Delta relationship violated"
        
    def test_gamma_positive(self):
        """Gamma must be a positive number"""
        
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        gamma = bs.gamma()
        
        assert gamma > 0, f"Gamma {gamma} should always be positive"
        
    def test_gamma_symmetry(self): 
        """Gamma is the same for calls and puts"""
        
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        gamma = bs.gamma()
        
        epsilon = 0.01
        bs_up = BlackScholes(S=100 + epsilon, K=100, T=1.0, r=0.05, sigma=0.2)
        bs_down = BlackScholes(S=100 - epsilon, K=100, T=1.0, r=0.05, sigma=0.2)
        
        delta_up = bs_up.delta('call')
        delta_down = bs_down.delta('call')
        assert delta_up is not None and delta_down is not None, "Delta should not be None"
        gamma_approx = (delta_up - delta_down) / (2 * epsilon)
        
        assert abs(gamma - gamma_approx) / gamma < 0.01, "Gamma calculation inconsistent"
        
        
    def test_vega_positive(self):
        """Vega must be a positive number"""
        
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        vega = bs.vega()
        
        assert  vega > 0, f"Vega {vega} must be a positive number"
        
    def test_theta_call_usually_negative(self):
        """Call theta should be negative for non dividend paying stocks"""
        
        bs=BlackScholes(S=100,K=100,T=1,r=0.05,sigma=0.2)
        theta = bs.theta('call')
        
        assert theta < 0, f"ATM call theta {theta} should be negative"
        
    def test_vega_atm_maximum(self):
        """Vega should be maximum near ATM."""
        bs_atm = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bs_itm = BlackScholes(S=120, K=100, T=1.0, r=0.05, sigma=0.2)
        bs_otm = BlackScholes(S=80, K=100, T=1.0, r=0.05, sigma=0.2)
        
        vega_atm = bs_atm.vega()
        vega_itm = bs_itm.vega()
        vega_otm = bs_otm.vega()
        
        assert vega_atm > vega_itm, "ATM vega should exceed ITM vega"
        assert vega_atm > vega_otm, "ATM vega should exceed OTM vega"
         

class TestInputValidation:
    """Test input validation."""
    
    def test_negative_stock_price(self):
        """Should raise error for negative stock price."""
        with pytest.raises(ValueError):
            bs = BlackScholes(S=-100, K=100, T=1.0, r=0.05, sigma=0.2)
            bs.call_price()
    
    def test_negative_strike(self):
        """Should raise error for negative strike."""
        with pytest.raises(ValueError):
            bs = BlackScholes(S=100, K=-100, T=1.0, r=0.05, sigma=0.2)
            bs.call_price()
    
    def test_negative_time(self):
        """Should raise error for negative time to maturity."""
        with pytest.raises(ValueError):
            bs = BlackScholes(S=100, K=100, T=-1.0, r=0.05, sigma=0.2)
            bs.call_price()
    
    def test_negative_volatility(self):
        """Should raise error for negative volatility."""
        with pytest.raises(ValueError):
            bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=-0.2)
            bs.call_price()
    
    def test_invalid_option_type(self):
        """Should raise error for invalid option type."""
        with pytest.raises(ValueError):
            bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
            bs.delta('invalid')
            
if __name__ == '__main__':
    pytest.main([__file__, '-v'])