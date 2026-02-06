"""
Unit tests for Binomial Tree option pricing
"""
import pytest
import sys
sys.path.append('..')

from models.binomial_tree import BinomialTree
from models.black_scholes import BlackScholes


class TestBinomialTreePricing:
    """Test binomial tree pricing functionality."""
    
    def test_european_call_convergence(self):
        """
        Test that binomial tree converges to Black-Scholes for European calls.
        As N increases, binomial price should approach BS price.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        # Black-Scholes price
        bs = BlackScholes(S, K, T, r, sigma)
        bs_price = bs.call_price()
        
        # Binomial tree with many steps
        bt = BinomialTree(S, K, T, r, sigma, N=500)
        bt_price = bt.price('call', american=False)
        
        # Should be within 0.5% of each other
        relative_error = abs(bt_price - bs_price) / bs_price
        assert relative_error < 0.01, \
            f"Binomial {bt_price} too far from BS {bs_price}"
            
    def test_european_put_convergence(self):
        """Test binomial tree convergence for European puts."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_price = bs.put_price()
        
        bt = BinomialTree(S, K, T, r, sigma, N=500)
        bt_price = bt.price('put', american=False)
        
        relative_error = abs(bt_price - bs_price) / bs_price
        assert relative_error < 0.01, \
            f"Binomial {bt_price} too far from BS {bs_price}"
    
    def test_american_call_no_dividend_equals_european(self):
        """
        For non-dividend paying stocks, American calls = European calls.
        (Early exercise never optimal)
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        bt = BinomialTree(S, K, T, r, sigma, N=200)
        
        european_call = bt.price('call', american=False)
        american_call = bt.price('call', american=True)
        
        assert abs(american_call - european_call) < 0.01, \
            "American and European calls should be equal (no dividends)"
    
    def test_american_put_greater_than_european(self):
        """
        American puts should be worth at least as much as European puts.
        (Early exercise can be optimal)
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        bt = BinomialTree(S, K, T, r, sigma, N=200)
        
        european_put = bt.price('put', american=False)
        american_put = bt.price('put', american=True)
        
        assert american_put >= european_put - 0.01, \
            "American put should be >= European put"
    
    def test_deep_itm_american_put_early_exercise(self):
        """
        Deep ITM American puts should show early exercise premium.
        """
        # Very deep ITM put
        S, K, T, r, sigma = 50, 100, 1.0, 0.05, 0.2
        bt = BinomialTree(S, K, T, r, sigma, N=200)
        
        european_put = bt.price('put', american=False)
        american_put = bt.price('put', american=True)
        
        # American should be noticeably higher
        assert american_put > european_put + 0.5, \
            "Deep ITM American put should have early exercise premium"
    
    def test_option_price_positive(self):
        """Option prices should always be positive."""
        bt = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=100)
        
        call_price = bt.price('call')
        put_price = bt.price('put')
        
        assert call_price > 0, "Call price should be positive"
        assert put_price > 0, "Put price should be positive"
    
    def test_increasing_steps_convergence(self):
        """Price should stabilize as number of steps increases."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bt_100 = BinomialTree(S, K, T, r, sigma, N=100)
        price_100 = bt_100.price('call')
        
        bt_200 = BinomialTree(S, K, T, r, sigma, N=200)
        price_200 = bt_200.price('call')
        
        bt_400 = BinomialTree(S, K, T, r, sigma, N=400)
        price_400 = bt_400.price('call')
        
        # Difference should decrease as N increases
        diff_1 = abs(price_200 - price_100)
        diff_2 = abs(price_400 - price_200)
        
        assert diff_2 < diff_1, "Price should converge as steps increase"


class TestBinomialTreeGreeks:
    """Test Greeks calculations for binomial tree."""
    
    def test_delta_call_range(self):
        """Call delta should be between 0 and 1."""
        bt = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=100)
        delta = bt.delta('call')
        
        assert 0 <= delta <= 1, f"Call delta {delta} out of range [0,1]"
    
    def test_delta_put_range(self):
        """Put delta should be between -1 and 0."""
        bt = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=100)
        delta = bt.delta('put')
        
        assert -1 <= delta <= 0, f"Put delta {delta} out of range [-1,0]"
    
    def test_delta_approximates_blackscholes(self):
        """Binomial delta should approximate Black-Scholes delta."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_delta = bs.delta('call')
        
        bt = BinomialTree(S, K, T, r, sigma, N=200)
        bt_delta = bt.delta('call')
        
        assert bs_delta is not None
        relative_error = abs(bt_delta - bs_delta) / abs(bs_delta)
        assert relative_error < 0.05, \
            f"Binomial delta {bt_delta} too far from BS {bs_delta}"

class TestBinomialTreeParameters:
    """Test tree parameter calculations."""
    
    def test_recombining_tree(self):
        """Test that tree recombines (u * d = 1)."""
        bt = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=100)
        
        # For CRR model: u * d should equal 1
        product = bt.u * bt.d
        assert abs(product - 1.0) < 1e-10, f"Tree doesn't recombine: u*d = {product}"
    
    def test_risk_neutral_probability_range(self):
        """Risk-neutral probability should be between 0 and 1."""
        bt = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=100)
        
        assert 0 < bt.p < 1, f"Risk-neutral probability {bt.p} out of range"
    
    def test_up_factor_greater_than_one(self):
        """Up factor should be > 1."""
        bt = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=100)
        
        assert bt.u > 1, f"Up factor {bt.u} should be > 1"
    
    def test_down_factor_less_than_one(self):
        """Down factor should be < 1."""
        bt = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=100)
        
        assert bt.d < 1, f"Down factor {bt.d} should be < 1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])