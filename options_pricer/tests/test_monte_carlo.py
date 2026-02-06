"""
Unit tests for Monte Carlo Option Pricing. 
"""
import numpy as np
import pytest
import sys
sys.path.append('..')

from models.monte_carlo import MonteCarlo
from models.black_scholes import BlackScholes


class TestMonteCarloPricing:
    """Test Monte Carlo pricing functionality."""
    
    def test_call_price_convergence(self):
        """
        Test that Monte Carlo converges to Black-Scholes for European calls.
        Uses many simulations and fixed seed for reproducibility.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
       
        bs = BlackScholes(S, K, T, r, sigma)
        bs_price = bs.call_price()
        
        
        mc = MonteCarlo(S, K, T, r, sigma, n_sims=500000, seed=42)
        mc_price, mc_std_error = mc.price('call')
        
        error = abs(float(mc_price) - bs_price)
        assert error < 3 * float(mc_std_error), \
            f"MC price {mc_price} too far from BS {bs_price}"
        
    def test_put_price_convergence(self):
        """Test Monte Carlo convergence for European puts."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_price = bs.put_price()
        
        mc = MonteCarlo(S, K, T, r, sigma, n_sims=500000, seed=42)
        mc_price, mc_std_error = mc.price('call')
        
        error = abs(float(mc_price) - bs_price)
        assert error < 3 * float(mc_std_error),  \
            f"MC price {mc_price} too far from BS {bs_price}" 
    
    def test_antithetic_variance_reduction(self):
        """
        Test that antithetic variates reduce standard error.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        n_sims = 100000
        
        # Regular Monte Carlo
        mc_regular = MonteCarlo(S, K, T, r, sigma, n_sims=n_sims, seed=42)
        _, mc_regular_std_error = mc_regular.price('call', antithetic=False)
        
        # Antithetic variates
        mc_anti = MonteCarlo(S, K, T, r, sigma, n_sims=n_sims, seed=42)
        _, mc_anti_std_error = mc_anti.price('call', antithetic=True)
        
        # Antithetic should have lower standard error
        assert float(mc_anti_std_error) < float(mc_regular_std_error),\
            "Antithetic variates should reduce variance"
            
    def test_increasing_simulations_reduces_error(self):
        """Standard error should decrease as √n."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        mc_1k = MonteCarlo(S, K, T, r, sigma, n_sims=10000, seed=42)
        _, result_1k_std_error = mc_1k.price('call')
        
        mc_10k = MonteCarlo(S, K, T, r, sigma, n_sims=100000, seed=42)
        _, result_10k_std_error = mc_10k.price('call')
        
        # Error should decrease by approximately √10
        ratio = float(result_1k_std_error) / float(result_10k_std_error)
        
        # Allow some tolerance due to randomness
        assert 2.5 < ratio < 4.0, \
            f"Error reduction ratio {ratio} not near √10 ≈ 3.16"
    
    def test_option_price_positive(self):
        """Option prices should always be positive."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=10000, seed=42)
        
        call_price, _ = mc.price('call')
        put_price, _ = mc.price('put')
        
        assert float(call_price) > 0, "Call price should be positive"
        assert float(put_price)> 0, "Put price should be positive"
    
    def test_confidence_interval_contains_bs_price(self):
        """
        95% CI should contain true Black-Scholes price most of the time.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_price = bs.call_price()
        
        mc = MonteCarlo(S, K, T, r, sigma, n_sims=100000, seed=42)
        _,_, ci = mc.price('call')
        
        ci_lower, ci_upper = ci
        
        assert float(ci_lower) <= bs_price <= float(ci_upper), \
            f"BS price {bs_price} outside CI [{ci_lower}, {ci_upper}]"
    
    def test_reproducibility_with_seed(self):
        """Same seed should produce same results."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        mc1 = MonteCarlo(S, K, T, r, sigma, n_sims=10000, seed=42)
        price1, _  = mc1.price('call')
        
        mc2 = MonteCarlo(S, K, T, r, sigma, n_sims=10000, seed=42)
        price2, _ = mc2.price('call')
        
        assert abs(float(price1) - float(price2)) < 1e-10, "Results should be reproducible with seed"

class TestMonteCarloGreeks:
    """Test Greeks calculations for Monte Carlo."""
    
    def test_delta_call_range(self):
        """Call delta should be between 0 and 1."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=50000, seed=42)
        delta = mc.delta('call')
        
        # Allow some tolerance for MC estimation
        assert -0.1 <= delta <= 1.1, f"Call delta {delta} far outside [0,1]"
    
    def test_delta_put_range(self):
        """Put delta should be between -1 and 0."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=50000, seed=42)
        delta = mc.delta('put')
        
        assert -1.1 <= delta <= 0.1, f"Put delta {delta} far outside [-1,0]"
    
    def test_delta_approximates_blackscholes(self):
        """MC delta should approximate Black-Scholes delta."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_delta_val = bs.delta('call')
        assert bs_delta_val is not None, "BlackScholes.delta('call') returned None"
        bs_delta = float(bs_delta_val)
        
        mc = MonteCarlo(S, K, T, r, sigma, n_sims=100000, seed=42)
        mc_delta = float(mc.delta('call', epsilon=0.01))
        
        relative_error = abs(mc_delta - bs_delta) / abs(bs_delta)
        # MC Greeks are noisier, so allow higher tolerance
        assert relative_error < 0.1, \
            f"MC delta {mc_delta} too far from BS {bs_delta}"
    
    def test_gamma_positive(self):
        """Gamma should be positive."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=50000, seed=42)
        gamma = mc.gamma('call')
        
        # Gamma can be slightly negative due to MC noise, but should be close to positive
        assert gamma > -0.001, f"Gamma {gamma} significantly negative"
    
    def test_vega_positive(self):
        """Vega should be positive."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=50000, seed=42)
        vega = mc.vega('call')
        
        assert vega > 0, f"Vega {vega} should be positive"
    
    def test_vega_approximates_blackscholes(self):
        """MC vega should approximate Black-Scholes vega."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_vega = bs.vega()
        
        mc = MonteCarlo(S, K, T, r, sigma, n_sims=100000, seed=42)
        mc_vega = mc.vega('call', epsilon=0.01)
        
        relative_error = abs(mc_vega - bs_vega) / bs_vega
        assert relative_error < 0.15, \
            f"MC vega {mc_vega} too far from BS {bs_vega}"


class TestMonteCarloPathGeneration:
    """Test path generation mechanics."""
    
    def test_paths_positive(self):
        """All simulated stock prices should be positive."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=10000, seed=42)
        paths = mc._simulate_paths(n_steps=1)
        
        assert np.all(paths > 0), "All stock prices should be positive"
    
    def test_expected_terminal_price(self):
        """
        Expected terminal stock price under risk-neutral measure: S * e^(rT)
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        mc = MonteCarlo(S, K, T, r, sigma, n_sims=500000, seed=42)
        paths = mc._simulate_paths(n_steps=1)
        
        expected_price = S * np.exp(r * T)
        simulated_mean = np.mean(paths)
        
        # Should be within 1% due to large sample
        relative_error = abs(simulated_mean - expected_price) / expected_price
        assert relative_error < 0.01, \
            f"Mean price {simulated_mean} far from expected {expected_price}"
    
    def test_multi_step_paths_shape(self):
        """Test that multi-step path generation works correctly."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=1000, seed=42)
        n_steps = 10
        paths = mc._simulate_paths(n_steps=n_steps)
        
        # Shape should be (n_simulations, n_steps + 1)
        expected_shape = (1000, n_steps + 1)
        assert paths.shape == expected_shape, \
            f"Path shape {paths.shape} != expected {expected_shape}"
    
    def test_multi_step_paths_start_at_s(self):
        """All paths should start at initial stock price."""
        mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, 
                       n_sims=1000, seed=42)
        paths = mc._simulate_paths(n_steps=10)
        
        # All paths should start at S=100
        assert np.all(paths[:, 0] == 100), "All paths should start at S"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])