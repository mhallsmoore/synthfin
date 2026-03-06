import numpy as np
import pytest

from synthfin.models import (
    GeometricBrownianMotion,
    JumpDiffusion,
    AR1GBM,
    GARCH,
    ARMAGARCH,
    HestonStochasticVolatility,
    OrnsteinUhlenbeck,
    TimeSeriesModel,
)


N_STEPS = 252
START_PRICE = 100.0


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _shocks_1d(rng, n=N_STEPS):
    return rng.standard_normal(n)


def _shocks_2d(rng, n=N_STEPS, cols=2):
    return rng.standard_normal((n, cols))


# ---------------------------------------------------------------------------
# Base class attribute
# ---------------------------------------------------------------------------

class TestBaseClass:
    def test_n_shock_streams_default(self):
        assert GeometricBrownianMotion.n_shock_streams == 1

    def test_heston_n_shock_streams(self):
        assert HestonStochasticVolatility.n_shock_streams == 2


# ---------------------------------------------------------------------------
# AR1-GBM
# ---------------------------------------------------------------------------

class TestAR1GBM:
    def test_output_shape(self, rng):
        model = AR1GBM(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert prices.shape == (N_STEPS,)

    def test_positive_prices(self, rng):
        model = AR1GBM(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert np.all(prices > 0)

    def test_ar_coeff_validation(self):
        with pytest.raises(ValueError, match="ar_coeff"):
            AR1GBM(start_price=START_PRICE, ar_coeff=1.0)
        with pytest.raises(ValueError, match="ar_coeff"):
            AR1GBM(start_price=START_PRICE, ar_coeff=-1.0)

    def test_returns_show_autocorrelation(self):
        """With high AR coefficient, returns should be autocorrelated."""
        np.random.seed(123)
        model = AR1GBM(start_price=START_PRICE, drift=0.0, volatility=0.2, ar_coeff=0.5)
        shocks = np.random.standard_normal(5000)
        prices = model.generate_path(5000, shocks)
        log_returns = np.diff(np.log(np.concatenate([[START_PRICE], prices])))
        # Lag-1 autocorrelation
        autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        assert autocorr > 0.1, f"Expected positive autocorrelation, got {autocorr}"


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

class TestGARCH:
    def test_output_shape(self, rng):
        model = GARCH(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert prices.shape == (N_STEPS,)

    def test_positive_prices(self, rng):
        model = GARCH(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert np.all(prices > 0)

    def test_alpha_beta_validation(self):
        with pytest.raises(ValueError, match="alpha \\+ beta"):
            GARCH(start_price=START_PRICE, alpha=0.5, beta=0.5)

    def test_omega_auto_computed(self):
        model = GARCH(start_price=START_PRICE, volatility=0.2, alpha=0.05, beta=0.90)
        expected_omega = 0.2**2 * (1/252) * (1 - 0.05 - 0.90)
        assert abs(model.omega - expected_omega) < 1e-12

    def test_squared_returns_autocorrelation(self):
        """Squared returns should show autocorrelation (volatility clustering)."""
        np.random.seed(456)
        model = GARCH(start_price=START_PRICE, drift=0.0, volatility=0.3,
                      alpha=0.10, beta=0.85)
        shocks = np.random.standard_normal(5000)
        prices = model.generate_path(5000, shocks)
        log_returns = np.diff(np.log(np.concatenate([[START_PRICE], prices])))
        sq_returns = log_returns**2
        autocorr = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]
        assert autocorr > 0.05, f"Expected positive autocorrelation in squared returns, got {autocorr}"


# ---------------------------------------------------------------------------
# ARMA-GARCH
# ---------------------------------------------------------------------------

class TestARMAGARCH:
    def test_output_shape(self, rng):
        model = ARMAGARCH(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert prices.shape == (N_STEPS,)

    def test_positive_prices(self, rng):
        model = ARMAGARCH(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert np.all(prices > 0)

    def test_ar_coeff_validation(self):
        with pytest.raises(ValueError, match="ar_coeff"):
            ARMAGARCH(start_price=START_PRICE, ar_coeff=1.5)

    def test_alpha_beta_validation(self):
        with pytest.raises(ValueError, match="alpha \\+ beta"):
            ARMAGARCH(start_price=START_PRICE, alpha=0.6, beta=0.5)


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------

class TestHeston:
    def test_output_shape(self, rng):
        model = HestonStochasticVolatility(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_2d(rng))
        assert prices.shape == (N_STEPS,)

    def test_positive_prices(self, rng):
        model = HestonStochasticVolatility(start_price=START_PRICE, drift=0.05, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_2d(rng))
        assert np.all(prices > 0)

    def test_n_shock_streams(self):
        model = HestonStochasticVolatility(start_price=START_PRICE)
        assert model.n_shock_streams == 2

    def test_theta_v_defaults_to_vol_squared(self):
        model = HestonStochasticVolatility(start_price=START_PRICE, volatility=0.3)
        assert abs(model.theta_v - 0.09) < 1e-10


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck
# ---------------------------------------------------------------------------

class TestOrnsteinUhlenbeck:
    def test_output_shape(self, rng):
        model = OrnsteinUhlenbeck(start_price=START_PRICE, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert prices.shape == (N_STEPS,)

    def test_positive_prices(self, rng):
        model = OrnsteinUhlenbeck(start_price=START_PRICE, volatility=0.2)
        prices = model.generate_path(N_STEPS, _shocks_1d(rng))
        assert np.all(prices > 0)

    def test_theta_defaults_to_log_start_price(self):
        model = OrnsteinUhlenbeck(start_price=START_PRICE)
        assert abs(model.theta - np.log(START_PRICE)) < 1e-10

    def test_mean_reversion(self):
        """Prices should revert toward exp(theta) over long horizons."""
        np.random.seed(789)
        theta = np.log(50.0)
        model = OrnsteinUhlenbeck(start_price=200.0, volatility=0.1, kappa=10.0, theta=theta)
        shocks = np.random.standard_normal(2000)
        prices = model.generate_path(2000, shocks)
        # Last 500 prices should be close to exp(theta) = 50
        mean_last = np.mean(prices[-500:])
        assert abs(mean_last - 50.0) < 15.0, f"Expected mean near 50, got {mean_last}"
