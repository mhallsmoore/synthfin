import numpy as np
import pytest

from synthfin.correlation import HierarchicalCorrelationGenerator
from synthfin.generator import CorrelatedTimeSeriesGenerator
from synthfin.models import (
    GeometricBrownianMotion,
    GARCH,
    HestonStochasticVolatility,
    OrnsteinUhlenbeck,
    AR1GBM,
)


class TestMultiStreamGenerator:
    """Integration tests for multi-stream shock support in the generator."""

    def test_single_stream_models(self):
        """Standard single-stream models work as before."""
        np.random.seed(100)
        n_assets = 5
        n_days = 252
        corr_gen = HierarchicalCorrelationGenerator(n_assets)
        models = [GeometricBrownianMotion(start_price=100.0, drift=0.05, volatility=0.2)
                  for _ in range(n_assets)]
        gen = CorrelatedTimeSeriesGenerator(corr_gen, models)
        prices, corr_matrix = gen.generate(n_days)
        assert prices.shape == (n_days, n_assets)
        assert np.all(prices > 0)

    def test_heston_models(self):
        """All-Heston portfolio generates correctly."""
        np.random.seed(200)
        n_assets = 5
        n_days = 252
        corr_gen = HierarchicalCorrelationGenerator(n_assets)
        models = [HestonStochasticVolatility(start_price=100.0, drift=0.05, volatility=0.2)
                  for _ in range(n_assets)]
        gen = CorrelatedTimeSeriesGenerator(corr_gen, models)
        prices, corr_matrix = gen.generate(n_days)
        assert prices.shape == (n_days, n_assets)
        assert np.all(prices > 0)

    def test_mixed_models(self):
        """Mix of single-stream and multi-stream models."""
        np.random.seed(300)
        n_assets = 5
        n_days = 252
        corr_gen = HierarchicalCorrelationGenerator(n_assets)
        models = [
            GeometricBrownianMotion(start_price=100.0, drift=0.05, volatility=0.2),
            GARCH(start_price=150.0, drift=0.05, volatility=0.25),
            HestonStochasticVolatility(start_price=200.0, drift=0.05, volatility=0.3),
            AR1GBM(start_price=80.0, drift=0.05, volatility=0.2, ar_coeff=0.1),
            OrnsteinUhlenbeck(start_price=50.0, volatility=0.15, kappa=5.0),
        ]
        gen = CorrelatedTimeSeriesGenerator(corr_gen, models)
        prices, corr_matrix = gen.generate(n_days)
        assert prices.shape == (n_days, n_assets)
        assert np.all(prices > 0)

    def test_correlation_structure_maintained(self):
        """Cross-asset correlation structure should be maintained."""
        np.random.seed(400)
        n_assets = 10
        n_days = 5000  # Long series for stable correlation estimates
        corr_gen = HierarchicalCorrelationGenerator(
            n_assets, intra_cluster_corr=0.8, inter_cluster_corr=0.1
        )
        models = [GeometricBrownianMotion(start_price=100.0, drift=0.0, volatility=0.2)
                  for _ in range(n_assets)]
        gen = CorrelatedTimeSeriesGenerator(corr_gen, models)
        prices, target_corr = gen.generate(n_days)

        # Compute realized return correlations
        log_returns = np.diff(np.log(np.vstack([
            np.full((1, n_assets), 100.0), prices
        ])), axis=0)
        realized_corr = np.corrcoef(log_returns.T)

        # Mean absolute error should be small
        upper_tri = np.triu_indices(n_assets, k=1)
        mae = np.mean(np.abs(realized_corr[upper_tri] - target_corr[upper_tri]))
        assert mae < 0.15, f"Correlation MAE too high: {mae:.3f}"

    def test_large_mixed_portfolio(self):
        """50-asset portfolio with mixed model types."""
        np.random.seed(500)
        n_assets = 50
        n_days = 252
        corr_gen = HierarchicalCorrelationGenerator(n_assets)

        model_classes = [
            lambda: GeometricBrownianMotion(start_price=100.0, drift=0.05, volatility=0.2),
            lambda: GARCH(start_price=100.0, drift=0.05, volatility=0.2),
            lambda: AR1GBM(start_price=100.0, drift=0.05, volatility=0.2),
            lambda: HestonStochasticVolatility(start_price=100.0, drift=0.05, volatility=0.2),
            lambda: OrnsteinUhlenbeck(start_price=100.0, volatility=0.2, kappa=5.0),
        ]
        models = [model_classes[i % len(model_classes)]() for i in range(n_assets)]
        gen = CorrelatedTimeSeriesGenerator(corr_gen, models)
        prices, corr_matrix = gen.generate(n_days)

        assert prices.shape == (n_days, n_assets)
        assert np.all(prices > 0)
        assert np.all(np.isfinite(prices))
