from typing import List, Optional

import numpy as np

from synthfin.correlation import CorrelationMatrixGenerator
from synthfin.models import TimeSeriesModel


class CorrelatedTimeSeriesGenerator:
    """
    Generates correlated time series using Cholesky decomposition.
    """
    
    def __init__(self, correlation_generator: CorrelationMatrixGenerator,
                 time_series_models: List[TimeSeriesModel]):
        """
        Initialize the correlated time series generator.
        
        Args:
            correlation_generator: Generator for correlation matrices
            time_series_models: List of time series models (one per asset)
        """
        self.correlation_generator = correlation_generator
        self.time_series_models = time_series_models
        self.n_assets = len(time_series_models)
        
        if self.n_assets != correlation_generator.n:
            raise ValueError(f"Number of models ({self.n_assets}) must match "
                           f"correlation matrix size ({correlation_generator.n})")
    
    def generate(self, n_days: int) -> tuple:
        """
        Generate correlated time series.
        
        Args:
            n_days: Number of days to simulate
        
        Returns:
            Tuple of (price_matrix, correlation_matrix) where price_matrix is n_days x n_assets
        """
        # Generate correlation matrix
        correlation_matrix = self.correlation_generator.generate()
        
        # Perform Cholesky decomposition
        try:
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition as fallback
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            eigenvalues[eigenvalues < 0] = 1e-8
            cholesky_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        # Generate independent random shocks
        independent_shocks = np.random.standard_normal((n_days, self.n_assets))
        
        # Create correlated shocks using Cholesky decomposition
        correlated_shocks = independent_shocks @ cholesky_matrix.T
        
        # Generate price paths for each asset
        price_matrix = np.zeros((n_days, self.n_assets))
        
        for i, model in enumerate(self.time_series_models):
            price_matrix[:, i] = model.generate_path(n_days, correlated_shocks[:, i])
        
        return price_matrix, correlation_matrix