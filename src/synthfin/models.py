from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np


class TimeSeriesModel(ABC):
    """
    Abstract base class for time series models.
    """
    
    def __init__(self, start_price: float, drift: float = 0.0, volatility: float = 0.2,
                 dt: float = 1/252, **kwargs):
        """
        Initialize the time series model.
        
        Args:
            start_price: Initial price of the asset
            drift: Annual drift parameter (mu)
            volatility: Annual volatility parameter (sigma)
            dt: Time step (default: 1/252 for daily data)
            **kwargs: Additional model-specific parameters
        """
        self.start_price = start_price
        self.drift = drift
        self.volatility = volatility
        self.dt = dt
        self.kwargs = kwargs
    
    @abstractmethod
    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        """
        Generate a price path given random shocks.
        
        Args:
            n_steps: Number of time steps
            random_shocks: Array of random shocks (already correlated)
        
        Returns:
            Array of prices
        """
        pass


class GeometricBrownianMotion(TimeSeriesModel):
    """
    Geometric Brownian Motion model for asset prices.
    """
    
    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        """
        Generate a GBM price path.
        """
        prices = np.zeros(n_steps + 1)
        prices[0] = self.start_price
        
        # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        dt_sqrt = np.sqrt(self.dt)
        
        for t in range(n_steps):
            drift_component = (self.drift - 0.5 * self.volatility**2) * self.dt
            diffusion_component = self.volatility * dt_sqrt * random_shocks[t]
            prices[t + 1] = prices[t] * np.exp(drift_component + diffusion_component)
        
        return prices[1:]  # Return prices excluding the initial price


class JumpDiffusion(TimeSeriesModel):
    """
    Jump-Diffusion model (Merton model) for asset prices.
    """
    
    def __init__(self, start_price: float, drift: float = 0.0, volatility: float = 0.2,
                 dt: float = 1/252, jump_intensity: float = 0.1, jump_mean: float = 0.0,
                 jump_std: float = 0.1, **kwargs):
        """
        Initialize the Jump-Diffusion model.
        
        Args:
            jump_intensity: Average number of jumps per year (lambda)
            jump_mean: Mean of jump size (log-normal)
            jump_std: Standard deviation of jump size (log-normal)
        """
        super().__init__(start_price, drift, volatility, dt, **kwargs)
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
    
    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        """
        Generate a Jump-Diffusion price path.
        """
        prices = np.zeros(n_steps + 1)
        prices[0] = self.start_price
        
        dt_sqrt = np.sqrt(self.dt)
        
        # Generate jump occurrences
        jump_probs = self.jump_intensity * self.dt
        jumps_occur = np.random.binomial(1, jump_probs, n_steps)
        
        # Generate jump sizes
        jump_sizes = np.zeros(n_steps)
        n_jumps = jumps_occur.sum()
        if n_jumps > 0:
            # Log-normal jumps
            jump_sizes[jumps_occur == 1] = np.exp(
                np.random.normal(self.jump_mean, self.jump_std, n_jumps)
            ) - 1
        
        for t in range(n_steps):
            # GBM component
            drift_component = (self.drift - 0.5 * self.volatility**2) * self.dt
            diffusion_component = self.volatility * dt_sqrt * random_shocks[t]
            
            # Jump component
            jump_component = jump_sizes[t]
            
            # Combined price evolution
            prices[t + 1] = prices[t] * np.exp(drift_component + diffusion_component) * (1 + jump_component)
        
        return prices[1:]  # Return prices excluding the initial price