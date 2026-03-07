from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np


class TimeSeriesModel(ABC):
    """
    Abstract base class for time series models.
    """

    n_shock_streams: int = 1

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


class AR1GBM(TimeSeriesModel):
    """
    Autoregressive AR(1) model with GBM price dynamics.
    """

    def __init__(self, start_price: float, drift: float = 0.0, volatility: float = 0.2,
                 dt: float = 1/252, ar_coeff: float = 0.05, **kwargs):
        super().__init__(start_price, drift, volatility, dt, **kwargs)
        if not -1 < ar_coeff < 1:
            raise ValueError(f"ar_coeff must be in (-1, 1), got {ar_coeff}")
        self.ar_coeff = ar_coeff

    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        prices = np.zeros(n_steps + 1)
        prices[0] = self.start_price

        dt_sqrt = np.sqrt(self.dt)
        mu = self.drift
        sigma = self.volatility
        phi = self.ar_coeff

        r_prev = 0.0

        for t in range(n_steps):
            r_t = (phi * r_prev
                   + (1 - phi) * mu * self.dt
                   - 0.5 * sigma**2 * self.dt
                   + sigma * dt_sqrt * random_shocks[t])
            prices[t + 1] = prices[t] * np.exp(r_t)
            r_prev = r_t

        return prices[1:]


class GARCH(TimeSeriesModel):
    """
    GARCH(1,1) model with time-varying volatility.
    """

    def __init__(self, start_price: float, drift: float = 0.0, volatility: float = 0.2,
                 dt: float = 1/252, alpha: float = 0.05, beta: float = 0.90,
                 omega: float = None, **kwargs):
        super().__init__(start_price, drift, volatility, dt, **kwargs)
        if alpha + beta >= 1:
            raise ValueError(f"alpha + beta must be < 1, got {alpha + beta}")
        self.alpha = alpha
        self.beta = beta
        self.omega = omega if omega is not None else volatility**2 * dt * (1 - alpha - beta)

    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        prices = np.zeros(n_steps + 1)
        prices[0] = self.start_price

        mu = self.drift
        v_t = self.volatility**2 * self.dt  # Initial variance
        epsilon_prev = 0.0

        for t in range(n_steps):
            v_t = self.omega + self.alpha * epsilon_prev**2 + self.beta * v_t
            v_t = max(v_t, 1e-12)
            epsilon_t = np.sqrt(v_t) * random_shocks[t]
            r_t = mu * self.dt - 0.5 * v_t + epsilon_t
            prices[t + 1] = prices[t] * np.exp(r_t)
            epsilon_prev = epsilon_t

        return prices[1:]


class ARMAGARCH(TimeSeriesModel):
    """
    ARMA(1,1)-GARCH(1,1) model combining autoregressive returns with time-varying volatility.
    """

    def __init__(self, start_price: float, drift: float = 0.0, volatility: float = 0.2,
                 dt: float = 1/252, ar_coeff: float = 0.05, ma_coeff: float = -0.05,
                 alpha: float = 0.05, beta: float = 0.90, omega: float = None, **kwargs):
        super().__init__(start_price, drift, volatility, dt, **kwargs)
        if not -1 < ar_coeff < 1:
            raise ValueError(f"ar_coeff must be in (-1, 1), got {ar_coeff}")
        if alpha + beta >= 1:
            raise ValueError(f"alpha + beta must be < 1, got {alpha + beta}")
        self.ar_coeff = ar_coeff
        self.ma_coeff = ma_coeff
        self.alpha = alpha
        self.beta = beta
        self.omega = omega if omega is not None else volatility**2 * dt * (1 - alpha - beta)

    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        prices = np.zeros(n_steps + 1)
        prices[0] = self.start_price

        c = self.drift * self.dt
        phi = self.ar_coeff
        theta = self.ma_coeff

        v_t = self.volatility**2 * self.dt
        epsilon_prev = 0.0
        r_prev = 0.0

        for t in range(n_steps):
            v_t = self.omega + self.alpha * epsilon_prev**2 + self.beta * v_t
            v_t = max(v_t, 1e-12)
            epsilon_t = np.sqrt(v_t) * random_shocks[t]
            r_t = c + phi * r_prev + epsilon_t + theta * epsilon_prev
            prices[t + 1] = prices[t] * np.exp(r_t)
            epsilon_prev = epsilon_t
            r_prev = r_t

        return prices[1:]


class HestonStochasticVolatility(TimeSeriesModel):
    """
    Heston stochastic volatility model with mean-reverting variance.
    Requires two shock streams: one for price (cross-asset correlated), one for variance.
    """

    n_shock_streams: int = 2

    def __init__(self, start_price: float, drift: float = 0.0, volatility: float = 0.2,
                 dt: float = 1/252, kappa: float = 2.0, theta_v: float = None,
                 sigma_v: float = 0.3, rho: float = -0.7, **kwargs):
        super().__init__(start_price, drift, volatility, dt, **kwargs)
        self.kappa = kappa
        self.theta_v = theta_v if theta_v is not None else volatility**2
        self.sigma_v = sigma_v
        self.rho = rho

    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        prices = np.zeros(n_steps + 1)
        prices[0] = self.start_price

        dt_sqrt = np.sqrt(self.dt)
        mu = self.drift

        # random_shocks is (n_steps, 2): col 0 = correlated price shock, col 1 = independent shock
        z1 = random_shocks[:, 0]
        w = random_shocks[:, 1]
        # Construct variance shock with internal rho correlation
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * w

        v_t = self.volatility**2  # Initial variance

        for t in range(n_steps):
            v_pos = max(v_t, 0.0)
            v_t = (v_pos
                   + self.kappa * (self.theta_v - v_pos) * self.dt
                   + self.sigma_v * np.sqrt(v_pos) * dt_sqrt * z2[t])
            v_pos_new = max(v_t, 0.0)
            prices[t + 1] = prices[t] * np.exp(
                (mu - 0.5 * v_pos_new) * self.dt + np.sqrt(v_pos_new) * dt_sqrt * z1[t]
            )

        return prices[1:]


class OrnsteinUhlenbeck(TimeSeriesModel):
    """
    Ornstein-Uhlenbeck mean-reverting process for log-prices.

    Note: The ``drift`` parameter is accepted for interface compatibility
    but is not used. Mean-reversion toward ``theta`` governs the dynamics.
    """

    def __init__(self, start_price: float, drift: float = 0.0, volatility: float = 0.2,
                 dt: float = 1/252, kappa: float = 5.0, theta: float = None, **kwargs):
        super().__init__(start_price, drift, volatility, dt, **kwargs)
        self.kappa = kappa
        self.theta = theta if theta is not None else np.log(start_price)

    def generate_path(self, n_steps: int, random_shocks: np.ndarray) -> np.ndarray:
        prices = np.zeros(n_steps + 1)
        prices[0] = self.start_price

        dt_sqrt = np.sqrt(self.dt)
        sigma = self.volatility
        x_t = np.log(self.start_price)

        for t in range(n_steps):
            x_t = x_t + self.kappa * (self.theta - x_t) * self.dt + sigma * dt_sqrt * random_shocks[t]
            prices[t + 1] = np.exp(x_t)

        return prices[1:]