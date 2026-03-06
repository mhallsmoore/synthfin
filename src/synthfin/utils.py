import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml


def create_config(
    correlation_model: str = "hierarchical",
    correlation_params: Optional[Dict[str, Any]] = None,
    time_series_model: str = "gbm",
    time_series_params: Optional[Dict[str, Any]] = None,
    n_assets: int = 20,
    n_days: int = 252,
    start_date: str = "2024-01-02",
    price_min: float = 10,
    price_max: float = 1000,
    save_csv: bool = True,
    csv_filename: str = "synthetic_prices.csv",
    enable_viz: bool = True,
    save_plots: bool = True,
    plot_prefix: str = "synthetic_"
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for the pipeline.
    
    Args:
        correlation_model: Name of correlation model
        correlation_params: Parameters for correlation model
        time_series_model: Name of time series model
        time_series_params: Parameters for time series model
        n_assets: Number of assets to simulate
        n_days: Number of days to simulate
        start_date: Starting date for simulation
        price_min: Minimum initial price
        price_max: Maximum initial price
        save_csv: Whether to save output to CSV
        csv_filename: Filename for CSV output
        enable_viz: Whether to enable visualization
        save_plots: Whether to save plots
        plot_prefix: Prefix for saved plots
        
    Returns:
        Configuration dictionary
    """
    # Default parameters
    default_corr_params = {
        "naive": {},
        "hierarchical": {
            "n_clusters": None,
            "intra_cluster_corr": 0.7,
            "inter_cluster_corr": 0.2,
            "noise_level": 0.1
        }
    }
    
    default_ts_params = {
        "gbm": {
            "drift": 0.05,
            "volatility": 0.2,
            "dt": 1/252
        },
        "jump_diffusion": {
            "drift": 0.05,
            "volatility": 0.2,
            "dt": 1/252,
            "jump_intensity": 0.1,
            "jump_mean": 0.0,
            "jump_std": 0.1
        },
        "ar1_gbm": {
            "drift": 0.05,
            "volatility": 0.2,
            "dt": 1/252,
            "ar_coeff": 0.05
        },
        "garch": {
            "drift": 0.05,
            "volatility": 0.2,
            "dt": 1/252,
            "alpha": 0.05,
            "beta": 0.90
        },
        "arma_garch": {
            "drift": 0.05,
            "volatility": 0.2,
            "dt": 1/252,
            "ar_coeff": 0.05,
            "ma_coeff": -0.05,
            "alpha": 0.05,
            "beta": 0.90
        },
        "heston": {
            "drift": 0.05,
            "volatility": 0.2,
            "dt": 1/252,
            "kappa": 2.0,
            "sigma_v": 0.3,
            "rho": -0.7
        },
        "ornstein_uhlenbeck": {
            "drift": 0.05,
            "volatility": 0.2,
            "dt": 1/252,
            "kappa": 5.0
        }
    }
    
    # Get default parameters for selected models
    corr_params = correlation_params or default_corr_params.get(correlation_model, {})
    
    # For time series, merge common and specific parameters
    ts_common = {
        "drift": 0.05,
        "volatility": 0.2,
        "dt": 1/252
    }
    ts_specific = time_series_params or {}
    
    # Update common parameters with any provided values
    for key in ["drift", "volatility", "dt"]:
        if key in ts_specific:
            ts_common[key] = ts_specific.pop(key)
    
    config = {
        "correlation": {
            "model": correlation_model,
            "parameters": {
                correlation_model: corr_params
            }
        },
        "time_series": {
            "model": time_series_model,
            "common": ts_common,
            "parameters": {
                time_series_model: ts_specific
            }
        },
        "simulation": {
            "n_assets": n_assets,
            "n_days": n_days,
            "start_date": start_date,
            "price_range": {
                "min": price_min,
                "max": price_max
            }
        },
        "output": {
            "formatter": "dataframe",
            "dataframe": {
                "save_to_csv": save_csv,
                "csv_filename": csv_filename
            }
        },
        "visualization": {
            "enabled": enable_viz,
            "save_plots": save_plots,
            "plot_prefix": plot_prefix,
            "figsize": [15, 12]
        }
    }
    
    return config


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    path = Path(filepath)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def validate_correlation_matrix(matrix: np.ndarray) -> bool:
    """
    Validate that a matrix is a valid correlation matrix.
    
    Args:
        matrix: Matrix to validate
        
    Returns:
        True if valid correlation matrix, False otherwise
    """
    # Check if square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if symmetric
    if not np.allclose(matrix, matrix.T):
        return False
    
    # Check diagonal elements are 1
    if not np.allclose(np.diag(matrix), 1):
        return False
    
    # Check all elements are in [-1, 1]
    if np.any(matrix < -1) or np.any(matrix > 1):
        return False
    
    # Check if positive semi-definite
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues < -1e-8):  # Small tolerance for numerical errors
        return False
    
    return True


def generate_default_config(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate and optionally save a default configuration file.
    
    Args:
        output_path: Path to save configuration (optional)
        
    Returns:
        Default configuration dictionary
    """
    config = create_config()
    
    if output_path:
        save_config(config, output_path)
        print(f"Default configuration saved to: {output_path}")
    
    return config