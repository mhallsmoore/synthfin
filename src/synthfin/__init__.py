# src/synthfin/__init__.py

"""
SynthFin: Synthetic Financial Time Series Generation

A Python package for generating realistic synthetic financial time series data
with customizable correlation structures and stochastic process models.
"""

__version__ = "0.1.0"
__author__ = "Michael Halls-Moore"
__email__ = "support@quantstart.com"

# Import main components for easy access
from .correlation import (
    CorrelationMatrixGenerator,
    NaiveCorrelationGenerator,
    HierarchicalCorrelationGenerator,
)
from .models import (
    TimeSeriesModel,
    GeometricBrownianMotion,
    JumpDiffusion,
)
from .generator import CorrelatedTimeSeriesGenerator
from .output import (
    OutputFormatter,
    DataFrameFormatter,
)
from .visualization import (
    TimeSeriesVisualizer,
    create_individual_plots,
)
from .main import (
    SyntheticTimeSeriesPipeline,
    CORRELATION_MODELS,
    TIME_SERIES_MODELS,
    OUTPUT_FORMATTERS,
)
from .utils import (
    create_config,
    load_config,
    save_config,
    validate_correlation_matrix,
)

# Define what should be imported with "from synthfin import *"
__all__ = [
    # Version info
    "__version__",
    
    # Correlation generators
    "CorrelationMatrixGenerator",
    "NaiveCorrelationGenerator",
    "HierarchicalCorrelationGenerator",
    
    # Time series models
    "TimeSeriesModel",
    "GeometricBrownianMotion",
    "JumpDiffusion",
    
    # Core functionality
    "CorrelatedTimeSeriesGenerator",
    "SyntheticTimeSeriesPipeline",
    
    # Output and visualization
    "OutputFormatter",
    "DataFrameFormatter",
    "TimeSeriesVisualizer",
    "create_individual_plots",
    
    # Utilities
    "create_config",
    "load_config",
    "save_config",
    "validate_correlation_matrix",
    
    # Model registries
    "CORRELATION_MODELS",
    "TIME_SERIES_MODELS",
    "OUTPUT_FORMATTERS",
]