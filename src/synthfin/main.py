from datetime import datetime
from typing import Dict, Any

import numpy as np
import yaml

from synthfin.correlation import NaiveCorrelationGenerator, HierarchicalCorrelationGenerator
from synthfin.models import GeometricBrownianMotion, JumpDiffusion
from synthfin.generator import CorrelatedTimeSeriesGenerator
from synthfin.output import DataFrameFormatter
from synthfin.visualization import TimeSeriesVisualizer


# Model registries
CORRELATION_MODELS = {
    "naive": NaiveCorrelationGenerator,
    "hierarchical": HierarchicalCorrelationGenerator
}

TIME_SERIES_MODELS = {
    "gbm": GeometricBrownianMotion,
    "jump_diffusion": JumpDiffusion
}

OUTPUT_FORMATTERS = {
    "dataframe": DataFrameFormatter
}


class SyntheticTimeSeriesPipeline:
    """
    Main pipeline for generating synthetic financial time series.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.correlation_generator = None
        self.time_series_models = None
        self.generator = None
        self.formatter = None
        self.visualizer = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup(self):
        """
        Set up all pipeline components based on configuration.
        """
        # Set up correlation generator
        corr_config = self.config['correlation']
        corr_model_name = corr_config['model']
        corr_params = corr_config['parameters'].get(corr_model_name, {})
        
        if corr_model_name not in CORRELATION_MODELS:
            raise ValueError(f"Unknown correlation model: {corr_model_name}")
        
        CorrelationModel = CORRELATION_MODELS[corr_model_name]
        n_assets = self.config['simulation']['n_assets']
        self.correlation_generator = CorrelationModel(n_assets, **corr_params)
        
        # Set up time series models
        ts_config = self.config['time_series']
        ts_model_name = ts_config['model']
        common_params = ts_config['common']
        specific_params = ts_config['parameters'].get(ts_model_name, {})
        
        if ts_model_name not in TIME_SERIES_MODELS:
            raise ValueError(f"Unknown time series model: {ts_model_name}")
        
        TimeSeriesModel = TIME_SERIES_MODELS[ts_model_name]
        
        # Generate random starting prices
        price_range = self.config['simulation']['price_range']
        start_prices = np.random.uniform(
            price_range['min'], 
            price_range['max'], 
            n_assets
        )
        
        # Create time series models for each asset
        self.time_series_models = []
        for i in range(n_assets):
            model = TimeSeriesModel(
                start_price=start_prices[i],
                **common_params,
                **specific_params
            )
            self.time_series_models.append(model)
        
        # Set up generator
        self.generator = CorrelatedTimeSeriesGenerator(
            self.correlation_generator,
            self.time_series_models
        )
        
        # Set up output formatter
        output_config = self.config['output']
        formatter_name = output_config['formatter']
        formatter_params = output_config.get(formatter_name, {})
        
        if formatter_name not in OUTPUT_FORMATTERS:
            raise ValueError(f"Unknown output formatter: {formatter_name}")
        
        FormatterClass = OUTPUT_FORMATTERS[formatter_name]
        self.formatter = FormatterClass(**formatter_params)
        
        # Set up visualizer
        viz_config = self.config['visualization']
        if viz_config['enabled']:
            self.visualizer = TimeSeriesVisualizer(
                figsize=tuple(viz_config['figsize'])
            )
    
    def run(self):
        """
        Run the complete pipeline.
        """
        print("Setting up pipeline components...")
        self.setup()
        
        # Generate time series
        print("Generating correlated time series...")
        n_days = self.config['simulation']['n_days']
        price_matrix, correlation_matrix = self.generator.generate(n_days)
        
        # Format output
        print("Formatting output...")
        start_date = datetime.strptime(
            self.config['simulation']['start_date'], 
            "%Y-%m-%d"
        )
        df = self.formatter.format(price_matrix, start_date)
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print(f"Number of assets: {len(df.columns)}")
        print(f"Number of trading days: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\nPrice ranges:")
        for ticker in df.columns[:5]:  # Show first 5 assets
            print(f"  {ticker}: ${df[ticker].min():.2f} - ${df[ticker].max():.2f}")
        if len(df.columns) > 5:
            print(f"  ... and {len(df.columns) - 5} more assets")
        
        # Create visualizations
        if self.visualizer and self.config['visualization']['enabled']:
            print("\nCreating visualizations...")
            self.visualizer.plot_all(
                df, 
                correlation_matrix,
                save_plots=self.config['visualization']['save_plots'],
                plot_prefix=self.config['visualization']['plot_prefix']
            )
        
        return df, correlation_matrix


def main():
    """
    Main entry point.
    """
    # Create and run pipeline
    pipeline = SyntheticTimeSeriesPipeline("config.yaml")
    df, correlation_matrix = pipeline.run()
    
    print("\nPipeline completed successfully!")
    print(f"Generated data shape: {df.shape}")
    
    # Display first few rows
    print("\nFirst few rows of generated data:")
    print(df.head())
    
    # Display correlation matrix stats
    print(f"\nCorrelation matrix statistics:")
    print(f"  Mean correlation: {np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}")
    print(f"  Std correlation: {np.std(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}")


if __name__ == "__main__":
    main()