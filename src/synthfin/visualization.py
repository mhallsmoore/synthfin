# visualization.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import seaborn as sns


class TimeSeriesVisualizer:
    """Visualizes correlation matrices and time series data."""
    
    def __init__(self, figsize: tuple = (15, 12)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
    
    def plot_all(self, df: pd.DataFrame, correlation_matrix: np.ndarray,
                 save_plots: bool = True, plot_prefix: str = "synthetic_"):
        """
        Create all three plots.
        
        Args:
            df: DataFrame with price data
            correlation_matrix: Correlation matrix used for generation
            save_plots: Whether to save plots to files
            plot_prefix: Prefix for saved plot files
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Create a grid spec for better control
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Plot correlation matrix (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_correlation_matrix(correlation_matrix, ax1)
        
        # 2. Plot price heatmap (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_price_heatmap(df, ax2)
        
        # 3. Plot time series (bottom row, spanning both columns)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_time_series(df, ax3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_prefix}analysis.png", dpi=300, bbox_inches='tight')
            print(f"Saved plots to {plot_prefix}analysis.png")
        
        plt.show()
    
    def _plot_correlation_matrix(self, correlation_matrix: np.ndarray, ax: plt.Axes):
        """Plot the correlation matrix."""
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Asset Index')
        ax.set_ylabel('Asset Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    def _plot_price_heatmap(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot price data as a heatmap."""
        # Normalize prices for better visualization
        normalized_prices = (df - df.min()) / (df.max() - df.min())
        
        im = ax.imshow(normalized_prices.T, aspect='auto', cmap='viridis')
        ax.set_title('Normalized Price Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Assets')
        
        # Set y-tick labels to asset names
        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Price', rotation=270, labelpad=15)
    
    def _plot_time_series(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot all time series on one chart."""
        # Plot each asset
        for column in df.columns:
            ax.plot(df.index, df[column], label=column, alpha=0.7, linewidth=1)
        
        ax.set_title('Synthetic Price Time Series', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Remove x-axis margins to make data touch the edges
        ax.set_xlim(df.index[0], df.index[-1])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Legend
        if len(df.columns) <= 20:  # Only show legend if not too many assets
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)


def create_individual_plots(df: pd.DataFrame, correlation_matrix: np.ndarray,
                          save_plots: bool = True):
    """
    Create individual plots (alternative to combined plot).
    
    Args:
        df: DataFrame with price data
        correlation_matrix: Correlation matrix
        save_plots: Whether to save plots
    """
    # 1. Correlation matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix')
    plt.xlabel('Asset Index')
    plt.ylabel('Asset Index')
    if save_plots:
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Price heatmap
    plt.figure(figsize=(10, 6))
    normalized_prices = (df - df.min()) / (df.max() - df.min())
    plt.imshow(normalized_prices.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Price')
    plt.title('Normalized Price Heatmap')
    plt.xlabel('Time (days)')
    plt.ylabel('Assets')
    plt.yticks(range(len(df.columns)), df.columns, fontsize=8)
    if save_plots:
        plt.savefig('price_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Time series
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column, alpha=0.7, linewidth=1)
    plt.title('Synthetic Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    if len(df.columns) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    if save_plots:
        plt.savefig('time_series.png', dpi=300, bbox_inches='tight')
    plt.show()