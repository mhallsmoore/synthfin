from abc import ABC, abstractmethod
from datetime import datetime
import random
import string

import numpy as np
import pandas as pd


class OutputFormatter(ABC):
    """
    Abstract base class for output formatters.
    """
    
    @abstractmethod
    def format(self, price_matrix: np.ndarray, start_date: datetime) -> any:
        """
        Format the price matrix into the desired output format.
        """
        pass


class DataFrameFormatter(OutputFormatter):
    """
    Formats price matrix as a Pandas DataFrame with business days.
    """
    
    def __init__(self, save_to_csv: bool = True, csv_filename: str = "synthetic_prices.csv"):
        """
        Initialize the DataFrame formatter.
        
        Args:
            save_to_csv: Whether to save the DataFrame to CSV
            csv_filename: Filename for CSV output
        """
        self.save_to_csv = save_to_csv
        self.csv_filename = csv_filename
    
    def _generate_ticker(self, length: int) -> str:
        """
        Generate a random ticker symbol.
        """
        return ''.join(random.choices(string.ascii_uppercase, k=length))
    
    def format(self, price_matrix: np.ndarray, start_date: datetime) -> pd.DataFrame:
        """
        Format the price matrix as a DataFrame.
        
        Args:
            price_matrix: n_days x n_assets matrix of prices
            start_date: Starting date for the time series
        
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        n_days, n_assets = price_matrix.shape
        
        # Generate business day index
        date_index = pd.bdate_range(start=start_date, periods=n_days)
        
        # Generate random ticker symbols (3-5 characters)
        tickers = []
        used_tickers = set()
        
        for _ in range(n_assets):
            while True:
                length = random.randint(3, 5)
                ticker = self._generate_ticker(length)
                if ticker not in used_tickers:
                    used_tickers.add(ticker)
                    tickers.append(ticker)
                    break
        
        # Create DataFrame
        df = pd.DataFrame(price_matrix, index=date_index, columns=tickers)
        
        # Save to CSV if requested
        if self.save_to_csv:
            df.to_csv(self.csv_filename)
            print(f"Saved price data to {self.csv_filename}")
        
        return df