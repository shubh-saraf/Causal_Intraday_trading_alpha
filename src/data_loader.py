"""
DataLoader Module - Handles loading and basic preprocessing of intraday data.

Causality: This module ensures data is loaded in chronological order and
preserves the temporal structure required for causal processing.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads intraday trading data from CSV files.
    
    Ensures proper ordering by timestamp and handles basic validation.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        if data_dir is None:
            # Default to 'data/' relative to project root
            data_dir = str(Path(__file__).parent.parent / 'data')
        self.data_dir = Path(data_dir)
        self.price_cols = ['P1', 'P2', 'P3', 'P4']
        self.timestamp_col = 'ts_ns'
        
    def load_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a single CSV file and perform basic validation.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with validated data sorted by timestamp
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns
        if self.timestamp_col not in df.columns:
            raise ValueError(f"Missing timestamp column: {self.timestamp_col}")
        if 'P3' not in df.columns:
            raise ValueError("Missing P3 (tradeable price) column")
        
        # Ensure chronological order (CAUSAL: critical for time-series)
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Add row index as explicit time reference
        df['bar_index'] = np.arange(len(df))
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_multiple_files(self, file_numbers: List[int]) -> pd.DataFrame:
        """
        Load multiple files (days) for training.
        
        Args:
            file_numbers: List of file numbers to load (e.g., [1, 2, 3])
            
        Returns:
            Combined DataFrame with day identifier
        """
        dfs = []
        for num in file_numbers:
            filepath = self.data_dir / f"{num}.csv"
            if filepath.exists():
                df = self.load_file(str(filepath))
                df['day'] = num
                dfs.append(df)
            else:
                logger.warning(f"File not found: {filepath}")
        
        if not dfs:
            raise ValueError("No valid files found")
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(file_numbers)} days: {len(combined)} total rows")
        return combined
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding prices, timestamps, etc.).
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        exclude = set(self.price_cols + [self.timestamp_col, 'bar_index', 'day'])
        return [c for c in df.columns if c not in exclude]
    
    def get_available_days(self) -> List[int]:
        """Get list of available day numbers in the data directory."""
        files = list(self.data_dir.glob("*.csv"))
        days = []
        for f in files:
            try:
                days.append(int(f.stem))
            except ValueError:
                continue
        return sorted(days)