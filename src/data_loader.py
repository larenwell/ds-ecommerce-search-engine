"""
Data loading and preprocessing utilities.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from loguru import logger

from .config import config


class DataLoader:
    """
    Handles loading and basic validation of WANDS dataset.
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir or config.data.data_dir
        
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all dataset files.
        
        Returns:
            Tuple of (query_df, product_df, label_df)
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        # Load files
        query_df = self._load_queries()
        product_df = self._load_products()
        label_df = self._load_labels()
        
        # Validate
        self._validate_data(query_df, product_df, label_df)
        
        logger.info("Data loaded and validated successfully")
        
        return query_df, product_df, label_df
    
    def _load_queries(self) -> pd.DataFrame:
        """Load query data."""
        path = self.data_dir / config.data.query_file
        logger.info(f"Loading queries from {path}")
        df = pd.read_csv(path, sep=config.data.separator)
        logger.info(f"Loaded {len(df)} queries")
        return df
    
    def _load_products(self) -> pd.DataFrame:
        """Load product data."""
        path = self.data_dir / config.data.product_file
        logger.info(f"Loading products from {path}")
        df = pd.read_csv(path, sep=config.data.separator)
        logger.info(f"Loaded {len(df)} products")
        return df
    
    def _load_labels(self) -> pd.DataFrame:
        """Load label data."""
        path = self.data_dir / config.data.label_file
        logger.info(f"Loading labels from {path}")
        df = pd.read_csv(path, sep=config.data.separator)
        logger.info(f"Loaded {len(df)} labels")
        return df
    
    def _validate_data(
        self, 
        query_df: pd.DataFrame,
        product_df: pd.DataFrame,
        label_df: pd.DataFrame
    ) -> None:
        """
        Validate loaded data.
        
        Args:
            query_df: Query dataframe
            product_df: Product dataframe
            label_df: Label dataframe
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_query_cols = ['query_id', 'query']
        required_product_cols = ['product_id', 'product_name']
        required_label_cols = ['query_id', 'product_id', 'label']
        
        missing_query = set(required_query_cols) - set(query_df.columns)
        if missing_query:
            raise ValueError(f"Query data missing columns: {missing_query}")
        
        missing_product = set(required_product_cols) - set(product_df.columns)
        if missing_product:
            raise ValueError(f"Product data missing columns: {missing_product}")
        
        missing_label = set(required_label_cols) - set(label_df.columns)
        if missing_label:
            raise ValueError(f"Label data missing columns: {missing_label}")
        
        logger.info("Data validation passed")