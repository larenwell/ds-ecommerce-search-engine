"""
Base retriever abstract class.

Defines the interface that all retrieval models must implement.
This ensures consistency across different retrieval strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval models.
    
    This class defines the interface that all concrete retrievers must implement,
    ensuring consistency across different retrieval strategies (TF-IDF, BM25, 
    Semantic, Hybrid, etc.).
    
    All retrievers must implement:
    - fit(): Train/index the product catalog
    - retrieve(): Get top-k products for a query
    
    Optional methods that can be overridden:
    - retrieve_with_scores(): Get products with relevance scores
    - batch_retrieve(): Process multiple queries efficiently
    """
    
    def __init__(self, top_k: int = 10):
        """
        Initialize the base retriever.
        
        Args:
            top_k: Default number of top results to retrieve
        """
        self.top_k = top_k
        self.is_fitted = False
        self.product_df = None
        
        logger.debug(f"Initialized {self.__class__.__name__} with top_k={top_k}")
    
    @abstractmethod
    def fit(self, product_df: pd.DataFrame) -> 'BaseRetriever':
        """
        Fit the retriever on the product catalog.
        
        This method should:
        1. Store the product dataframe
        2. Build necessary indexes (TF-IDF matrix, embeddings, etc.)
        3. Set self.is_fitted = True
        
        Args:
            product_df: DataFrame containing product information with at least
                       'product_id' and 'product_name' columns
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If product_df is invalid or missing required columns
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve top-k products for a given query.
        
        This method should:
        1. Validate that the retriever has been fitted
        2. Process the query
        3. Compute relevance scores
        4. Return top-k product IDs ranked by relevance
        
        Args:
            query: Search query string
            top_k: Number of results to retrieve (overrides self.top_k if provided)
            
        Returns:
            List of product IDs ranked by relevance (most relevant first)
            
        Raises:
            RuntimeError: If retriever hasn't been fitted yet
        """
        pass
    
    def retrieve_with_scores(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k products with their relevance scores.
        
        Default implementation returns scores as 1.0. Subclasses should
        override this for actual score computation.
        
        Args:
            query: Search query string
            top_k: Number of results to retrieve
            
        Returns:
            List of tuples (product_id, score) ranked by relevance
        """
        product_ids = self.retrieve(query, top_k)
        return [(pid, 1.0) for pid in product_ids]
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[int]]:
        """
        Retrieve results for multiple queries.
        
        Default implementation processes queries sequentially. Subclasses
        can override for optimized batch processing.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            show_progress: Whether to show progress bar
            
        Returns:
            List of result lists, one per query
        """
        results = []
        iterator = queries
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(queries, desc="Batch retrieval")
            except ImportError:
                logger.warning("tqdm not available, progress bar disabled")
        
        for query in iterator:
            try:
                results.append(self.retrieve(query, top_k))
            except Exception as e:
                logger.error(f"Error retrieving results for query '{query}': {e}")
                results.append([])  # Append empty list on error
        
        return results
    
    def _validate_fitted(self) -> None:
        """
        Check if the retriever has been fitted.
        
        Raises:
            RuntimeError: If the retriever hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before retrieving. "
                "Call fit(product_df) first."
            )
    
    def _validate_product_df(self, df: pd.DataFrame) -> None:
        """
        Validate that product dataframe has required columns.
        
        Args:
            df: Product dataframe to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['product_id']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(
                f"Product dataframe missing required columns: {missing_cols}"
            )
        
        if df.empty:
            raise ValueError("Product dataframe is empty")
        
        logger.debug(f"Product dataframe validated: {len(df)} products")
    
    def _prepare_text_field(
        self, 
        df: pd.DataFrame, 
        fields: List[str],
        fill_na: str = "",
        separator: str = " "
    ) -> pd.Series:
        """
        Combine and clean text fields from dataframe.
        
        This helper method combines multiple text columns into a single
        text representation for each product, handling missing values.
        
        Args:
            df: Source dataframe
            fields: List of column names to combine
            fill_na: Value to use for NaN/missing fields
            separator: String to use between fields
            
        Returns:
            Series with combined text for each row
            
        Example:
            >>> text = self._prepare_text_field(
            ...     df, 
            ...     ['product_name', 'product_description'],
            ...     fill_na=""
            ... )
        """
        # Check which fields exist in dataframe
        available_fields = [f for f in fields if f in df.columns]
        
        if not available_fields:
            logger.warning(
                f"None of the specified fields {fields} found in dataframe. "
                f"Using empty strings."
            )
            return pd.Series([""] * len(df), index=df.index)
        
        if len(available_fields) < len(fields):
            missing = set(fields) - set(available_fields)
            logger.warning(f"Fields not found in dataframe: {missing}")
        
        # Start with first field
        combined = df[available_fields[0]].fillna(fill_na).astype(str)
        
        # Add remaining fields
        for field in available_fields[1:]:
            combined = combined + separator + df[field].fillna(fill_na).astype(str)
        
        logger.debug(
            f"Combined {len(available_fields)} text fields: {available_fields}"
        )
        
        return combined
    
    def get_product_info(self, product_id: int) -> Optional[pd.Series]:
        """
        Get full information for a product ID.
        
        Args:
            product_id: Product ID to look up
            
        Returns:
            Series with product information, or None if not found
        """
        if self.product_df is None:
            logger.warning("Product dataframe not loaded")
            return None
        
        matches = self.product_df[self.product_df['product_id'] == product_id]
        
        if matches.empty:
            logger.warning(f"Product ID {product_id} not found")
            return None
        
        return matches.iloc[0]
    
    def __repr__(self) -> str:
        """String representation of the retriever."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_products = len(self.product_df) if self.product_df is not None else 0
        return (
            f"{self.__class__.__name__}("
            f"top_k={self.top_k}, "
            f"status={status}, "
            f"n_products={n_products})"
        )
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()