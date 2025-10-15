"""
Enhanced TF-IDF Retriever with improved text processing.

Improvements over baseline:
1. Uses bigrams (1-2 grams) to capture phrases like "dining table"
2. Adds min/max document frequency filters to reduce noise
3. Combines multiple product fields (name, description, features)
4. Better text normalization with stop words removal
5. Configurable parameters via config system
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from .base import BaseRetriever
from ..config import config


class TFIDFRetriever(BaseRetriever):
    """
    Enhanced TF-IDF based retrieval with multiple improvements.
    
    Key improvements over baseline:
    - Uses n-grams (1-2) to capture phrases like "dining table"
    - Filters by document frequency to reduce noise (min_df, max_df)
    - Combines name, description, and features for richer representation
    - Removes stop words for better focus on content words
    - Proper text cleaning and normalization
    
    Example:
        >>> retriever = TFIDFRetriever(top_k=10, ngram_range=(1, 2))
        >>> retriever.fit(product_df)
        >>> results = retriever.retrieve("comfortable armchair")
    """
    
    def __init__(
        self,
        top_k: int = 10,
        max_features: Optional[int] = None,
        ngram_range: Optional[Tuple[int, int]] = None,
        min_df: Optional[int] = None,
        max_df: Optional[float] = None,
        text_fields: Optional[List[str]] = None
    ):
        """
        Initialize TF-IDF retriever.
        
        Args:
            top_k: Number of top results to retrieve
            max_features: Maximum number of features (vocabulary size)
                         If None, uses config default
            ngram_range: Range of n-grams to consider (min_n, max_n)
                        If None, uses config default
            min_df: Minimum document frequency for terms
                    If None, uses config default
            max_df: Maximum document frequency for terms (as ratio)
                    If None, uses config default
            text_fields: List of product fields to use for text representation
                        If None, uses default: name + description + features
        """
        super().__init__(top_k)
        
        # Use config defaults if not provided
        self.max_features = max_features if max_features is not None else config.retriever.tfidf_max_features
        self.ngram_range = ngram_range if ngram_range is not None else config.retriever.tfidf_ngram_range
        self.min_df = min_df if min_df is not None else config.retriever.tfidf_min_df
        self.max_df = max_df if max_df is not None else config.retriever.tfidf_max_df
        
        # Default to using name, description, and features
        self.text_fields = text_fields or [
            'product_name', 
            'product_description',
            'product_features'
        ]
        
        # Initialize vectorizer with improved parameters
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',  # Remove common English stop words
            token_pattern=r'\b\w+\b'  # Word tokenization
        )
        
        self.tfidf_matrix = None
        
        logger.info(
            f"TFIDFRetriever initialized: ngrams={self.ngram_range}, "
            f"min_df={self.min_df}, max_df={self.max_df}"
        )
    
    def fit(self, product_df: pd.DataFrame) -> 'TFIDFRetriever':
        """
        Fit the TF-IDF vectorizer on the product catalog.
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If product_df is invalid or missing required columns
        """
        try:
            logger.info(f"Fitting TF-IDF retriever on {len(product_df)} products")
            
            # Validate input
            self._validate_product_df(product_df)
            
            # Store product dataframe
            self.product_df = product_df.copy()
            
            # Combine text fields, handling missing values
            available_fields = [f for f in self.text_fields if f in product_df.columns]
            
            if not available_fields:
                raise ValueError(
                    f"None of the specified text fields {self.text_fields} "
                    f"found in product dataframe. Available columns: {list(product_df.columns)}"
                )
            
            logger.info(f"Using text fields: {available_fields}")
            combined_text = self._prepare_text_field(
                product_df, 
                available_fields,
                fill_na="",
                separator=" "
            )
            
            # Fit and transform TF-IDF
            logger.info("Computing TF-IDF matrix...")
            self.tfidf_matrix = self.vectorizer.fit_transform(combined_text)
            
            self.is_fitted = True
            
            logger.info(
                f"TF-IDF fitted successfully: "
                f"vocabulary_size={len(self.vectorizer.vocabulary_)}, "
                f"matrix_shape={self.tfidf_matrix.shape}, "
                f"sparsity={1 - (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])):.4f}"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting TF-IDF retriever: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve top-k products for a query using TF-IDF similarity.
        
        Args:
            query: Search query string
            top_k: Number of results (overrides self.top_k if provided)
            
        Returns:
            List of product IDs ranked by relevance (descending)
            
        Raises:
            RuntimeError: If retriever hasn't been fitted
        """
        self._validate_fitted()
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k indices (sorted by similarity, descending)
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            # Get corresponding product IDs
            product_ids = self.product_df.iloc[top_indices]['product_id'].tolist()
            
            logger.debug(
                f"Retrieved {len(product_ids)} products for query: '{query[:50]}...' "
                f"(max_similarity={similarities[top_indices[0]]:.4f})"
            )
            
            return product_ids
            
        except Exception as e:
            logger.error(f"Error retrieving results for query '{query}': {e}")
            return []
    
    def retrieve_with_scores(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k products with their cosine similarity scores.
        
        Args:
            query: Search query string
            top_k: Number of results
            
        Returns:
            List of (product_id, score) tuples, sorted by score descending
        """
        self._validate_fitted()
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            top_scores = similarities[top_indices]
            
            # Get product IDs
            product_ids = self.product_df.iloc[top_indices]['product_id'].tolist()
            
            # Combine into tuples
            results = list(zip(product_ids, top_scores))
            
            logger.debug(
                f"Retrieved {len(results)} products with scores for query: '{query[:50]}...'"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving results with scores: {e}")
            return []
    
    def get_feature_importance(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most important features (terms) for a query.
        
        Useful for understanding what terms drive the ranking.
        
        Args:
            query: Search query string
            top_n: Number of top features to return
            
        Returns:
            List of (term, weight) tuples sorted by weight
        """
        self._validate_fitted()
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Get feature names and weights
            feature_names = self.vectorizer.get_feature_names_out()
            query_weights = query_vector.toarray()[0]
            
            # Get non-zero features
            non_zero_idx = query_weights.nonzero()[0]
            
            if len(non_zero_idx) == 0:
                logger.warning(f"No features found for query: '{query}'")
                return []
            
            # Sort by weight
            sorted_idx = non_zero_idx[np.argsort(query_weights[non_zero_idx])[::-1]]
            
            # Get top-n
            top_features = [
                (feature_names[idx], query_weights[idx]) 
                for idx in sorted_idx[:top_n]
            ]
            
            return top_features
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return []