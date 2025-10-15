"""
BM25 Retriever Implementation.

BM25 (Best Matching 25) is a ranking function used in information retrieval 
that often outperforms TF-IDF for search tasks. It considers:
- Term frequency with saturation (diminishing returns for repeated terms)
- Document length normalization (fair comparison across varying lengths)
- Inverse document frequency

This is particularly effective for e-commerce search where document
lengths (product descriptions) can vary significantly.

Reference: Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework: BM25 and Beyond"
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from loguru import logger

from .base import BaseRetriever
from ..config import config


class BM25Retriever(BaseRetriever):
    """
    BM25 (Best Matching 25) retrieval model.
    
    BM25 advantages over TF-IDF for e-commerce:
    1. Better handling of term frequency saturation
       - Repeated terms have diminishing returns (avoids spam)
    2. Document length normalization 
       - Important when products have varying description lengths
    3. Generally performs better for search ranking tasks
    4. More robust to common terms that appear frequently
    5. Industry standard (used in Elasticsearch, Lucene, etc.)
    
    Parameters:
    - k1: Controls term frequency saturation (typical: 1.2-2.0)
      - Higher k1 = less saturation, term frequency matters more
      - Lower k1 = more saturation, diminishing returns kick in faster
    - b: Controls document length normalization (typical: 0.75)
      - b=1: Full length normalization
      - b=0: No length normalization
      - 0 < b < 1: Partial normalization
    
    Example:
        >>> retriever = BM25Retriever(k1=1.5, b=0.75)
        >>> retriever.fit(product_df)
        >>> results = retriever.retrieve("comfortable armchair")
    """
    
    def __init__(
        self,
        top_k: int = 10,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        text_fields: Optional[List[str]] = None
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            top_k: Number of top results to retrieve
            k1: BM25 k1 parameter (term frequency saturation)
                If None, uses config default (1.5)
            b: BM25 b parameter (length normalization)
               If None, uses config default (0.75)
            text_fields: List of product fields to use for text
                        If None, uses: name + description + features
        """
        super().__init__(top_k)
        
        # Use config defaults if not provided
        self.k1 = k1 if k1 is not None else config.retriever.bm25_k1
        self.b = b if b is not None else config.retriever.bm25_b
        
        # Default text fields
        self.text_fields = text_fields or [
            'product_name',
            'product_description', 
            'product_features'
        ]
        
        self.bm25 = None
        self.tokenized_corpus = None
        
        logger.info(
            f"BM25Retriever initialized: k1={self.k1}, b={self.b}, "
            f"fields={self.text_fields}"
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Uses lowercase splitting by whitespace. For production,
        consider more sophisticated tokenization.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (lowercase)
        """
        # Simple whitespace tokenization with lowercasing
        # Could be enhanced with stemming, lemmatization, etc.
        return text.lower().split()
    
    def fit(self, product_df: pd.DataFrame) -> 'BM25Retriever':
        """
        Fit the BM25 model on the product catalog.
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If product_df is invalid
        """
        try:
            logger.info(f"Fitting BM25 retriever on {len(product_df)} products")
            
            # Validate input
            self._validate_product_df(product_df)
            
            # Store product dataframe
            self.product_df = product_df.copy()
            
            # Combine text fields
            available_fields = [f for f in self.text_fields if f in product_df.columns]
            
            if not available_fields:
                raise ValueError(
                    f"None of the specified text fields {self.text_fields} "
                    f"found in product dataframe. Available: {list(product_df.columns)}"
                )
            
            logger.info(f"Using text fields: {available_fields}")
            combined_text = self._prepare_text_field(
                product_df, 
                available_fields,
                fill_na="",
                separator=" "
            )
            
            # Tokenize corpus
            logger.info("Tokenizing corpus...")
            self.tokenized_corpus = [
                self._tokenize(text) for text in combined_text
            ]
            
            # Initialize BM25
            logger.info("Building BM25 index...")
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )
            
            self.is_fitted = True
            
            # Calculate stats
            avg_doc_len = np.mean([len(doc) for doc in self.tokenized_corpus])
            
            logger.info(
                f"BM25 fitted successfully: "
                f"corpus_size={len(self.tokenized_corpus)}, "
                f"avg_doc_length={avg_doc_len:.1f} tokens, "
                f"k1={self.k1}, b={self.b}"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting BM25 retriever: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve top-k products using BM25 scoring.
        
        Args:
            query: Search query string
            top_k: Number of results (overrides self.top_k if provided)
            
        Returns:
            List of product IDs ranked by BM25 score (descending)
            
        Raises:
            RuntimeError: If retriever hasn't been fitted
        """
        self._validate_fitted()
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)
            
            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices (sorted by score, descending)
            top_indices = np.argsort(scores)[-k:][::-1]
            
            # Get corresponding product IDs
            product_ids = self.product_df.iloc[top_indices]['product_id'].tolist()
            
            logger.debug(
                f"Retrieved {len(product_ids)} products for query: '{query[:50]}...' "
                f"(max_score={scores[top_indices[0]]:.4f})"
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
        Retrieve top-k products with their BM25 scores.
        
        Args:
            query: Search query string
            top_k: Number of results
            
        Returns:
            List of (product_id, score) tuples sorted by score descending
        """
        self._validate_fitted()
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[-k:][::-1]
            top_scores = scores[top_indices]
            
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
    
    def get_query_terms_impact(self, query: str) -> List[Tuple[str, float]]:
        """
        Analyze which query terms have the most impact.
        
        Useful for understanding what drives the BM25 ranking.
        
        Args:
            query: Search query string
            
        Returns:
            List of (term, avg_score_contribution) tuples
        """
        self._validate_fitted()
        
        try:
            tokenized_query = self._tokenize(query)
            
            # Get scores for each term individually
            term_impacts = []
            
            for term in set(tokenized_query):  # Unique terms
                single_term_scores = self.bm25.get_scores([term])
                avg_score = np.mean(single_term_scores[single_term_scores > 0])
                
                if not np.isnan(avg_score):
                    term_impacts.append((term, avg_score))
            
            # Sort by impact
            term_impacts.sort(key=lambda x: x[1], reverse=True)
            
            return term_impacts
            
        except Exception as e:
            logger.error(f"Error analyzing query terms: {e}")
            return []