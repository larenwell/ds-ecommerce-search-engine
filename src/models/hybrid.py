"""
Hybrid Retriever combining multiple retrieval strategies.

Combines TF-IDF, BM25, and Semantic search using weighted score fusion.
This leverages the strengths of each approach:
- TF-IDF/BM25: Fast, good for exact keyword matching
- Semantic: Captures meaning, handles synonyms and paraphrases

Score fusion strategy: Weighted average of normalized scores
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseRetriever
from .tfidf import TFIDFRetriever
from .bm25 import BM25Retriever
from ..config import config

# Conditional import for semantic retriever
try:
    from .semantic import SemanticRetriever
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning("SemanticRetriever not available (missing sentence-transformers)")


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining multiple retrieval methods.
    
    Strategy:
    1. Get scored results from each enabled retriever (top-K candidates)
    2. Normalize scores to [0, 1] range using min-max scaling
    3. Combine using weighted average of normalized scores
    4. Re-rank by combined scores and return top-k
    
    Benefits:
    - Leverages strengths of different approaches
    - More robust than any single method
    - Can be tuned via weights for specific use cases
    - Handles cases where individual methods fail
    
    Trade-offs:
    - Slower than individual methods (runs multiple retrievers)
    - More complex to tune (multiple hyperparameters)
    - Requires more memory
    
    Example:
        >>> hybrid = HybridRetriever(
        ...     use_tfidf=True,
        ...     use_bm25=True,
        ...     use_semantic=False,  # Skip if slow
        ...     tfidf_weight=0.4,
        ...     bm25_weight=0.6
        ... )
        >>> hybrid.fit(product_df)
        >>> results = hybrid.retrieve("comfortable armchair")
    """
    
    def __init__(
        self,
        top_k: int = 10,
        use_tfidf: bool = True,
        use_bm25: bool = True,
        use_semantic: bool = False,
        tfidf_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        retrieval_k: Optional[int] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            top_k: Number of final results to return
            use_tfidf: Whether to use TF-IDF retriever
            use_bm25: Whether to use BM25 retriever
            use_semantic: Whether to use semantic retriever
            tfidf_weight: Weight for TF-IDF scores (None = use config)
            bm25_weight: Weight for BM25 scores (None = use config)
            semantic_weight: Weight for semantic scores (None = use config)
            retrieval_k: Number of candidates to retrieve from each model
                        (None = use config default, typically 50)
        """
        super().__init__(top_k)
        
        self.use_tfidf = use_tfidf
        self.use_bm25 = use_bm25
        self.use_semantic = use_semantic and SEMANTIC_AVAILABLE
        
        if use_semantic and not SEMANTIC_AVAILABLE:
            logger.warning(
                "Semantic search requested but not available. "
                "Install sentence-transformers to enable."
            )
        
        # Use config defaults if not provided
        self.tfidf_weight = tfidf_weight if tfidf_weight is not None else config.retriever.hybrid_tfidf_weight
        self.bm25_weight = bm25_weight if bm25_weight is not None else config.retriever.hybrid_bm25_weight
        self.semantic_weight = semantic_weight if semantic_weight is not None else config.retriever.hybrid_semantic_weight
        
        # Normalize weights to sum to 1
        self._normalize_weights()
        
        # Number of candidates to retrieve from each model
        self.retrieval_k = retrieval_k if retrieval_k is not None else config.retriever.hybrid_retrieval_k
        
        # Initialize retrievers
        self.retrievers = {}
        if self.use_tfidf:
            self.retrievers['tfidf'] = TFIDFRetriever(top_k=self.retrieval_k)
        if self.use_bm25:
            self.retrievers['bm25'] = BM25Retriever(top_k=self.retrieval_k)
        if self.use_semantic:
            self.retrievers['semantic'] = SemanticRetriever(top_k=self.retrieval_k)
        
        if not self.retrievers:
            raise ValueError("At least one retriever must be enabled")
        
        logger.info(
            f"HybridRetriever initialized with {len(self.retrievers)} models: "
            f"{list(self.retrievers.keys())}"
        )
        logger.info(
            f"Weights: TF-IDF={self.tfidf_weight:.2f}, "
            f"BM25={self.bm25_weight:.2f}, "
            f"Semantic={self.semantic_weight:.2f}"
        )
    
    def _normalize_weights(self) -> None:
        """
        Normalize weights to sum to 1.
        
        Sets weights to 0 for disabled retrievers and rescales
        active weights to sum to 1.0.
        """
        active_weights = []
        
        if self.use_tfidf:
            active_weights.append(self.tfidf_weight)
        else:
            self.tfidf_weight = 0.0
            
        if self.use_bm25:
            active_weights.append(self.bm25_weight)
        else:
            self.bm25_weight = 0.0
            
        if self.use_semantic:
            active_weights.append(self.semantic_weight)
        else:
            self.semantic_weight = 0.0
        
        total = sum(active_weights)
        if total > 0:
            factor = 1.0 / total
            self.tfidf_weight *= factor
            self.bm25_weight *= factor
            self.semantic_weight *= factor
        
        logger.debug(
            f"Normalized weights: TF-IDF={self.tfidf_weight:.3f}, "
            f"BM25={self.bm25_weight:.3f}, Semantic={self.semantic_weight:.3f}"
        )
    
    def fit(self, product_df: pd.DataFrame) -> 'HybridRetriever':
        """
        Fit all component retrievers.
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info(
                f"Fitting hybrid retriever with {len(self.retrievers)} models "
                f"on {len(product_df)} products"
            )
            
            # Validate input
            self._validate_product_df(product_df)
            
            # Store product dataframe
            self.product_df = product_df.copy()
            
            # Fit each retriever
            for name, retriever in self.retrievers.items():
                logger.info(f"Fitting {name} retriever...")
                retriever.fit(product_df)
                logger.info(f"{name} retriever fitted")
            
            self.is_fitted = True
            
            logger.info("Hybrid retriever fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting hybrid retriever: {e}")
            raise
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max scaling.
        
        Args:
            scores: Array of scores to normalize
            
        Returns:
            Normalized scores in [0, 1] range
        """
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        # Handle case where all scores are the same
        if max_score == min_score:
            return np.ones_like(scores)
        
        # Min-max normalization
        normalized = (scores - min_score) / (max_score - min_score)
        
        return normalized
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve using hybrid approach.
        
        Args:
            query: Search query string
            top_k: Number of final results (overrides self.top_k if provided)
            
        Returns:
            List of product IDs ranked by combined score
        """
        results_with_scores = self.retrieve_with_scores(query, top_k)
        return [pid for pid, _ in results_with_scores]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve with hybrid scoring.
        
        Process:
        1. Get scored results from each retriever
        2. Normalize each retriever's scores to [0, 1]
        3. Combine scores using weighted average
        4. Sort by combined score and return top-k
        
        Args:
            query: Search query string
            top_k: Number of results (overrides self.top_k if provided)
            
        Returns:
            List of (product_id, combined_score) tuples sorted by score
        """
        self._validate_fitted()
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Dictionary to accumulate scores: product_id -> combined_score
            all_scores: Dict[int, float] = {}
            
            # Collect results from each retriever
            for name, retriever in self.retrievers.items():
                logger.debug(f"Retrieving from {name}...")
                
                # Get scored results
                results = retriever.retrieve_with_scores(query, self.retrieval_k)
                
                if not results:
                    logger.warning(f"No results from {name} for query: '{query}'")
                    continue
                
                # Extract product IDs and scores
                product_ids = np.array([pid for pid, _ in results])
                scores = np.array([score for _, score in results])
                
                # Normalize scores to [0, 1]
                normalized_scores = self._normalize_scores(scores)
                
                # Get weight for this retriever
                weight = getattr(self, f"{name}_weight")
                
                logger.debug(
                    f"{name}: {len(results)} results, "
                    f"weight={weight:.3f}, "
                    f"score_range=[{scores.min():.3f}, {scores.max():.3f}]"
                )
                
                # Add weighted scores to accumulator
                for pid, norm_score in zip(product_ids, normalized_scores):
                    if pid not in all_scores:
                        all_scores[pid] = 0.0
                    all_scores[pid] += weight * norm_score
            
            if not all_scores:
                logger.warning(f"No results found for query: '{query}'")
                return []
            
            # Sort by combined score (descending)
            sorted_results = sorted(
                all_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Return top-k
            top_results = sorted_results[:k]
            
            logger.debug(
                f"Hybrid retrieval: {len(all_scores)} unique products, "
                f"returning top {len(top_results)}, "
                f"score_range=[{top_results[-1][1]:.3f}, {top_results[0][1]:.3f}]"
            )
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval for query '{query}': {e}")
            return []
    
    def get_retriever_contributions(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get individual contributions from each retriever for analysis.
        
        Useful for understanding which retriever contributes what to
        the final ranking.
        
        Args:
            query: Search query string
            top_k: Number of results per retriever
            
        Returns:
            Dictionary mapping retriever name to list of (product_id, score) tuples
        """
        self._validate_fitted()
        
        k = top_k if top_k is not None else self.top_k
        
        contributions = {}
        
        for name, retriever in self.retrievers.items():
            try:
                results = retriever.retrieve_with_scores(query, k)
                contributions[name] = results
            except Exception as e:
                logger.error(f"Error getting contribution from {name}: {e}")
                contributions[name] = []
        
        return contributions