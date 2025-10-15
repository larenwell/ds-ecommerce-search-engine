"""
Semantic Search using Sentence Transformers.

Uses pre-trained language models to create dense embeddings that capture
semantic meaning. This allows matching queries like "comfortable sofa" with
products described as "cozy couch" even without lexical overlap.

Model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
Alternative models:
- all-mpnet-base-v2: Better quality, slower (768 dim)
- paraphrase-MiniLM-L6-v2: Alternative lightweight model
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import torch

from .base import BaseRetriever
from ..config import config


class SemanticRetriever(BaseRetriever):
    """
    Semantic search using sentence transformers.
    
    Advantages:
    1. Captures semantic similarity beyond keyword matching
    2. Handles synonyms and paraphrases naturally
    3. Works well with natural language queries
    4. Can find relevant products even without exact word matches
    
    Trade-offs:
    - Slower than TF-IDF/BM25 (requires neural network inference)
    - Requires more memory for embeddings
    - May not always respect exact keyword matches
    """
    
    def __init__(
        self,
        top_k: int = 10,
        model_name: str = None,
        batch_size: int = 32,
        text_fields: List[str] = None,
        device: str = None
    ):
        """
        Initialize semantic retriever.
        
        Args:
            top_k: Number of top results to retrieve
            model_name: Name of sentence transformer model
            batch_size: Batch size for encoding
            text_fields: List of product fields to use
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        super().__init__(top_k)
        
        self.model_name = model_name or config.retriever.semantic_model_name
        self.batch_size = batch_size or config.retriever.semantic_batch_size
        
        self.text_fields = text_fields or [
            'product_name',
            'product_description'
        ]
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing semantic model: {self.model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
        
        self.product_embeddings = None
    
    def fit(self, product_df: pd.DataFrame) -> 'SemanticRetriever':
        """
        Encode all products into embeddings.
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info(f"Fitting semantic retriever on {len(product_df)} products")
            
            self.product_df = product_df.copy()
            
            # Combine text fields
            available_fields = [f for f in self.text_fields if f in product_df.columns]
            
            if not available_fields:
                raise ValueError(
                    f"None of the specified text fields {self.text_fields} "
                    f"found in product dataframe"
                )
            
            logger.info(f"Using fields: {available_fields}")
            combined_text = self._prepare_text_field(product_df, available_fields)
            
            # Encode products in batches
            logger.info("Encoding products... (this may take a while)")
            self.product_embeddings = self.model.encode(
                combined_text.tolist(),
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.is_fitted = True
            
            logger.info(
                f"Semantic retriever fitted: embeddings shape={self.product_embeddings.shape}"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting semantic retriever: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve top-k products using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of results
            
        Returns:
            List of product IDs ranked by semantic similarity
        """
        self._validate_fitted()
        
        k = top_k or self.top_k
        
        try:
            # Encode query
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True
            )
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                query_embedding,
                self.product_embeddings
            ).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            # Get product IDs
            product_ids = self.product_df.iloc[top_indices]['product_id'].tolist()
            
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
        Retrieve top-k products with their semantic similarity scores.
        
        Args:
            query: Search query string
            top_k: Number of results
            
        Returns:
            List of (product_id, score) tuples
        """
        self._validate_fitted()
        
        k = top_k or self.top_k
        
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(
                query_embedding,
                self.product_embeddings
            ).flatten()
            
            top_indices = np.argsort(similarities)[-k:][::-1]
            top_scores = similarities[top_indices]
            
            product_ids = self.product_df.iloc[top_indices]['product_id'].tolist()
            
            return list(zip(product_ids, top_scores))
            
        except Exception as e:
            logger.error(f"Error retrieving results with scores: {e}")
            return []