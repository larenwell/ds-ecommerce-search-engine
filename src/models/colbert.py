"""
ColBERT Retriever - Contextualized Late Interaction over BERT.

ColBERT uses token-level embeddings and late interaction (MaxSim) for
more precise matching than traditional dense retrieval.

Reference: Khattab & Zaharia (2020) "ColBERT: Efficient and Effective 
Passage Search via Contextualized Late Interaction over BERT"
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger
import torch

try:
    from transformers import BertModel, BertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")

from .base import BaseRetriever
from ..config import config


class ColBERTRetriever(BaseRetriever):
    """
    ColBERT: Contextualized Late Interaction over BERT.
    
    Key innovation: Instead of single vector per document,
    ColBERT creates embeddings for each token, then uses
    MaxSim operation for late interaction scoring.
    
    Architecture:
    1. Query: BERT → token embeddings [CLS, tok1, tok2, ...]
    2. Document: BERT → token embeddings [CLS, tok1, tok2, ...]
    3. Score: MaxSim between query and document tokens
    
    MaxSim Formula:
        Score(Q, D) = Σ max(cosine_sim(q_i, d_j)) for all q_i
    
    Advantages:
    - More precise than bi-encoder (token-level matching)
    - More efficient than cross-encoder (pre-compute doc embeddings)
    - State-of-the-art for dense retrieval
    - Handles multi-token phrases well
    
    Trade-offs:
    - Slower than TF-IDF/BM25
    - Requires more memory (token embeddings for all docs)
    - More complex to implement
    - Best with GPU
    
    Example:
        >>> retriever = ColBERTRetriever(
        ...     model_name='bert-base-uncased',
        ...     dim=128  # Compression dimension
        ... )
        >>> retriever.fit(product_df)
        >>> results = retriever.retrieve("comfortable sofa")
    """
    
    def __init__(
        self,
        top_k: int = 10,
        model_name: str = 'bert-base-uncased',
        dim: int = 128,
        max_length: int = 256,
        device: str = None,
        text_fields: Optional[List[str]] = None
    ):
        """
        Initialize ColBERT retriever.
        
        Args:
            top_k: Number of results to return
            model_name: BERT model name from HuggingFace
            dim: Compression dimension (original ColBERT uses 128)
            max_length: Maximum sequence length
            device: Device ('cuda', 'cpu', or None for auto)
            text_fields: Product fields to use
        """
        super().__init__(top_k)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers required. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.dim = dim
        self.max_length = max_length
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.text_fields = text_fields or [
            'product_name',
            'product_description'
        ]
        
        # Load BERT model and tokenizer
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(self.device)
        self.bert.eval()
        
        # Compression layer (768 → dim)
        self.compression = torch.nn.Linear(768, dim).to(self.device)
        
        # Storage for document embeddings
        self.doc_embeddings = {}
        self.doc_masks = {}
        
        logger.info(
            f"ColBERTRetriever initialized: "
            f"model={model_name}, dim={dim}, device={self.device}"
        )
    
    def _encode_text(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode texts to token-level embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            (embeddings, masks) tensors
        """
        all_embeddings = []
        all_masks = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get BERT embeddings
                outputs = self.bert(**encoded)
                token_embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]
                
                # Compress to lower dimension
                compressed = self.compression(token_embeddings)  # [batch, seq_len, dim]
                
                # L2 normalize
                compressed = torch.nn.functional.normalize(compressed, p=2, dim=2)
                
                all_embeddings.append(compressed.cpu())
                all_masks.append(encoded['attention_mask'].cpu())
        
        return torch.cat(all_embeddings), torch.cat(all_masks)
    
    def fit(self, product_df: pd.DataFrame) -> 'ColBERTRetriever':
        """
        Fit retriever by encoding all products.
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting ColBERT retriever...")
        
        self._validate_product_df(product_df)
        self.product_df = product_df.copy()
        
        # Prepare product texts
        available_fields = [
            f for f in self.text_fields if f in product_df.columns
        ]
        
        logger.info(f"Using fields: {available_fields}")
        combined_texts = self._prepare_text_field(
            product_df,
            available_fields,
            separator=" "
        )
        
        # Encode all products
        logger.info(f"Encoding {len(product_df)} products (this may take a while)...")
        
        product_ids = product_df['product_id'].tolist()
        texts = combined_texts.tolist()
        
        # Encode in batches
        embeddings, masks = self._encode_text(texts, batch_size=8)
        
        # Store embeddings and masks per product
        for i, product_id in enumerate(product_ids):
            self.doc_embeddings[product_id] = embeddings[i]
            self.doc_masks[product_id] = masks[i]
        
        self.is_fitted = True
        
        logger.info(
            f"ColBERT fitted: {len(self.doc_embeddings)} products encoded, "
            f"embedding dim={self.dim}"
        )
        
        return self
    
    def _maxsim_score(
        self,
        query_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_mask: torch.Tensor
    ) -> float:
        """
        Compute MaxSim score between query and document.
        
        MaxSim: For each query token, find max similarity with any doc token,
        then sum over query tokens.
        
        Args:
            query_embeddings: Query token embeddings [seq_len_q, dim]
            query_mask: Query attention mask [seq_len_q]
            doc_embeddings: Document token embeddings [seq_len_d, dim]
            doc_mask: Document attention mask [seq_len_d]
            
        Returns:
            MaxSim score
        """
        # Get valid tokens (exclude padding)
        query_valid = query_mask.bool()
        doc_valid = doc_mask.bool()
        
        query_tokens = query_embeddings[query_valid]  # [n_q, dim]
        doc_tokens = doc_embeddings[doc_valid]  # [n_d, dim]
        
        if len(query_tokens) == 0 or len(doc_tokens) == 0:
            return 0.0
        
        # Compute similarity matrix [n_q, n_d]
        sim_matrix = torch.matmul(query_tokens, doc_tokens.T)
        
        # MaxSim: for each query token, max similarity with doc
        max_sims = sim_matrix.max(dim=1)[0]  # [n_q]
        
        # Sum over query tokens
        score = max_sims.sum().item()
        
        return score
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve products using ColBERT scoring.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of product IDs ranked by MaxSim score
        """
        results = self.retrieve_with_scores(query, top_k)
        return [pid for pid, _ in results]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve with MaxSim scores.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (product_id, score) tuples
        """
        self._validate_fitted()
        
        k = top_k or self.top_k
        
        try:
            # Encode query
            logger.debug("Encoding query...")
            query_embeddings, query_mask = self._encode_text([query], batch_size=1)
            query_embeddings = query_embeddings[0]  # [seq_len, dim]
            query_mask = query_mask[0]  # [seq_len]
            
            # Score all documents
            logger.debug(f"Scoring {len(self.doc_embeddings)} products...")
            scores = []
            
            for product_id in self.product_df['product_id']:
                if product_id not in self.doc_embeddings:
                    continue
                
                doc_embeddings = self.doc_embeddings[product_id]
                doc_mask = self.doc_masks[product_id]
                
                score = self._maxsim_score(
                    query_embeddings,
                    query_mask,
                    doc_embeddings,
                    doc_mask
                )
                
                scores.append((product_id, score))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(
                f"ColBERT scoring complete: "
                f"top score={scores[0][1]:.4f}"
            )
            
            return scores[:k]
            
        except Exception as e:
            logger.error(f"Error in ColBERT retrieval: {e}")
            return []
    
    def save_index(self, path: str) -> None:
        """
        Save pre-computed embeddings to disk.
        
        Args:
            path: Path to save index
        """
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'doc_embeddings': self.doc_embeddings,
            'doc_masks': self.doc_masks,
            'product_df': self.product_df
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"ColBERT index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """
        Load pre-computed embeddings from disk.
        
        Args:
            path: Path to load index from
        """
        import pickle
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.doc_embeddings = index_data['doc_embeddings']
        self.doc_masks = index_data['doc_masks']
        self.product_df = index_data['product_df']
        self.is_fitted = True
        
        logger.info(f"ColBERT index loaded from {path}")