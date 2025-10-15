"""
Dense Passage Retrieval (DPR) - Facebook Research.

DPR uses two separate BERT encoders (bi-encoder architecture):
- Question encoder for queries
- Context encoder for documents

Trained with contrastive learning on positive/negative pairs.

Reference: Karpukhin et al. (2020) "Dense Passage Retrieval for 
Open-Domain Question Answering"
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import torch

try:
    from transformers import DPRQuestionEncoder, DPRContextEncoder
    from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available (install for faster search)")

from .base import BaseRetriever
from ..config import config


class DPRRetriever(BaseRetriever):
    """
    Dense Passage Retrieval using dual encoders.
    
    Architecture:
    - Query Encoder: Maps queries to dense vectors
    - Context Encoder: Maps documents to dense vectors
    - Scoring: Dot product similarity
    
    Key advantage: Pre-compute all document embeddings once,
    then only encode query at inference time.
    
    Advantages:
    - Efficient at inference (only encode query)
    - Good semantic understanding
    - Scalable with FAISS indexing
    - State-of-the-art for open-domain QA
    
    Trade-offs:
    - Requires large training data for fine-tuning
    - Two separate models to maintain
    - Pre-trained models may not transfer well to e-commerce
    - Memory intensive (stores all embeddings)
    
    Performance:
    - With pre-trained models: Similar to SemanticRetriever
    - With fine-tuning: Can significantly improve
    
    Example:
        >>> retriever = DPRRetriever(
        ...     question_encoder='facebook/dpr-question_encoder-single-nq-base',
        ...     ctx_encoder='facebook/dpr-ctx_encoder-single-nq-base'
        ... )
        >>> retriever.fit(product_df)
        >>> results = retriever.retrieve("comfortable chair")
    """
    
    def __init__(
        self,
        top_k: int = 10,
        question_encoder: str = 'facebook/dpr-question_encoder-single-nq-base',
        ctx_encoder: str = 'facebook/dpr-ctx_encoder-single-nq-base',
        batch_size: int = 32,
        use_faiss: bool = True,
        device: str = None,
        text_fields: Optional[List[str]] = None
    ):
        """
        Initialize DPR retriever.
        
        Args:
            top_k: Number of results to return
            question_encoder: Question encoder model name
            ctx_encoder: Context encoder model name
            batch_size: Batch size for encoding
            use_faiss: Whether to use FAISS for fast search
            device: Device ('cuda', 'cpu', or None for auto)
            text_fields: Product fields to use
        """
        super().__init__(top_k)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers required. "
                "Install with: pip install transformers torch"
            )
        
        self.batch_size = batch_size
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.text_fields = text_fields or [
            'product_name',
            'product_description'
        ]
        
        # Load question encoder
        logger.info(f"Loading question encoder: {question_encoder}")
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            question_encoder
        ).to(self.device)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            question_encoder
        )
        self.question_encoder.eval()
        
        # Load context encoder
        logger.info(f"Loading context encoder: {ctx_encoder}")
        self.ctx_encoder = DPRContextEncoder.from_pretrained(
            ctx_encoder
        ).to(self.device)
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            ctx_encoder
        )
        self.ctx_encoder.eval()
        
        # Storage
        self.doc_embeddings = None
        self.product_ids = None
        self.faiss_index = None
        
        if self.use_faiss and not FAISS_AVAILABLE:
            logger.warning(
                "FAISS not available, falling back to numpy. "
                "Install with: pip install faiss-cpu"
            )
            self.use_faiss = False
        
        logger.info(
            f"DPRRetriever initialized: "
            f"device={self.device}, use_faiss={self.use_faiss}"
        )
    
    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries to dense vectors.
        
        Args:
            queries: List of queries
            
        Returns:
            Query embeddings [n_queries, dim]
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(queries), self.batch_size):
                batch = queries[i:i + self.batch_size]
                
                # Tokenize
                encoded = self.question_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                ).to(self.device)
                
                # Encode
                outputs = self.question_encoder(**encoded)
                embeddings = outputs.pooler_output  # [batch, 768]
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _encode_contexts(self, contexts: List[str]) -> np.ndarray:
        """
        Encode contexts/documents to dense vectors.
        
        Args:
            contexts: List of context texts
            
        Returns:
            Context embeddings [n_contexts, dim]
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(contexts), self.batch_size):
                batch = contexts[i:i + self.batch_size]
                
                # Tokenize
                encoded = self.ctx_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                ).to(self.device)
                
                # Encode
                outputs = self.ctx_encoder(**encoded)
                embeddings = outputs.pooler_output  # [batch, 768]
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                if (i + self.batch_size) % 1000 == 0:
                    logger.debug(f"Encoded {i + self.batch_size} contexts")
        
        return np.vstack(all_embeddings)
    
    def fit(self, product_df: pd.DataFrame) -> 'DPRRetriever':
        """
        Fit retriever by encoding all products.
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting DPR retriever...")
        
        self._validate_product_df(product_df)
        self.product_df = product_df.copy()
        
        # Prepare texts
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
        logger.info(f"Encoding {len(product_df)} products...")
        self.doc_embeddings = self._encode_contexts(combined_texts.tolist())
        self.product_ids = product_df['product_id'].values
        
        # Build FAISS index if available
        if self.use_faiss:
            logger.info("Building FAISS index...")
            dim = self.doc_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product
            
            # Normalize for cosine similarity
            faiss.normalize_L2(self.doc_embeddings)
            self.faiss_index.add(self.doc_embeddings)
            
            logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors")
        
        self.is_fitted = True
        
        logger.info(
            f"DPR fitted: {len(self.doc_embeddings)} products, "
            f"embedding dim={self.doc_embeddings.shape[1]}"
        )
        
        return self
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve products using DPR.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of product IDs
        """
        results = self.retrieve_with_scores(query, top_k)
        return [pid for pid, _ in results]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve with similarity scores.
        
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
            query_embedding = self._encode_queries([query])[0]  # [dim]
            
            if self.use_faiss:
                # FAISS search
                query_embedding = query_embedding.reshape(1, -1)
                faiss.normalize_L2(query_embedding)
                
                scores, indices = self.faiss_index.search(query_embedding, k)
                scores = scores[0]
                indices = indices[0]
                
                product_ids = self.product_ids[indices]
                
            else:
                # Numpy search
                # Normalize
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                doc_norms = self.doc_embeddings / np.linalg.norm(
                    self.doc_embeddings,
                    axis=1,
                    keepdims=True
                )
                
                # Cosine similarity
                scores = np.dot(doc_norms, query_norm)
                
                # Top-k
                top_indices = np.argsort(scores)[-k:][::-1]
                product_ids = self.product_ids[top_indices]
                scores = scores[top_indices]
            
            results = list(zip(product_ids, scores))
            
            logger.debug(
                f"DPR retrieval: top score={results[0][1]:.4f}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in DPR retrieval: {e}")
            return []
    
    def save_index(self, path: str) -> None:
        """Save embeddings and FAISS index to disk."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'doc_embeddings': self.doc_embeddings,
            'product_ids': self.product_ids,
            'product_df': self.product_df
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        if self.use_faiss:
            faiss_path = path.with_suffix('.faiss')
            faiss.write_index(self.faiss_index, str(faiss_path))
            logger.info(f"FAISS index saved to {faiss_path}")
        
        logger.info(f"DPR index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """Load embeddings and FAISS index from disk."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.doc_embeddings = data['doc_embeddings']
        self.product_ids = data['product_ids']
        self.product_df = data['product_df']
        
        if self.use_faiss:
            faiss_path = path.with_suffix('.faiss')
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"FAISS index loaded from {faiss_path}")
        
        self.is_fitted = True
        logger.info(f"DPR index loaded from {path}")