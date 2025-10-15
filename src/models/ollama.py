"""
Ollama LLM-based Retriever for Advanced Search.

Uses local LLMs via Ollama for:
1. Re-ranking: LLM scores BM25 candidates
2. Query expansion: LLM expands queries with synonyms
3. Direct generation: LLM generates product recommendations

Requires:
- Ollama installed and running
- Model pulled (e.g., llama3, mistral, phi3)
- ollama Python package
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger
import json
import re

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not available")

from .base import BaseRetriever
from .bm25 import BM25Retriever
from ..config import config


class OllamaRetriever(BaseRetriever):
    """
    LLM-based retriever using Ollama for advanced search strategies.
    
    Strategies:
    1. "rerank": Use LLM to re-rank BM25 candidates
    2. "expand": Use LLM to expand queries with synonyms
    3. "generate": Use LLM to directly generate product recommendations
    
    Advantages:
    - Leverages powerful LLM reasoning
    - No training required
    - Can understand complex queries
    - Handles semantic understanding
    - Works with any Ollama model
    
    Trade-offs:
    - Requires Ollama server running
    - Slower than traditional methods
    - Depends on model quality
    - Higher resource usage
    
    Performance:
    - With good prompts: Can achieve MAP@10 > 0.45
    - Best for complex, semantic queries
    - Re-ranking strategy is most effective
    
    Example:
        >>> retriever = OllamaRetriever(
        ...     model="llama3",
        ...     strategy="rerank",
        ...     candidate_k=50
        ... )
        >>> retriever.fit(product_df)
        >>> results = retriever.retrieve("comfortable office chair for long hours")
    """
    
    def __init__(
        self,
        top_k: int = 10,
        model: str = "llama3",
        strategy: str = "rerank",
        candidate_k: int = 50,
        base_retriever: Optional[BaseRetriever] = None,
        text_fields: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize Ollama retriever.
        
        Args:
            top_k: Number of final results
            model: Ollama model name (llama3, mistral, phi3, gemma)
            strategy: Strategy to use ('rerank', 'expand', 'generate')
            candidate_k: Number of candidates for re-ranking
            base_retriever: Base retriever for candidates (rerank strategy)
            text_fields: Product fields to use
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum tokens for LLM response
        """
        super().__init__(top_k)
        
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama package required. "
                "Install with: pip install ollama"
            )
        
        self.model = model
        self.strategy = strategy
        self.candidate_k = candidate_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.text_fields = text_fields or [
            'product_name',
            'product_description'
        ]
        
        # Base retriever for candidates (rerank strategy)
        if strategy == "rerank":
            self.base_retriever = base_retriever or BM25Retriever(top_k=candidate_k)
        else:
            self.base_retriever = None
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        logger.info(
            f"OllamaRetriever initialized: "
            f"model={model}, strategy={strategy}, candidate_k={candidate_k}"
        )
    
    def _test_ollama_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            # Test with a simple prompt
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                options={"temperature": 0.1, "num_predict": 10}
            )
            logger.info(f"Ollama connection successful with model: {self.model}")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise
    
    def fit(self, product_df: pd.DataFrame) -> 'OllamaRetriever':
        """
        Fit retriever (mainly fits base retriever for rerank strategy).
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Ollama retriever...")
        
        self._validate_product_df(product_df)
        self.product_df = product_df.copy()
        
        # Fit base retriever if using rerank strategy
        if self.strategy == "rerank" and self.base_retriever:
            logger.info("Fitting base retriever for candidate generation...")
            self.base_retriever.fit(product_df)
        
        # Prepare product texts for efficient lookup
        self._prepare_product_texts()
        
        self.is_fitted = True
        logger.info("Ollama retriever fitted successfully")
        
        return self
    
    def _prepare_product_texts(self) -> None:
        """Prepare product texts for efficient retrieval."""
        self.product_texts = {}
        
        available_fields = [
            f for f in self.text_fields if f in self.product_df.columns
        ]
        
        logger.info(f"Preparing texts from fields: {available_fields}")
        
        for _, row in self.product_df.iterrows():
            product_id = row['product_id']
            
            # Combine fields
            text_parts = [
                str(row[field]) for field in available_fields
                if pd.notna(row[field])
            ]
            text = " ".join(text_parts)
            
            # Truncate if too long
            if len(text) > 2000:
                text = text[:2000]
            
            self.product_texts[product_id] = text
        
        logger.debug(f"Prepared {len(self.product_texts)} product texts")
    
    def _create_rerank_prompt(
        self,
        query: str,
        candidates: List[Tuple[int, str]]
    ) -> str:
        """Create prompt for re-ranking candidates."""
        prompt = f"""You are an expert e-commerce search assistant. Given a search query and a list of product candidates, rank them by relevance.

Search Query: "{query}"

Product Candidates:
"""
        
        for i, (product_id, product_text) in enumerate(candidates, 1):
            prompt += f"{i}. Product ID {product_id}: {product_text[:300]}...\n"
        
        prompt += f"""
Please rank these {len(candidates)} products by relevance to the query. Return ONLY a JSON list of product IDs in order of relevance (most relevant first).

Example format: [12345, 67890, 11111, ...]

JSON Response:"""
        
        return prompt
    
    def _create_expand_prompt(self, query: str) -> str:
        """Create prompt for query expansion."""
        prompt = f"""You are an expert e-commerce search assistant. Given a search query, expand it with relevant synonyms and related terms to improve search results.

Original Query: "{query}"

Please provide 3-5 expanded versions of this query that would help find relevant products. Each expansion should:
- Use different words/phrases with similar meaning
- Include specific product attributes when relevant
- Maintain the original intent

Return ONLY a JSON list of expanded queries.

Example format: ["comfortable office chair", "ergonomic work seat", "desk chair for long hours", "executive office seating"]

JSON Response:"""
        
        return prompt
    
    def _create_generate_prompt(self, query: str) -> str:
        """Create prompt for direct product generation."""
        prompt = f"""You are an expert e-commerce search assistant. Given a search query, recommend the most relevant products from our catalog.

Search Query: "{query}"

Available Products (first 20):
"""
        
        # Show first 20 products as context
        for i, (product_id, product_text) in enumerate(list(self.product_texts.items())[:20], 1):
            prompt += f"{i}. Product ID {product_id}: {product_text[:200]}...\n"
        
        prompt += f"""
Based on the query and available products, recommend the most relevant products. Return ONLY a JSON list of product IDs in order of relevance.

Example format: [12345, 67890, 11111, ...]

JSON Response:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[int]:
        """Parse LLM response to extract product IDs."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\[[\d\s,\]]+\]', response)
            if json_match:
                json_str = json_match.group()
                product_ids = json.loads(json_str)
                return [int(pid) for pid in product_ids if isinstance(pid, (int, str)) and str(pid).isdigit()]
            
            # Fallback: extract numbers from response
            numbers = re.findall(r'\d+', response)
            return [int(num) for num in numbers[:self.top_k]]
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return []
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Retrieve products using LLM strategy.
        
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
        Retrieve with LLM-based scores.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (product_id, score) tuples
        """
        self._validate_fitted()
        
        k = top_k or self.top_k
        
        try:
            if self.strategy == "rerank":
                return self._retrieve_rerank(query, k)
            elif self.strategy == "expand":
                return self._retrieve_expand(query, k)
            elif self.strategy == "generate":
                return self._retrieve_generate(query, k)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
                
        except Exception as e:
            logger.error(f"Error in Ollama retrieval: {e}")
            return []
    
    def _retrieve_rerank(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Re-rank BM25 candidates using LLM."""
        logger.debug("Using re-ranking strategy...")
        
        # Get candidates from base retriever
        candidates = self.base_retriever.retrieve(query, self.candidate_k)
        
        if not candidates:
            return []
        
        # Prepare candidate texts
        candidate_texts = [
            (pid, self.product_texts.get(pid, ""))
            for pid in candidates
            if pid in self.product_texts
        ]
        
        if not candidate_texts:
            return []
        
        # Create prompt
        prompt = self._create_rerank_prompt(query, candidate_texts)
        
        # Get LLM response
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        
        # Parse response
        ranked_ids = self._parse_llm_response(response['message']['content'])
        
        # Map to scores (higher position = higher score)
        results = []
        for i, pid in enumerate(ranked_ids[:k]):
            if pid in self.product_texts:
                score = 1.0 - (i * 0.1)  # Decreasing scores
                results.append((pid, score))
        
        logger.debug(f"Re-ranking complete: {len(results)} products ranked")
        return results
    
    def _retrieve_expand(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Expand query and search with expanded terms."""
        logger.debug("Using query expansion strategy...")
        
        # Create expansion prompt
        prompt = self._create_expand_prompt(query)
        
        # Get LLM response
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        
        # Parse expanded queries
        try:
            expanded_queries = json.loads(response['message']['content'])
            if not isinstance(expanded_queries, list):
                expanded_queries = [query]
        except:
            expanded_queries = [query]
        
        # Search with each expanded query using BM25
        all_results = {}
        
        for expanded_query in expanded_queries:
            if self.base_retriever:
                candidates = self.base_retriever.retrieve(expanded_query, k)
                for i, pid in enumerate(candidates):
                    if pid in self.product_texts:
                        score = 1.0 - (i * 0.1)
                        if pid not in all_results or score > all_results[pid]:
                            all_results[pid] = score
        
        # Sort by score
        results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Query expansion complete: {len(results)} products found")
        return results[:k]
    
    def _retrieve_generate(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Generate product recommendations directly."""
        logger.debug("Using direct generation strategy...")
        
        # Create generation prompt
        prompt = self._create_generate_prompt(query)
        
        # Get LLM response
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        
        # Parse response
        recommended_ids = self._parse_llm_response(response['message']['content'])
        
        # Map to scores
        results = []
        for i, pid in enumerate(recommended_ids[:k]):
            if pid in self.product_texts:
                score = 1.0 - (i * 0.1)
                results.append((pid, score))
        
        logger.debug(f"Direct generation complete: {len(results)} products recommended")
        return results
    
    def explain_recommendation(
        self,
        query: str,
        product_id: int
    ) -> Dict:
        """
        Get LLM explanation for why a product was recommended.
        
        Args:
            query: Search query
            product_id: Product ID to explain
            
        Returns:
            Dictionary with explanation
        """
        if product_id not in self.product_texts:
            return {"error": "Product not found"}
        
        product_text = self.product_texts[product_id]
        
        prompt = f"""Explain why this product is relevant to the search query.

Search Query: "{query}"
Product: {product_text[:500]}...

Provide a brief explanation (1-2 sentences) of why this product matches the query.

Explanation:"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3,
                    "num_predict": 200
                }
            )
            
            return {
                "product_id": product_id,
                "query": query,
                "explanation": response['message']['content'].strip(),
                "strategy": self.strategy
            }
        except Exception as e:
            return {
                "product_id": product_id,
                "query": query,
                "error": f"Failed to generate explanation: {e}",
                "strategy": self.strategy
            }
