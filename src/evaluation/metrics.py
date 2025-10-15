"""
Evaluation metrics for retrieval performance.

Implements:
1. Standard MAP@K (Mean Average Precision)
2. Weighted MAP@K (considers partial matches)
3. NDCG@K (Normalized Discounted Cumulative Gain)
4. MRR (Mean Reciprocal Rank)
5. Precision@K and Recall@K
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger

from ..config import config
from ..utils import safe_divide


class MetricsCalculator:
    """
    Standard metrics calculator for retrieval evaluation.
    
    Treats labels as binary: Exact = relevant, others = irrelevant
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize metrics calculator.
        
        Args:
            k_values: List of K values to compute metrics for
        """
        self.k_values = k_values or config.evaluation.k_values
    
    @staticmethod
    def map_at_k(
        true_ids: List[int],
        predicted_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Mean Average Precision at K.
        
        This is the standard MAP@K metric used in information retrieval.
        It measures both precision and ranking quality.
        
        Args:
            true_ids: List of relevant product IDs
            predicted_ids: List of predicted product IDs (ranked)
            k: Number of top predictions to consider
            
        Returns:
            MAP@K score between 0 and 1
        """
        if not true_ids or not predicted_ids:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, pred_id in enumerate(predicted_ids[:k]):
            if pred_id in true_ids and pred_id not in predicted_ids[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return safe_divide(score, min(len(true_ids), k))
    
    @staticmethod
    def precision_at_k(
        true_ids: List[int],
        predicted_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Precision at K.
        
        Precision@K = (# relevant items in top-K) / K
        
        Args:
            true_ids: List of relevant product IDs
            predicted_ids: List of predicted product IDs
            k: Number of top predictions
            
        Returns:
            Precision@K score
        """
        if not predicted_ids or k == 0:
            return 0.0
        
        predicted_k = predicted_ids[:k]
        num_relevant = sum(1 for pid in predicted_k if pid in true_ids)
        
        return safe_divide(num_relevant, k)
    
    @staticmethod
    def recall_at_k(
        true_ids: List[int],
        predicted_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Recall at K.
        
        Recall@K = (# relevant items in top-K) / (total # relevant items)
        
        Args:
            true_ids: List of relevant product IDs
            predicted_ids: List of predicted product IDs
            k: Number of top predictions
            
        Returns:
            Recall@K score
        """
        if not true_ids:
            return 0.0
        
        predicted_k = predicted_ids[:k]
        num_relevant = sum(1 for pid in predicted_k if pid in true_ids)
        
        return safe_divide(num_relevant, len(true_ids))
    
    @staticmethod
    def mrr(true_ids: List[int], predicted_ids: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / (rank of first relevant item)
        
        Args:
            true_ids: List of relevant product IDs
            predicted_ids: List of predicted product IDs
            
        Returns:
            MRR score
        """
        if not true_ids or not predicted_ids:
            return 0.0
        
        for i, pred_id in enumerate(predicted_ids):
            if pred_id in true_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_query(
        self,
        true_ids: List[int],
        predicted_ids: List[int]
    ) -> Dict[str, float]:
        """
        Calculate all metrics for a single query.
        
        Args:
            true_ids: List of relevant product IDs
            predicted_ids: List of predicted product IDs
            
        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}
        
        # Calculate for each K value
        for k in self.k_values:
            metrics[f'map@{k}'] = self.map_at_k(true_ids, predicted_ids, k)
            metrics[f'precision@{k}'] = self.precision_at_k(true_ids, predicted_ids, k)
            metrics[f'recall@{k}'] = self.recall_at_k(true_ids, predicted_ids, k)
        
        # MRR is independent of K
        metrics['mrr'] = self.mrr(true_ids, predicted_ids)
        
        return metrics
    
    def evaluate_queries(
        self,
        query_df: pd.DataFrame,
        true_col: str = 'relevant_ids',
        pred_col: str = 'top_product_ids'
    ) -> Dict[str, float]:
        """
        Evaluate metrics across all queries.
        
        Args:
            query_df: DataFrame with true and predicted IDs
            true_col: Column name for true relevant IDs
            pred_col: Column name for predicted IDs
            
        Returns:
            Dictionary of averaged metrics
        """
        all_metrics = []
        
        for _, row in query_df.iterrows():
            true_ids = row[true_col]
            pred_ids = row[pred_col]
            
            # Skip if either is empty or invalid
            if not isinstance(true_ids, (list, np.ndarray)) or \
               not isinstance(pred_ids, (list, np.ndarray)):
                continue
            
            metrics = self.evaluate_query(list(true_ids), list(pred_ids))
            all_metrics.append(metrics)
        
        # Average across all queries
        if not all_metrics:
            logger.warning("No valid queries to evaluate")
            return {}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


class WeightedMetricsCalculator(MetricsCalculator):
    """
    Weighted metrics calculator that considers partial matches.
    
    Key improvement: Instead of treating partial matches as irrelevant,
    we assign them a partial weight (default 0.5).
    
    This provides a fairer assessment when:
    - Products are partially relevant (e.g., similar category but not exact match)
    - The distinction between Exact and Partial is subjective
    
    Justification:
    - In e-commerce, showing a "partial match" is better than irrelevant
    - User might still find partial matches useful
    - Gives credit to models that at least get "close" to the right product
    
    Trade-offs:
    - More complex to interpret than binary metrics
    - Requires careful tuning of weights
    - May inflate scores if weights are too generous
    """
    
    def __init__(
        self,
        k_values: List[int] = None,
        exact_weight: float = None,
        partial_weight: float = None,
        irrelevant_weight: float = None
    ):
        """
        Initialize weighted metrics calculator.
        
        Args:
            k_values: List of K values
            exact_weight: Weight for exact matches (default 1.0)
            partial_weight: Weight for partial matches (default 0.5)
            irrelevant_weight: Weight for irrelevant (default 0.0)
        """
        super().__init__(k_values)
        
        self.exact_weight = exact_weight if exact_weight is not None else config.evaluation.exact_weight
        self.partial_weight = partial_weight if partial_weight is not None else config.evaluation.partial_weight
        self.irrelevant_weight = irrelevant_weight if irrelevant_weight is not None else config.evaluation.irrelevant_weight
        
        logger.info(
            f"Weighted metrics - Exact: {self.exact_weight}, "
            f"Partial: {self.partial_weight}, Irrelevant: {self.irrelevant_weight}"
        )
    
    def weighted_map_at_k(
        self,
        relevance_dict: Dict[int, float],
        predicted_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate weighted MAP@K considering partial relevance.
        
        Instead of binary relevance, uses weighted relevance scores:
        - Exact match: weight = 1.0
        - Partial match: weight = 0.5
        - Irrelevant: weight = 0.0
        
        Args:
            relevance_dict: Dict mapping product_id to relevance score
            predicted_ids: List of predicted product IDs
            k: Number of top predictions
            
        Returns:
            Weighted MAP@K score
        """
        if not relevance_dict or not predicted_ids:
            return 0.0
        
        score = 0.0
        cumulative_relevance = 0.0
        
        for i, pred_id in enumerate(predicted_ids[:k]):
            relevance = relevance_dict.get(pred_id, self.irrelevant_weight)
            
            if relevance > 0:
                cumulative_relevance += relevance
                precision_at_i = cumulative_relevance / (i + 1.0)
                score += relevance * precision_at_i
        
        # Normalize by ideal score (all exact matches at top)
        max_relevance = sum(sorted(relevance_dict.values(), reverse=True)[:k])
        
        return safe_divide(score, max_relevance)
    
    def ndcg_at_k(
        self,
        relevance_dict: Dict[int, float],
        predicted_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG is a standard IR metric that naturally handles graded relevance.
        It discounts relevance by position (items lower in ranking contribute less).
        
        Formula: DCG@K = sum(rel_i / log2(i+1)) for i in 1..k
                 NDCG@K = DCG@K / IDCG@K
        
        where IDCG is the DCG of the ideal ranking.
        
        Args:
            relevance_dict: Dict mapping product_id to relevance score
            predicted_ids: List of predicted product IDs
            k: Number of top predictions
            
        Returns:
            NDCG@K score between 0 and 1
        """
        if not relevance_dict or not predicted_ids:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, pred_id in enumerate(predicted_ids[:k]):
            relevance = relevance_dict.get(pred_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG (sort by relevance)
        ideal_relevances = sorted(relevance_dict.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return safe_divide(dcg, idcg)
    
    def evaluate_query_weighted(
        self,
        relevance_dict: Dict[int, float],
        predicted_ids: List[int]
    ) -> Dict[str, float]:
        """
        Calculate weighted metrics for a single query.
        
        Args:
            relevance_dict: Dict of product_id to relevance score
            predicted_ids: List of predicted product IDs
            
        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}
        
        for k in self.k_values:
            metrics[f'weighted_map@{k}'] = self.weighted_map_at_k(
                relevance_dict, predicted_ids, k
            )
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(
                relevance_dict, predicted_ids, k
            )
        
        return metrics
    
    def evaluate_queries_weighted(
        self,
        label_df: pd.DataFrame,
        query_df: pd.DataFrame,
        pred_col: str = 'top_product_ids'
    ) -> Dict[str, float]:
        """
        Evaluate weighted metrics across all queries.
        
        Args:
            label_df: DataFrame with query_id, product_id, label
            query_df: DataFrame with query_id and predictions
            pred_col: Column name for predicted IDs
            
        Returns:
            Dictionary of averaged weighted metrics
        """
        # Create relevance mapping from labels
        label_mapping = config.evaluation.label_mapping
        
        # Group labels by query
        grouped_labels = label_df.groupby('query_id')
        
        all_metrics = []
        
        for _, row in query_df.iterrows():
            query_id = row['query_id']
            pred_ids = row[pred_col]
            
            if not isinstance(pred_ids, (list, np.ndarray)):
                continue
            
            # Get relevance dict for this query
            try:
                query_labels = grouped_labels.get_group(query_id)
                relevance_dict = {}
                
                for _, label_row in query_labels.iterrows():
                    product_id = label_row['product_id']
                    label = label_row['label']
                    relevance_dict[product_id] = label_mapping.get(label, 0.0)
                
                if relevance_dict:
                    metrics = self.evaluate_query_weighted(relevance_dict, list(pred_ids))
                    all_metrics.append(metrics)
                    
            except KeyError:
                # Query not in labels
                continue
        
        # Average across all queries
        if not all_metrics:
            logger.warning("No valid queries to evaluate with weighted metrics")
            return {}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics