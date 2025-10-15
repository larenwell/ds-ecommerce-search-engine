"""
Search Pipeline - Orchestrates the entire search and evaluation workflow.

This module provides a high-level interface for:
1. Loading data
2. Training retrieval models
3. Running searches
4. Evaluating performance
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from .models import BaseRetriever
from .evaluation import MetricsCalculator, WeightedMetricsCalculator
from .data_loader import DataLoader
from .config import config
from .results_manager import ResultsManager


class SearchPipeline:
    """
    End-to-end search pipeline.
    
    Handles:
    - Data loading and validation
    - Model training
    - Batch retrieval
    - Comprehensive evaluation
    - Results reporting
    """
    
    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        data_dir: Path = None,
        save_results: bool = True
    ):
        """
        Initialize search pipeline.
        
        Args:
            retriever: Retriever instance (if None, will need to call set_retriever)
            data_dir: Directory containing data files
            save_results: Whether to automatically save results
        """
        self.retriever = retriever
        self.data_dir = data_dir or config.data.data_dir
        self.save_results = save_results
        
        # Initialize data loader
        self.data_loader = DataLoader(self.data_dir)
        
        # DataFrames
        self.product_df = None
        self.query_df = None
        self.label_df = None
        
        # Evaluators
        self.metrics_calculator = MetricsCalculator()
        self.weighted_metrics_calculator = WeightedMetricsCalculator()
        
        # Results manager
        self.results_manager = ResultsManager() if save_results else None
        
        logger.info(f"SearchPipeline initialized with data_dir: {self.data_dir}")
    
    def set_retriever(self, retriever: BaseRetriever) -> 'SearchPipeline':
        """
        Set or replace the retriever.
        
        Args:
            retriever: Retriever instance
            
        Returns:
            Self for method chaining
        """
        self.retriever = retriever
        logger.info(f"Retriever set to: {retriever}")
        return self
    
    def load_data(
        self,
        query_file: str = None,
        product_file: str = None,
        label_file: str = None
    ) -> 'SearchPipeline':
        """
        Load all data files using DataLoader.
        
        Args:
            query_file: Query file name (uses config default if None)
            product_file: Product file name (uses config default if None)
            label_file: Label file name (uses config default if None)
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info("Loading data files...")
            
            # Update data loader paths if provided
            if query_file:
                config.data.query_file = query_file
            if product_file:
                config.data.product_file = product_file
            if label_file:
                config.data.label_file = label_file
            
            # Use data loader to load all files
            self.query_df, self.product_df, self.label_df = self.data_loader.load_all()
            
            logger.info("Data loaded successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def fit(self) -> 'SearchPipeline':
        """
        Fit the retriever on product data.
        
        Returns:
            Self for method chaining
        """
        if self.retriever is None:
            raise ValueError("No retriever set. Call set_retriever() first.")
        
        if self.product_df is None:
            raise ValueError("No product data loaded. Call load_data() first.")
        
        logger.info("Fitting retriever...")
        self.retriever.fit(self.product_df)
        logger.info("Retriever fitted successfully")
        
        return self
    
    def run_search(
        self,
        query: Optional[str] = None,
        query_id: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Run search for a single query.
        
        Args:
            query: Query string (provide either query or query_id)
            query_id: Query ID from query_df
            top_k: Number of results
            
        Returns:
            List of product IDs
        """
        if self.retriever is None or not self.retriever.is_fitted:
            raise ValueError("Retriever not fitted. Call fit() first.")
        
        # Get query string
        if query is None:
            if query_id is None:
                raise ValueError("Must provide either query or query_id")
            query = self.query_df[self.query_df['query_id'] == query_id]['query'].iloc[0]
        
        return self.retriever.retrieve(query, top_k)
    
    def run_batch_search(
        self,
        queries: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Run search for all queries and add results to query_df.
        
        Args:
            queries: List of queries (if None, uses all from query_df)
            top_k: Number of results per query
            show_progress: Whether to show progress bar
            
        Returns:
            Updated query_df with search results
        """
        if self.retriever is None or not self.retriever.is_fitted:
            raise ValueError("Retriever not fitted. Call fit() first.")
        
        if self.query_df is None:
            raise ValueError("No query data loaded. Call load_data() first.")
        
        logger.info("Running batch search...")
        
        # Use all queries if not provided
        if queries is None:
            queries = self.query_df['query'].tolist()
        
        # Run retrieval
        results = []
        iterator = tqdm(queries, desc="Searching") if show_progress else queries
        
        for query in iterator:
            try:
                product_ids = self.retriever.retrieve(query, top_k)
                results.append(product_ids)
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
                results.append([])
        
        # Add to dataframe
        self.query_df = self.query_df.copy()
        self.query_df['top_product_ids'] = results
        
        logger.info(f"Batch search completed for {len(queries)} queries")
        
        return self.query_df
    
    def prepare_ground_truth(self) -> pd.DataFrame:
        """
        Prepare ground truth data by grouping labels per query.
        
        Adds 'relevant_ids' column to query_df containing list of
        exact match product IDs for each query.
        
        Returns:
            Updated query_df with ground truth
        """
        if self.label_df is None:
            raise ValueError("No label data loaded. Call load_data() first.")
        
        logger.info("Preparing ground truth...")
        
        # Group labels by query and get exact matches
        grouped = self.label_df[self.label_df['label'] == 'Exact'].groupby('query_id')
        
        relevant_dict = {}
        for query_id, group in grouped:
            relevant_dict[query_id] = group['product_id'].tolist()
        
        # Add to query_df
        self.query_df = self.query_df.copy()
        self.query_df['relevant_ids'] = self.query_df['query_id'].map(relevant_dict)
        
        # Fill NaN with empty lists
        self.query_df['relevant_ids'] = self.query_df['relevant_ids'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        logger.info("Ground truth prepared")
        
        return self.query_df
    
    def evaluate(
        self,
        include_weighted: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            include_weighted: Whether to include weighted metrics
            
        Returns:
            Dictionary of all metrics
        """
        if 'relevant_ids' not in self.query_df.columns:
            self.prepare_ground_truth()
        
        if 'top_product_ids' not in self.query_df.columns:
            raise ValueError("No search results. Call run_batch_search() first.")
        
        logger.info("Evaluating retrieval performance...")
        
        # Standard metrics
        standard_metrics = self.metrics_calculator.evaluate_queries(self.query_df)
        
        results = {"standard": standard_metrics}
        
        # Weighted metrics
        if include_weighted:
            weighted_metrics = self.weighted_metrics_calculator.evaluate_queries_weighted(
                self.label_df,
                self.query_df
            )
            results["weighted"] = weighted_metrics
        
        logger.info("Evaluation completed")
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print evaluation results.
        
        Args:
            results: Dictionary of metrics from evaluate()
        """
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        if "standard" in results:
            print("\n📊 Standard Metrics (Binary: Exact=1, Others=0)")
            print("-" * 70)
            for metric, value in sorted(results["standard"].items()):
                print(f"  {metric:.<25} {value:.4f}")
        
        if "weighted" in results:
            print("\n⚖️  Weighted Metrics (Exact=1.0, Partial=0.5, Irrelevant=0.0)")
            print("-" * 70)
            for metric, value in sorted(results["weighted"].items()):
                print(f"  {metric:.<25} {value:.4f}")
        
        print("\n" + "="*70)
        
        # Highlight key metric
        if "standard" in results and "map@10" in results["standard"]:
            map_10 = results["standard"]["map@10"]
            print(f"\n🎯 KEY METRIC: MAP@10 = {map_10:.4f}")
            
            baseline = 0.29
            if map_10 > baseline:
                improvement = ((map_10 - baseline) / baseline) * 100
                print(f"   ✅ Improvement over baseline: +{improvement:.1f}%")
            else:
                print(f"   ⚠️  Below baseline of {baseline:.2f}")
        
        print()
    
    def run_full_pipeline(
        self,
        include_weighted: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete pipeline: load, fit, search, evaluate.
        
        Args:
            include_weighted: Whether to compute weighted metrics
            show_progress: Whether to show progress bars
            
        Returns:
            Dictionary with results and metrics
        """
        logger.info("Starting full pipeline execution")
        
        # Ensure data is loaded
        if self.product_df is None:
            self.load_data()
        
        # Fit retriever
        self.fit()
        
        # Prepare ground truth
        self.prepare_ground_truth()
        
        # Run batch search
        self.run_batch_search(show_progress=show_progress)
        
        # Evaluate
        results = self.evaluate(include_weighted=include_weighted)
        
        # Print results
        self.print_results(results)
        
        # Save results if enabled
        if self.save_results and self.results_manager:
            self._save_pipeline_results(results)
        
        logger.info("Pipeline execution completed")
        
        return {
            "metrics": results,
            "query_df": self.query_df,
            "retriever": self.retriever,
            "experiment_path": str(self.results_manager.experiment_dir) if self.results_manager else None
        }
    
    def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """
        Save all pipeline results using ResultsManager.
        
        Args:
            results: Dictionary containing metrics and results
        """
        try:
            logger.info("Saving pipeline results...")
            
            # Save model
            if self.retriever and hasattr(self.retriever, 'is_fitted') and self.retriever.is_fitted:
                model_name = f"{self.retriever.__class__.__name__}_{self.results_manager.experiment_id}"
                model_metadata = {
                    'retriever_type': self.retriever.__class__.__name__,
                    'top_k': getattr(self.retriever, 'top_k', None),
                    'is_fitted': self.retriever.is_fitted,
                    'n_products': len(self.product_df) if self.product_df is not None else 0
                }
                self.results_manager.save_model(self.retriever, model_name, model_metadata)
            
            # Save metrics
            self.results_manager.save_metrics(results, "metrics")
            
            # Save search results
            if self.query_df is not None:
                self.results_manager.save_search_results(self.query_df, "search_results")
            
            # Save configuration
            config_dict = config.to_dict()
            self.results_manager.save_config(config_dict, "config")
            
            # Save experiment summary
            summary = {
                'retriever_type': self.retriever.__class__.__name__ if self.retriever else None,
                'n_queries': len(self.query_df) if self.query_df is not None else 0,
                'n_products': len(self.product_df) if self.product_df is not None else 0,
                'n_labels': len(self.label_df) if self.label_df is not None else 0,
                'key_metrics': {
                    'map@10': results.get('standard', {}).get('map@10', None),
                    'weighted_map@10': results.get('weighted', {}).get('weighted_map@10', None),
                    'ndcg@10': results.get('weighted', {}).get('ndcg@10', None)
                }
            }
            self.results_manager.save_experiment_summary(summary)
            
            logger.info(f"Results saved to: {self.results_manager.experiment_dir}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
            # Don't raise exception to avoid breaking the pipeline