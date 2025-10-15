"""
Results Manager - Handles saving and loading of experiment results.

This module provides functionality to:
1. Save trained models
2. Save detailed metrics
3. Save search results
4. Save configurations
5. Organize results by experiment timestamp
"""

import json
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from loguru import logger

from .config import config


class ResultsManager:
    """
    Manages saving and loading of experiment results.
    
    Organizes results in timestamped folders for easy comparison
    and analysis of different experiments.
    """
    
    def __init__(self, base_dir: Path = None):
        """
        Initialize results manager.
        
        Args:
            base_dir: Base directory for results (default: results/)
        """
        self.base_dir = base_dir or Path("results")
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / f"experiment_{self.experiment_id}"
        
        # Create experiment directory structure
        self._create_experiment_structure()
        
        # Setup logging for this experiment
        from .config import config
        config.setup_logging(experiment_dir=self.experiment_dir)
        
        logger.info(f"ResultsManager initialized: {self.experiment_dir}")
    
    def _create_experiment_structure(self):
        """Create directory structure for current experiment."""
        dirs = [
            "models",
            "metrics", 
            "search_results",
            "configs",
            "logs",
            "plots"
        ]
        
        for dir_name in dirs:
            (self.experiment_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: Dict = None) -> Path:
        """
        Save trained model with metadata.
        
        Args:
            model: Trained model object
            model_name: Name for the model file
            metadata: Additional metadata about the model
            
        Returns:
            Path to saved model file
        """
        model_path = self.experiment_dir / "models" / f"{model_name}.pkl"
        
        try:
            # Save model using joblib (better for sklearn models)
            joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata_path = model_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str) -> Any:
        """
        Load saved model.
        
        Args:
            model_name: Name of the model file (without extension)
            
        Returns:
            Loaded model object
        """
        model_path = self.experiment_dir / "models" / f"{model_name}.pkl"
        
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def save_metrics(self, metrics: Dict[str, Any], metrics_name: str = "metrics") -> Path:
        """
        Save detailed metrics to JSON and CSV.
        
        Args:
            metrics: Dictionary of metrics
            metrics_name: Name for the metrics file
            
        Returns:
            Path to saved metrics file
        """
        # Save as JSON (preserves structure)
        json_path = self.experiment_dir / "metrics" / f"{metrics_name}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save as CSV (flattened for easy analysis)
        csv_path = self.experiment_dir / "metrics" / f"{metrics_name}.csv"
        self._flatten_metrics_to_csv(metrics, csv_path)
        
        logger.info(f"Metrics saved: {json_path}, {csv_path}")
        return json_path
    
    def _flatten_metrics_to_csv(self, metrics: Dict, csv_path: Path):
        """Flatten nested metrics dictionary to CSV format."""
        rows = []
        
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                for metric_name, value in category_metrics.items():
                    rows.append({
                        'category': category,
                        'metric': metric_name,
                        'value': value
                    })
            else:
                rows.append({
                    'category': 'general',
                    'metric': category,
                    'value': category_metrics
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    
    def save_search_results(self, query_df: pd.DataFrame, results_name: str = "search_results") -> Path:
        """
        Save search results DataFrame.
        
        Args:
            query_df: DataFrame with search results
            results_name: Name for the results file
            
        Returns:
            Path to saved results file
        """
        results_path = self.experiment_dir / "search_results" / f"{results_name}.csv"
        
        try:
            query_df.to_csv(results_path, index=False)
            logger.info(f"Search results saved: {results_path}")
            return results_path
        except Exception as e:
            logger.error(f"Error saving search results: {e}")
            raise
    
    def save_config(self, config_dict: Dict, config_name: str = "config") -> Path:
        """
        Save configuration used for experiment.
        
        Args:
            config_dict: Configuration dictionary
            config_name: Name for the config file
            
        Returns:
            Path to saved config file
        """
        config_path = self.experiment_dir / "configs" / f"{config_name}.json"
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Config saved: {config_path}")
            return config_path
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    
    def save_experiment_summary(self, summary: Dict[str, Any]) -> Path:
        """
        Save experiment summary with key results.
        
        Args:
            summary: Summary dictionary with key metrics and info
            
        Returns:
            Path to saved summary file
        """
        summary_path = self.experiment_dir / "experiment_summary.json"
        
        # Add experiment metadata
        summary.update({
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'experiment_dir': str(self.experiment_dir)
        })
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Experiment summary saved: {summary_path}")
            return summary_path
        except Exception as e:
            logger.error(f"Error saving experiment summary: {e}")
            raise
    
    def get_experiment_path(self) -> Path:
        """Get path to current experiment directory."""
        return self.experiment_dir
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all available experiments.
        
        Returns:
            List of experiment information dictionaries
        """
        experiments = []
        
        if not self.base_dir.exists():
            return experiments
        
        for exp_dir in self.base_dir.glob("experiment_*"):
            if exp_dir.is_dir():
                summary_path = exp_dir / "experiment_summary.json"
                
                exp_info = {
                    'experiment_id': exp_dir.name,
                    'path': str(exp_dir),
                    'created': exp_dir.stat().st_ctime
                }
                
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                        exp_info.update(summary)
                    except Exception as e:
                        logger.warning(f"Could not load summary for {exp_dir}: {e}")
                
                experiments.append(exp_info)
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x['created'], reverse=True)
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Compare metrics across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_dir = self.base_dir / exp_id
            metrics_path = exp_dir / "metrics" / "metrics.json"
            
            if metrics_path.exists():
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    
                    # Flatten metrics for comparison
                    flat_metrics = {'experiment_id': exp_id}
                    for category, category_metrics in metrics.items():
                        if isinstance(category_metrics, dict):
                            for metric_name, value in category_metrics.items():
                                flat_metrics[f"{category}_{metric_name}"] = value
                        else:
                            flat_metrics[category] = category_metrics
                    
                    comparison_data.append(flat_metrics)
                    
                except Exception as e:
                    logger.warning(f"Could not load metrics for {exp_id}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def cleanup_old_experiments(self, keep_last_n: int = 10):
        """
        Clean up old experiments, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent experiments to keep
        """
        experiments = self.list_experiments()
        
        if len(experiments) <= keep_last_n:
            logger.info(f"Only {len(experiments)} experiments found, no cleanup needed")
            return
        
        # Remove old experiments
        for exp in experiments[keep_last_n:]:
            exp_path = Path(exp['path'])
            try:
                import shutil
                shutil.rmtree(exp_path)
                logger.info(f"Removed old experiment: {exp_path}")
            except Exception as e:
                logger.error(f"Error removing experiment {exp_path}: {e}")


# Global results manager instance
results_manager = ResultsManager()
