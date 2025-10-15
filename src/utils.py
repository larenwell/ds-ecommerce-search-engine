"""
Utility functions shared across the package.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path


def load_wands_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all WANDS dataset files.
    
    Args:
        data_dir: Directory containing WANDS data files
        
    Returns:
        Dictionary with 'queries', 'products', 'labels' dataframes
    """
    data = {
        'queries': pd.read_csv(data_dir / 'query.csv', sep='\t'),
        'products': pd.read_csv(data_dir / 'product.csv', sep='\t'),
        'labels': pd.read_csv(data_dir / 'label.csv', sep='\t')
    }
    return data


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores to [0, 1] range.
    
    Args:
        scores: Array of scores
        
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that returns default value when denominator is 0.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default