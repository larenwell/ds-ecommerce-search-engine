"""
Evaluation metrics for search engine performance.

This module provides standard and weighted evaluation metrics:
- MetricsCalculator: Standard IR metrics (MAP, Precision, Recall, MRR)
- WeightedMetricsCalculator: Metrics with partial match support
"""

from .metrics import MetricsCalculator, WeightedMetricsCalculator

__all__ = [
    'MetricsCalculator',
    'WeightedMetricsCalculator',
]

