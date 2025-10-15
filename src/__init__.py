"""
E-commerce Search Engine Package

A modular, production-ready search engine with multiple retrieval strategies
and comprehensive evaluation metrics.
"""

__version__ = "0.2.0"
__author__ = "Laren Osorio"

from .models import BaseRetriever, TFIDFRetriever, BM25Retriever, SemanticRetriever, HybridRetriever
from .evaluation import MetricsCalculator, WeightedMetricsCalculator
from .pipeline import SearchPipeline
from .data_loader import DataLoader
from .config import config

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever", 
    "BM25Retriever",
    "SemanticRetriever",
    "HybridRetriever",
    "MetricsCalculator",
    "WeightedMetricsCalculator",
    "SearchPipeline",
    "DataLoader",
    "config",
]