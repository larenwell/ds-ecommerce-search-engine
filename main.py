#!/usr/bin/env python3
"""
Main script for E-commerce Search Engine.

Usage:
    python main.py --retriever bm25
    python main.py --retriever hybrid --data-dir WANDS/dataset
    python main.py --retriever tfidf --top-k 20
"""

import argparse
import sys
from pathlib import Path

from src.models import TFIDFRetriever, BM25Retriever, HybridRetriever
from src.pipeline import SearchPipeline
from src.config import config
from loguru import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="E-commerce Search Engine - Train and evaluate retrieval models"
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='WANDS/dataset',
        help='Path to WANDS dataset directory (default: WANDS/dataset)'
    )
    
    # Retriever selection
    parser.add_argument(
        '--retriever',
        type=str,
        choices=['tfidf', 'bm25', 'hybrid'],
        default='bm25',
        help='Retriever type to use (default: bm25)'
    )
    
    # Retriever parameters
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve (default: 10)'
    )
    
    # TF-IDF parameters
    parser.add_argument(
        '--tfidf-ngrams',
        type=str,
        default='1,2',
        help='N-gram range for TF-IDF as "min,max" (default: 1,2)'
    )
    
    # BM25 parameters
    parser.add_argument(
        '--bm25-k1',
        type=float,
        default=1.5,
        help='BM25 k1 parameter (default: 1.5)'
    )
    parser.add_argument(
        '--bm25-b',
        type=float,
        default=0.75,
        help='BM25 b parameter (default: 0.75)'
    )
    
    # Hybrid parameters
    parser.add_argument(
        '--hybrid-use-semantic',
        action='store_true',
        help='Enable semantic search in hybrid (slower)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--no-weighted-metrics',
        action='store_true',
        help='Disable weighted metrics evaluation'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )
    
    # Output options
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save results CSV (default: None)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def create_retriever(args):
    """
    Create retriever based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        BaseRetriever instance
    """
    if args.retriever == 'tfidf':
        logger.info("Creating TF-IDF retriever")
        ngram_min, ngram_max = map(int, args.tfidf_ngrams.split(','))
        return TFIDFRetriever(
            top_k=args.top_k,
            ngram_range=(ngram_min, ngram_max)
        )
    
    elif args.retriever == 'bm25':
        logger.info("Creating BM25 retriever")
        return BM25Retriever(
            top_k=args.top_k,
            k1=args.bm25_k1,
            b=args.bm25_b
        )
    
    elif args.retriever == 'hybrid':
        logger.info("Creating Hybrid retriever")
        return HybridRetriever(
            top_k=args.top_k,
            use_tfidf=True,
            use_bm25=True,
            use_semantic=args.hybrid_use_semantic
        )
    
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever}")


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    config.logging.level = args.log_level
    config.setup_logging()
    
    logger.info("="*80)
    logger.info("E-COMMERCE SEARCH ENGINE")
    logger.info("="*80)
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please download the dataset: git clone https://github.com/wayfair/WANDS.git")
        return 1
    
    logger.info(f"Data directory: {data_dir}")
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = SearchPipeline(data_dir=data_dir)
    
    # Load data
    logger.info("Loading data...")
    pipeline.load_data()
    
    logger.info(f"Data loaded: {len(pipeline.query_df)} queries, {len(pipeline.product_df)} products")
    
    # Create retriever
    retriever = create_retriever(args)
    pipeline.set_retriever(retriever)
    
    logger.info(f"Retriever: {retriever}")
    
    # Run full pipeline
    logger.info("Running full pipeline...")
    results = pipeline.run_full_pipeline(
        include_weighted=not args.no_weighted_metrics,
        show_progress=not args.no_progress
    )
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_path}")
        results['query_df'].to_csv(output_path, index=False)
        logger.info("Results saved successfully")
    
    logger.info("="*80)
    logger.info("Pipeline completed successfully")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        sys.exit(1)