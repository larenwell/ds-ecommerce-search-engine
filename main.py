#!/usr/bin/env python3
"""
Unified E-commerce Search Engine - Main Entry Point

This script can run:
1. Individual retrieval methods: python main.py --method bm25
2. All methods in comparison: python main.py --method all
3. Comprehensive experiment: python main.py --comprehensive

Usage:
    python main.py --method bm25                    # Run single method
    python main.py --method all                     # Run all methods separately
    python main.py --comprehensive                  # Run all methods in organized structure
    python main.py --help                           # Show help
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import SearchPipeline, TFIDFRetriever, BM25Retriever, SemanticRetriever, HybridRetriever
from src.config import config


def run_single_method(method_name: str, save_results: bool = True):
    """Run a single retrieval method."""
    print(f"🚀 Running {method_name.upper()} Method")
    print("="*50)
    
    # Define method configurations
    method_configs = {
        'tfidf': (TFIDFRetriever, {
            'top_k': 10,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8
        }),
        'bm25': (BM25Retriever, {
            'top_k': 10,
            'k1': 1.5,
            'b': 0.75
        }),
        'semantic': (SemanticRetriever, {
            'top_k': 10,
            'model_name': 'all-MiniLM-L6-v2'
        }),
        'hybrid': (HybridRetriever, {
            'top_k': 10,
            'use_tfidf': True,
            'use_bm25': True,
            'use_semantic': True,
            'tfidf_weight': 0.2,
            'bm25_weight': 0.4,
            'semantic_weight': 0.4
        })
    }
    
    if method_name not in method_configs:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(method_configs.keys())}")
    
    retriever_class, kwargs = method_configs[method_name]
    
    # Initialize pipeline
    pipeline = SearchPipeline(
        data_dir=Path("WANDS/dataset"),
        save_results=save_results
    )
    
    # Create retriever
    retriever = retriever_class(**kwargs)
    pipeline.set_retriever(retriever)
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(include_weighted=True)
    
    # Extract key metrics
    metrics = results['metrics']
    
    print(f"✅ {method_name.upper()} completed!")
    print(f"📊 MAP@10: {metrics['standard']['map@10']:.4f}")
    
    return results


def run_all_methods_separately():
    """Run all methods separately (like run_all_experiments.py)."""
    print("🚀 Running All Methods Separately")
    print("="*50)
    
    methods = ['tfidf', 'bm25', 'semantic', 'hybrid']
    all_results = []
    
    for method in methods:
        print(f"\n🔍 Running {method.upper()}...")
        try:
            result = run_single_method(method, save_results=True)
            all_results.append({
                'method': method,
                'metrics': result['metrics'],
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"❌ Error running {method}: {e}")
            continue
    
    # Create comparison report
    if all_results:
        create_comparison_report(all_results)
    
    return all_results


def run_comprehensive_experiment():
    """Run all methods in organized structure (like run_comprehensive_experiment.py)."""
    print("🚀 Running Comprehensive Experiment")
    print("="*50)
    
    # Create main experiment directory
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_experiment_dir = Path("results") / f"experiment_{experiment_id}"
    main_experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Define experiments to run
    experiments = [
        ('tfidf', TFIDFRetriever, {
            'top_k': 10,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8
        }),
        ('bm25', BM25Retriever, {
            'top_k': 10,
            'k1': 1.5,
            'b': 0.75
        }),
        ('semantic', SemanticRetriever, {
            'top_k': 10,
            'model_name': 'all-MiniLM-L6-v2'
        }),
        ('hybrid', HybridRetriever, {
            'top_k': 10,
            'use_tfidf': True,
            'use_bm25': True,
            'use_semantic': True,
            'tfidf_weight': 0.2,
            'bm25_weight': 0.4,
            'semantic_weight': 0.4
        })
    ]
    
    all_results = []
    
    try:
        # Run each experiment
        for method_name, retriever_class, kwargs in experiments:
            print(f"\n🔍 Running {method_name.upper()} Experiment...")
            
            # Create method-specific directory
            method_dir = main_experiment_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            # Initialize pipeline with custom results manager
            pipeline = SearchPipeline(
                data_dir=Path("WANDS/dataset"),
                save_results=False  # We'll handle saving manually
            )
            
            # Create retriever
            retriever = retriever_class(**kwargs)
            pipeline.set_retriever(retriever)
            
            # Run full pipeline
            results = pipeline.run_full_pipeline(include_weighted=True)
            
            # Save method-specific results
            save_method_results(method_dir, method_name, results, pipeline)
            
            # Extract key metrics
            metrics = results['metrics']
            
            print(f"✅ {method_name.upper()} completed!")
            print(f"📊 MAP@10: {metrics['standard']['map@10']:.4f}")
            
            all_results.append({
                'method': method_name,
                'method_dir': str(method_dir),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
        
        # Create comprehensive comparison
        create_comprehensive_comparison(main_experiment_dir, all_results)
        
        # Find best performing method
        best_map10 = max(all_results, key=lambda x: x['metrics']['standard']['map@10'])
        print(f"\n🏆 Best MAP@10: {best_map10['method'].upper()} ({best_map10['metrics']['standard']['map@10']:.4f})")
        
        print(f"\n📁 All results saved to: {main_experiment_dir}")
        print("🎉 Comprehensive experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
    
    return all_results


def save_method_results(method_dir: Path, method_name: str, results: dict, pipeline):
    """Save results for a specific method with organized structure."""
    # Create subdirectories
    metrics_dir = method_dir / "metrics"
    configs_dir = method_dir / "configs"
    logs_dir = method_dir / "logs"
    search_results_dir = method_dir / "search_results"
    models_dir = method_dir / "models"
    
    for dir_path in [metrics_dir, configs_dir, logs_dir, search_results_dir, models_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Save metrics
    with open(metrics_dir / "metrics.json", 'w') as f:
        json.dump(results['metrics'], f, indent=2, default=str)
    
    # Flatten metrics to CSV
    metrics_data = []
    for category, category_metrics in results['metrics'].items():
        if isinstance(category_metrics, dict):
            for metric_name, value in category_metrics.items():
                metrics_data.append({
                    'category': category,
                    'metric': metric_name,
                    'value': value
                })
        else:
            metrics_data.append({
                'category': 'general',
                'metric': category,
                'value': category_metrics
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(metrics_dir / "metrics.csv", index=False)
    
    # Save search results
    if pipeline.query_df is not None:
        pipeline.query_df.to_csv(search_results_dir / "search_results.csv", index=False)
    
    # Save model metadata
    model_metadata = {
        'method': method_name,
        'retriever_type': pipeline.retriever.__class__.__name__,
        'top_k': getattr(pipeline.retriever, 'top_k', None),
        'is_fitted': pipeline.retriever.is_fitted,
        'n_products': len(pipeline.product_df) if pipeline.product_df is not None else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(models_dir / "model_metadata.json", 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    # Save configuration
    config_dict = config.to_dict()
    with open(configs_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Create a simple log file for this method
    log_file = logs_dir / f"{method_name}_experiment.log"
    with open(log_file, 'w') as f:
        f.write(f"Experiment: {method_name.upper()}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"MAP@10: {results['metrics']['standard']['map@10']:.4f}\n")
        f.write(f"Status: Completed Successfully\n")


def create_comprehensive_comparison(experiment_dir: Path, all_results: list):
    """Create comprehensive comparison report."""
    print("\n📊 Creating Comprehensive Comparison Report...")
    
    # Extract metrics for comparison
    comparison_data = []
    
    for result in all_results:
        method = result['method']
        metrics = result['metrics']
        
        # Standard metrics
        standard = metrics.get('standard', {})
        weighted = metrics.get('weighted', {})
        
        comparison_data.append({
            'Method': method.upper(),
            'MAP@5': standard.get('map@5', 0),
            'MAP@10': standard.get('map@10', 0),
            'MAP@20': standard.get('map@20', 0),
            'Precision@10': standard.get('precision@10', 0),
            'Recall@10': standard.get('recall@10', 0),
            'MRR': standard.get('mrr', 0),
            'Weighted_MAP@10': weighted.get('weighted_map@10', 0),
            'NDCG@10': weighted.get('ndcg@10', 0),
            'NDCG@20': weighted.get('ndcg@20', 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_path = experiment_dir / "method_comparison.csv"
    df.to_csv(comparison_path, index=False)
    
    # Create comprehensive JSON report
    report = {
        'experiment_id': experiment_dir.name,
        'experiment_date': datetime.now().isoformat(),
        'total_methods': len(all_results),
        'comparison_table': df.to_dict('records'),
        'individual_results': []
    }
    
    # Add individual results
    for result in all_results:
        report['individual_results'].append({
            'method': result['method'],
            'method_dir': result['method_dir'],
            'timestamp': result['timestamp'],
            'metrics': result['metrics']
        })
    
    # Save comprehensive report
    report_path = experiment_dir / "comprehensive_results.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create markdown report
    markdown_report = create_markdown_report(df, all_results, experiment_dir)
    markdown_path = experiment_dir / "experiment_report.md"
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    
    print(f"📄 Comparison saved: {comparison_path}")
    print(f"📄 Comprehensive report: {report_path}")
    print(f"📄 Markdown report: {markdown_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return df


def create_markdown_report(df: pd.DataFrame, all_results: list, experiment_dir: Path) -> str:
    """Create a comprehensive markdown report."""
    report = f"""# E-commerce Search Engine - Experiment Report

**Experiment ID:** {experiment_dir.name}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Methods:** {len(all_results)}

## Executive Summary

This experiment compares {len(all_results)} different retrieval methods for e-commerce search:

"""
    
    # Add method descriptions
    method_descriptions = {
        'tfidf': 'Term Frequency-Inverse Document Frequency with n-grams and document frequency filtering',
        'bm25': 'Best Matching 25 algorithm with term frequency saturation and document length normalization',
        'semantic': 'Semantic search using SentenceTransformer embeddings (all-MiniLM-L6-v2)',
        'hybrid': 'Weighted combination of TF-IDF, BM25, and Semantic search'
    }
    
    for result in all_results:
        method = result['method']
        description = method_descriptions.get(method, 'Unknown method')
        map10 = result['metrics']['standard']['map@10']
        report += f"- **{method.upper()}**: {description} (MAP@10: {map10:.4f})\n"
    
    report += f"""

## Results Comparison

### Performance Metrics

| Method | MAP@5 | MAP@10 | MAP@20 | Precision@10 | Recall@10 | MRR | Weighted MAP@10 | NDCG@10 | NDCG@20 |
|--------|-------|--------|--------|--------------|-----------|-----|-----------------|---------|---------|
"""
    
    # Add table rows
    for _, row in df.iterrows():
        report += f"| {row['Method']} | {row['MAP@5']:.4f} | {row['MAP@10']:.4f} | {row['MAP@20']:.4f} | {row['Precision@10']:.4f} | {row['Recall@10']:.4f} | {row['MRR']:.4f} | {row['Weighted_MAP@10']:.4f} | {row['NDCG@10']:.4f} | {row['NDCG@20']:.4f} |\n"
    
    # Find best method
    best_method = df.loc[df['MAP@10'].idxmax()]
    report += f"""

## Key Findings

- **Best Performing Method**: {best_method['Method']} with MAP@10 of {best_method['MAP@10']:.4f}
- **Target Achievement**: {'✅' if best_method['MAP@10'] >= 0.30 else '❌'} Target MAP@10 ≥ 0.30 {'(ACHIEVED)' if best_method['MAP@10'] >= 0.30 else '(NOT ACHIEVED)'}

## Directory Structure

```
{experiment_dir.name}/
├── experiment_report.md          # This report
├── method_comparison.csv         # Detailed comparison table
├── comprehensive_results.json    # Complete results data
├── tfidf/                        # TF-IDF method results
│   ├── configs/
│   ├── logs/
│   ├── metrics/
│   ├── search_results/
│   └── models/
├── bm25/                         # BM25 method results
│   ├── configs/
│   ├── logs/
│   ├── metrics/
│   ├── search_results/
│   └── models/
├── semantic/                     # Semantic method results
│   ├── configs/
│   ├── logs/
│   ├── metrics/
│   ├── search_results/
│   └── models/
└── hybrid/                       # Hybrid method results
    ├── configs/
    ├── logs/
    ├── metrics/
    ├── search_results/
    └── models/
```

## Individual Method Results

"""
    
    # Add individual method details
    for result in all_results:
        method = result['method']
        metrics = result['metrics']
        standard = metrics.get('standard', {})
        weighted = metrics.get('weighted', {})
        
        report += f"""### {method.upper()}

**Standard Metrics:**
- MAP@5: {standard.get('map@5', 0):.4f}
- MAP@10: {standard.get('map@10', 0):.4f}
- MAP@20: {standard.get('map@20', 0):.4f}
- Precision@10: {standard.get('precision@10', 0):.4f}
- Recall@10: {standard.get('recall@10', 0):.4f}
- MRR: {standard.get('mrr', 0):.4f}

**Weighted Metrics:**
- Weighted MAP@10: {weighted.get('weighted_map@10', 0):.4f}
- NDCG@10: {weighted.get('ndcg@10', 0):.4f}
- NDCG@20: {weighted.get('ndcg@20', 0):.4f}

"""
    
    report += f"""
## Conclusion

The experiment successfully evaluated {len(all_results)} different retrieval methods for e-commerce search. The {best_method['Method']} method achieved the highest MAP@10 score of {best_method['MAP@10']:.4f}, {'exceeding' if best_method['MAP@10'] >= 0.30 else 'falling short of'} the target threshold of 0.30.

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report


def create_comparison_report(all_results: list):
    """Create comparison report for separate method runs."""
    print("\n📊 Creating Comparison Report...")
    
    # Extract metrics for comparison
    comparison_data = []
    
    for result in all_results:
        method = result['method']
        metrics = result['metrics']
        
        # Standard metrics
        standard = metrics.get('standard', {})
        weighted = metrics.get('weighted', {})
        
        comparison_data.append({
            'Method': method.upper(),
            'MAP@5': standard.get('map@5', 0),
            'MAP@10': standard.get('map@10', 0),
            'MAP@20': standard.get('map@20', 0),
            'Precision@10': standard.get('precision@10', 0),
            'Recall@10': standard.get('recall@10', 0),
            'MRR': standard.get('mrr', 0),
            'Weighted_MAP@10': weighted.get('weighted_map@10', 0),
            'NDCG@10': weighted.get('ndcg@10', 0),
            'NDCG@20': weighted.get('ndcg@20', 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_path = Path("results") / "method_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(comparison_path, index=False)
    
    # Create comprehensive JSON report
    report = {
        'experiment_date': datetime.now().isoformat(),
        'total_methods': len(all_results),
        'comparison_table': df.to_dict('records'),
        'individual_results': []
    }
    
    # Add individual results
    for result in all_results:
        report['individual_results'].append({
            'method': result['method'],
            'timestamp': result['timestamp'],
            'metrics': result['metrics']
        })
    
    # Save comprehensive report
    report_path = Path("results") / "comprehensive_results.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Comparison saved: {comparison_path}")
    print(f"📄 Comprehensive report: {report_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return df


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="E-commerce Search Engine - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # Run all methods in organized structure (default)
  python main.py --method bm25                    # Run single method
  python main.py --method all                     # Run all methods separately
  python main.py --comprehensive                  # Run all methods in organized structure (explicit)
  python main.py --method hybrid --no-save        # Run hybrid without saving results
        """
    )
    
    # Method selection
    method_group = parser.add_mutually_exclusive_group(required=False)
    method_group.add_argument(
        '--method',
        choices=['tfidf', 'bm25', 'semantic', 'hybrid', 'all'],
        help='Retrieval method to run (or "all" for all methods)'
    )
    method_group.add_argument(
        '--comprehensive',
        action='store_true',
        help='Run comprehensive experiment with organized structure (default behavior)'
    )
    
    # Other options
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving results (only for single method runs)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    config.logging.level = args.log_level
    
    # Setup basic logging
    from loguru import logger
    import sys
    logger.remove()
    logger.add(sys.stderr, level=config.logging.level, format=config.logging.format, colorize=True)
    
    logger.info("="*80)
    logger.info("E-COMMERCE SEARCH ENGINE - UNIFIED ENTRY POINT")
    logger.info("="*80)
    
    try:
        if args.method is None and not args.comprehensive:
            # Default behavior: run comprehensive experiment
            logger.info("Running comprehensive experiment (default behavior)...")
            run_comprehensive_experiment()
        elif args.comprehensive:
            # Run comprehensive experiment
            run_comprehensive_experiment()
        elif args.method == 'all':
            # Run all methods separately
            run_all_methods_separately()
        else:
            # Run single method
            run_single_method(args.method, save_results=not args.no_save)
        
        print("\n🎉 Execution completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()