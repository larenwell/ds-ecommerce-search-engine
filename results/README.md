# Results Directory

This directory contains all experiment results from the E-commerce Search Engine.

## Structure

```
results/
├── experiment_YYYYMMDD_HHMMSS/     # Individual experiment folders
│   ├── models/                     # Trained models
│   │   ├── ModelName_timestamp.pkl # Model file (large, excluded from git)
│   │   └── ModelName_timestamp.json # Model metadata
│   ├── metrics/                    # Evaluation metrics
│   │   ├── metrics.json           # Detailed metrics (JSON)
│   │   └── metrics.csv            # Flattened metrics (CSV)
│   ├── search_results/            # Search results
│   │   └── search_results.csv     # Query results DataFrame
│   ├── configs/                   # Configuration used
│   │   └── config.json            # Experiment configuration
│   ├── logs/                      # Execution logs
│   ├── plots/                     # Generated plots (future)
│   └── experiment_summary.json    # Experiment summary
├── experiment_comparison.csv       # Cross-experiment comparison
└── experiment_report.md           # Comprehensive report
```

## Usage

### Running Experiments
```bash
# Run single experiment
uv run python main.py --retriever bm25

# Run multiple experiments
uv run python examples/run_experiments.py
```

### Analyzing Results
```bash
# Analyze all experiments
uv run python examples/analyze_experiments.py

# Compare specific experiments
uv run python -c "
from src import ResultsManager
rm = ResultsManager()
experiments = rm.list_experiments()
print('Available experiments:', [exp['experiment_id'] for exp in experiments])
"
```

### Loading Saved Models
```python
from src import ResultsManager

# Load a specific model
rm = ResultsManager()
model = rm.load_model("BM25Retriever_20251015_082442")

# List all experiments
experiments = rm.list_experiments()
```

## Key Metrics

- **MAP@10**: Mean Average Precision at 10 (primary metric)
- **Weighted MAP@10**: MAP considering partial matches
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Precision@K**: Fraction of relevant results in top-K
- **Recall@K**: Fraction of relevant results found in top-K

## File Sizes

- **Model files (.pkl)**: ~200-500MB (excluded from git)
- **Metrics files**: ~1-5KB
- **Search results**: ~1-10MB
- **Config files**: ~1-2KB

## Best Practices

1. **Naming**: Experiments are automatically timestamped
2. **Cleanup**: Use `ResultsManager.cleanup_old_experiments()` to remove old experiments
3. **Comparison**: Always compare multiple experiments to validate improvements
4. **Documentation**: Check `experiment_summary.json` for quick overview
5. **Backup**: Large model files should be backed up separately

## Current Experiments

| Experiment ID | Retriever | MAP@10 | Status |
|---------------|-----------|--------|--------|
| 20251015_082442 | BM25Retriever | 0.3599 | ✅ Complete |

## Notes

- Model files (.pkl) are excluded from git due to size
- Only metadata and metrics are tracked in version control
- Use the analysis scripts to compare and visualize results
- Each experiment is self-contained and reproducible
