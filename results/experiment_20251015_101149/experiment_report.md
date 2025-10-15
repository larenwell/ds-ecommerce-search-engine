# E-commerce Search Engine - Experiment Report

**Experiment ID:** experiment_20251015_101149  
**Date:** 2025-10-15 10:22:04  
**Total Methods:** 4

## Executive Summary

This experiment compares 4 different retrieval methods for e-commerce search:

- **TFIDF**: Term Frequency-Inverse Document Frequency with n-grams and document frequency filtering (MAP@10: 0.1681)
- **BM25**: Best Matching 25 algorithm with term frequency saturation and document length normalization (MAP@10: 0.3599)
- **SEMANTIC**: Semantic search using SentenceTransformer embeddings (all-MiniLM-L6-v2) (MAP@10: 0.3353)
- **HYBRID**: Weighted combination of TF-IDF, BM25, and Semantic search (MAP@10: 0.3805)


## Results Comparison

### Performance Metrics

| Method | MAP@5 | MAP@10 | MAP@20 | Precision@10 | Recall@10 | MRR | Weighted MAP@10 | NDCG@10 | NDCG@20 |
|--------|-------|--------|--------|--------------|-----------|-----|-----------------|---------|---------|
| TFIDF | 0.1810 | 0.1681 | 0.0911 | 0.2096 | 0.0647 | 0.2875 | 0.3169 | 0.5033 | 0.3359 |
| BM25 | 0.3797 | 0.3599 | 0.2403 | 0.3204 | 0.1968 | 0.4964 | 0.4527 | 0.6449 | 0.4321 |
| SEMANTIC | 0.3540 | 0.3353 | 0.2115 | 0.3223 | 0.1726 | 0.4702 | 0.4821 | 0.6913 | 0.4630 |
| HYBRID | 0.3997 | 0.3805 | 0.2507 | 0.3427 | 0.2061 | 0.5223 | 0.4943 | 0.6982 | 0.4676 |


## Key Findings

- **Best Performing Method**: HYBRID with MAP@10 of 0.3805
- **Target Achievement**: ✅ Target MAP@10 ≥ 0.30 (ACHIEVED)

## Directory Structure

```
experiment_20251015_101149/
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

### TFIDF

**Standard Metrics:**
- MAP@5: 0.1810
- MAP@10: 0.1681
- MAP@20: 0.0911
- Precision@10: 0.2096
- Recall@10: 0.0647
- MRR: 0.2875

**Weighted Metrics:**
- Weighted MAP@10: 0.3169
- NDCG@10: 0.5033
- NDCG@20: 0.3359

### BM25

**Standard Metrics:**
- MAP@5: 0.3797
- MAP@10: 0.3599
- MAP@20: 0.2403
- Precision@10: 0.3204
- Recall@10: 0.1968
- MRR: 0.4964

**Weighted Metrics:**
- Weighted MAP@10: 0.4527
- NDCG@10: 0.6449
- NDCG@20: 0.4321

### SEMANTIC

**Standard Metrics:**
- MAP@5: 0.3540
- MAP@10: 0.3353
- MAP@20: 0.2115
- Precision@10: 0.3223
- Recall@10: 0.1726
- MRR: 0.4702

**Weighted Metrics:**
- Weighted MAP@10: 0.4821
- NDCG@10: 0.6913
- NDCG@20: 0.4630

### HYBRID

**Standard Metrics:**
- MAP@5: 0.3997
- MAP@10: 0.3805
- MAP@20: 0.2507
- Precision@10: 0.3427
- Recall@10: 0.2061
- MRR: 0.5223

**Weighted Metrics:**
- Weighted MAP@10: 0.4943
- NDCG@10: 0.6982
- NDCG@20: 0.4676


## Conclusion

The experiment successfully evaluated 4 different retrieval methods for e-commerce search. The HYBRID method achieved the highest MAP@10 score of 0.3805, exceeding the target threshold of 0.30.

---
*Report generated on 2025-10-15 10:22:04*
