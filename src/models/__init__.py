"""
Retriever models package.

Contains all retrieval implementations:
- BaseRetriever: Abstract base class
- TFIDFRetriever: Enhanced TF-IDF based retrieval
- BM25Retriever: BM25 ranking algorithm
- SemanticRetriever: Semantic search using transformers
- HybridRetriever: Combines multiple retrievers
- OllamaRetriever: LLM-based retrieval with Ollama
- CrossEncoderRetriever: Re-ranking with CrossEncoder
- ColBERTRetriever: Late interaction retrieval
- DPRRetriever: Dense Passage Retrieval
- BERTRetriever: Fine-tuned BERT for relevance
"""

from .base import BaseRetriever
from .tfidf import TFIDFRetriever
from .bm25 import BM25Retriever

# Optional imports (may not be available)
try:
    from .semantic import SemanticRetriever
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

from .hybrid import HybridRetriever

try:
    from .ollama_retriever import OllamaRetriever
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from .crossencoder_retriever import CrossEncoderRetriever
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

try:
    from .colbert_retriever import ColBERTRetriever
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False

try:
    from .dpr_retriever import DPRRetriever
    DPR_AVAILABLE = True
except ImportError:
    DPR_AVAILABLE = False

try:
    from .bert_retriever import BERTRetriever
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever",
    "BM25Retriever",
    "HybridRetriever",
]

# Add optional retrievers to __all__ if available
if SEMANTIC_AVAILABLE:
    __all__.append("SemanticRetriever")

if OLLAMA_AVAILABLE:
    __all__.append("OllamaRetriever")

if CROSSENCODER_AVAILABLE:
    __all__.append("CrossEncoderRetriever")

if COLBERT_AVAILABLE:
    __all__.append("ColBERTRetriever")

if DPR_AVAILABLE:
    __all__.append("DPRRetriever")

if BERT_AVAILABLE:
    __all__.append("BERTRetriever")