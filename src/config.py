"""
Configuration module for the search engine.

Centralizes all configuration parameters and constants using Pydantic for validation.
"""

from pathlib import Path
from typing import Dict, Tuple
from pydantic import BaseModel, Field
from loguru import logger
import sys


class DataConfig(BaseModel):
    """Data paths and loading configuration."""
    
    data_dir: Path = Field(default=Path("./data"), description="Base data directory")
    query_file: str = Field(default="query.csv", description="Query data filename")
    product_file: str = Field(default="product.csv", description="Product data filename")
    label_file: str = Field(default="label.csv", description="Label data filename")
    separator: str = Field(default="\t", description="CSV separator")
    
    @property
    def query_path(self) -> Path:
        """Full path to query file."""
        return self.data_dir / self.query_file
    
    @property
    def product_path(self) -> Path:
        """Full path to product file."""
        return self.data_dir / self.product_file
    
    @property
    def label_path(self) -> Path:
        """Full path to label file."""
        return self.data_dir / self.label_file


class RetrieverConfig(BaseModel):
    """Configuration for retrieval models."""
    
    # TF-IDF parameters
    tfidf_max_features: int = Field(
        default=10000, 
        description="Maximum number of features for TF-IDF vocabulary"
    )
    tfidf_ngram_range: Tuple[int, int] = Field(
        default=(1, 2), 
        description="N-gram range for TF-IDF (min_n, max_n)"
    )
    tfidf_min_df: int = Field(
        default=2, 
        description="Minimum document frequency for TF-IDF terms"
    )
    tfidf_max_df: float = Field(
        default=0.8, 
        description="Maximum document frequency ratio for TF-IDF terms"
    )
    
    # BM25 parameters
    bm25_k1: float = Field(
        default=1.5, 
        description="BM25 k1 parameter (term frequency saturation)"
    )
    bm25_b: float = Field(
        default=0.75, 
        description="BM25 b parameter (length normalization)"
    )
    
    # Semantic search parameters
    semantic_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name from HuggingFace"
    )
    semantic_batch_size: int = Field(
        default=32, 
        description="Batch size for encoding with semantic model"
    )
    semantic_device: str = Field(
        default="auto",
        description="Device for semantic model ('cuda', 'cpu', or 'auto')"
    )
    
    # General retrieval parameters
    top_k: int = Field(
        default=10, 
        description="Default number of results to retrieve"
    )
    
    # Hybrid retrieval weights
    hybrid_tfidf_weight: float = Field(
        default=0.3, 
        description="Weight for TF-IDF scores in hybrid retrieval"
    )
    hybrid_bm25_weight: float = Field(
        default=0.3, 
        description="Weight for BM25 scores in hybrid retrieval"
    )
    hybrid_semantic_weight: float = Field(
        default=0.4, 
        description="Weight for semantic scores in hybrid retrieval"
    )
    hybrid_retrieval_k: int = Field(
        default=50,
        description="Number of candidates to retrieve from each model in hybrid"
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics."""
    
    k_values: list = Field(
        default=[5, 10, 20], 
        description="K values for computing metrics@K"
    )
    
    # Weights for partial matches in weighted metrics
    exact_weight: float = Field(
        default=1.0, 
        description="Weight for exact matches in weighted metrics"
    )
    partial_weight: float = Field(
        default=0.5, 
        description="Weight for partial matches in weighted metrics"
    )
    irrelevant_weight: float = Field(
        default=0.0, 
        description="Weight for irrelevant items in weighted metrics"
    )
    
    # Label mapping for graded relevance
    label_mapping: Dict[str, float] = Field(
        default={
            "Exact": 1.0,
            "Partial": 0.5,
            "Irrelevant": 0.0
        },
        description="Mapping from label strings to relevance scores"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(
        default="INFO", 
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: str = Field(
        default="logs/search_engine.log", 
        description="Path to log file"
    )
    rotation: str = Field(
        default="500 MB", 
        description="Log file rotation size"
    )
    retention: str = Field(
        default="10 days", 
        description="Log file retention period"
    )
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        description="Log message format"
    )


class Config(BaseModel):
    """Master configuration class combining all config sections."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    def setup_logging(self) -> None:
        """
        Setup loguru logging with current configuration.
        
        Configures both console and file logging with rotation and retention.
        """
        # Remove default handler
        logger.remove()
        
        # Add console handler with colors
        logger.add(
            sys.stderr,
            level=self.logging.level,
            format=self.logging.format,
            colorize=True
        )
        
        # Add file handler with rotation
        log_path = Path(self.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            level=self.logging.level,
            rotation=self.logging.rotation,
            retention=self.logging.retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            compression="zip"  # Compress rotated logs
        )
        
        logger.info("Logging system configured successfully")
        logger.debug(f"Log level: {self.logging.level}")
        logger.debug(f"Log file: {log_path}")
    
    def update_from_dict(self, updates: Dict) -> None:
        """
        Update configuration from a dictionary.
        
        Args:
            updates: Dictionary with configuration updates
            
        Example:
            config.update_from_dict({
                'retriever': {'bm25_k1': 2.0},
                'evaluation': {'exact_weight': 1.0}
            })
        """
        for section, params in updates.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in params.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                        logger.debug(f"Updated {section}.{key} = {value}")
    
    def to_dict(self) -> Dict:
        """
        Export configuration as dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'data': self.data.model_dump(),
            'retriever': self.retriever.model_dump(),
            'evaluation': self.evaluation.model_dump(),
            'logging': self.logging.model_dump()
        }
    
    def print_config(self) -> None:
        """Print current configuration in a readable format."""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        
        print("\n📁 DATA:")
        print(f"  Directory: {self.data.data_dir}")
        print(f"  Query file: {self.data.query_file}")
        print(f"  Product file: {self.data.product_file}")
        print(f"  Label file: {self.data.label_file}")
        
        print("\n🔍 RETRIEVER:")
        print(f"  TF-IDF max features: {self.retriever.tfidf_max_features}")
        print(f"  TF-IDF n-grams: {self.retriever.tfidf_ngram_range}")
        print(f"  BM25 k1: {self.retriever.bm25_k1}")
        print(f"  BM25 b: {self.retriever.bm25_b}")
        print(f"  Semantic model: {self.retriever.semantic_model_name}")
        
        print("\n📊 EVALUATION:")
        print(f"  K values: {self.evaluation.k_values}")
        print(f"  Exact weight: {self.evaluation.exact_weight}")
        print(f"  Partial weight: {self.evaluation.partial_weight}")
        
        print("\n📝 LOGGING:")
        print(f"  Level: {self.logging.level}")
        print(f"  Log file: {self.logging.log_file}")
        
        print("\n" + "="*70 + "\n")


# Global config instance
config = Config()

# Setup logging on import
config.setup_logging()

logger.info("Configuration module loaded")