"""
Data Preparation Module for Advanced Retrieval Methods.

Transforms raw WANDS data into formats required by different advanced methods:
- CrossEncoder: (query, document, score) triplets
- DPR/ColBERT: (query, positive_docs, negative_docs) format
- BERT Fine-tuning: Classification format
- Contrastive Learning: Pairs with labels
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
import random

from .config import config


class DatasetPreparer:
    """
    Prepares WANDS dataset for different advanced retrieval methods.
    
    Handles:
    - Creating query-document pairs
    - Balancing positive/negative samples
    - Train/val/test splits
    - Hard negative mining
    """
    
    def __init__(
        self,
        query_df: pd.DataFrame,
        product_df: pd.DataFrame,
        label_df: pd.DataFrame
    ):
        """
        Initialize with WANDS dataframes.
        
        Args:
            query_df: DataFrame with queries
            product_df: DataFrame with products
            label_df: DataFrame with relevance labels
        """
        self.query_df = query_df
        self.product_df = product_df
        self.label_df = label_df
        
        # Create mappings
        self._create_mappings()
        
        logger.info(
            f"DatasetPreparer initialized: "
            f"{len(query_df)} queries, "
            f"{len(product_df)} products, "
            f"{len(label_df)} labels"
        )
    
    def _create_mappings(self) -> None:
        """Create efficient lookup mappings."""
        # Product ID to text
        self.product_id_to_text = {}
        for _, row in self.product_df.iterrows():
            text = f"{row['product_name']} {row.get('product_description', '')}"
            self.product_id_to_text[row['product_id']] = text.strip()
        
        # Query ID to text
        self.query_id_to_text = dict(
            zip(self.query_df['query_id'], self.query_df['query'])
        )
        
        logger.debug("Created product and query mappings")
    
    def prepare_crossencoder_data(
        self,
        label_mapping: Optional[Dict[str, float]] = None,
        balance_samples: bool = True,
        max_samples_per_query: int = 100
    ) -> pd.DataFrame:
        """
        Prepare data for CrossEncoder training/fine-tuning.
        
        Format: (query, document_text, relevance_score)
        
        Args:
            label_mapping: Map labels to scores (default from config)
            balance_samples: Whether to balance positive/negative samples
            max_samples_per_query: Max samples to include per query
            
        Returns:
            DataFrame with columns: [query, document, score]
        """
        logger.info("Preparing CrossEncoder dataset...")
        
        if label_mapping is None:
            label_mapping = config.evaluation.label_mapping
        
        data = []
        
        for query_id in self.query_df['query_id']:
            query_text = self.query_id_to_text[query_id]
            
            # Get all labels for this query
            query_labels = self.label_df[self.label_df['query_id'] == query_id]
            
            # Convert to triplets
            for _, label_row in query_labels.iterrows():
                product_id = label_row['product_id']
                label = label_row['label']
                
                # Skip if product not found
                if product_id not in self.product_id_to_text:
                    continue
                
                product_text = self.product_id_to_text[product_id]
                score = label_mapping.get(label, 0.0)
                
                data.append({
                    'query': query_text,
                    'document': product_text,
                    'score': score,
                    'label': label
                })
            
            # Balance if requested
            if balance_samples and len(data) > max_samples_per_query:
                # Keep all positives, sample negatives
                query_data = [d for d in data if d['query'] == query_text]
                positives = [d for d in query_data if d['score'] > 0]
                negatives = [d for d in query_data if d['score'] == 0]
                
                if len(negatives) > len(positives):
                    sampled_negatives = random.sample(negatives, len(positives))
                    data = [d for d in data if d['query'] != query_text]
                    data.extend(positives + sampled_negatives)
        
        df = pd.DataFrame(data)
        
        logger.info(
            f"CrossEncoder dataset prepared: {len(df)} pairs, "
            f"label distribution: {df['label'].value_counts().to_dict()}"
        )
        
        return df
    
    def prepare_dpr_data(
        self,
        num_hard_negatives: int = 5,
        num_random_negatives: int = 2
    ) -> List[Dict]:
        """
        Prepare data for Dense Passage Retrieval (DPR) training.
        
        Format: {
            'query': str,
            'positive': List[str],
            'hard_negative': List[str],
            'random_negative': List[str]
        }
        
        Args:
            num_hard_negatives: Number of hard negatives per query
            num_random_negatives: Number of random negatives per query
            
        Returns:
            List of training examples
        """
        logger.info("Preparing DPR dataset...")
        
        data = []
        
        for query_id in self.query_df['query_id']:
            query_text = self.query_id_to_text[query_id]
            query_labels = self.label_df[self.label_df['query_id'] == query_id]
            
            # Get positives (Exact matches)
            positive_ids = query_labels[
                query_labels['label'] == 'Exact'
            ]['product_id'].tolist()
            
            if not positive_ids:
                continue
            
            positives = [
                self.product_id_to_text[pid] 
                for pid in positive_ids 
                if pid in self.product_id_to_text
            ]
            
            # Get hard negatives (Partial matches - close but not exact)
            hard_negative_ids = query_labels[
                query_labels['label'] == 'Partial'
            ]['product_id'].tolist()
            
            hard_negatives = [
                self.product_id_to_text[pid]
                for pid in random.sample(
                    hard_negative_ids,
                    min(num_hard_negatives, len(hard_negative_ids))
                )
                if pid in self.product_id_to_text
            ]
            
            # Get random negatives (Irrelevant)
            irrelevant_ids = query_labels[
                query_labels['label'] == 'Irrelevant'
            ]['product_id'].tolist()
            
            random_negatives = [
                self.product_id_to_text[pid]
                for pid in random.sample(
                    irrelevant_ids,
                    min(num_random_negatives, len(irrelevant_ids))
                )
                if pid in self.product_id_to_text
            ]
            
            data.append({
                'query': query_text,
                'positive': positives,
                'hard_negative': hard_negatives,
                'random_negative': random_negatives
            })
        
        logger.info(
            f"DPR dataset prepared: {len(data)} queries, "
            f"avg positives: {np.mean([len(d['positive']) for d in data]):.1f}, "
            f"avg hard negatives: {np.mean([len(d['hard_negative']) for d in data]):.1f}"
        )
        
        return data
    
    def prepare_classification_data(
        self,
        format_type: str = "binary"
    ) -> pd.DataFrame:
        """
        Prepare data for classification tasks (BERT fine-tuning).
        
        Args:
            format_type: Type of classification
                - "binary": Relevant (1) vs Irrelevant (0)
                - "multiclass": Exact (2), Partial (1), Irrelevant (0)
            
        Returns:
            DataFrame with columns: [text, label]
        """
        logger.info(f"Preparing {format_type} classification dataset...")
        
        data = []
        
        for _, label_row in self.label_df.iterrows():
            query_id = label_row['query_id']
            product_id = label_row['product_id']
            label = label_row['label']
            
            if query_id not in self.query_id_to_text:
                continue
            if product_id not in self.product_id_to_text:
                continue
            
            query_text = self.query_id_to_text[query_id]
            product_text = self.product_id_to_text[product_id]
            
            # Combine query and product
            combined_text = f"[CLS] {query_text} [SEP] {product_text}"
            
            # Map label
            if format_type == "binary":
                label_value = 1 if label == "Exact" else 0
            else:  # multiclass
                label_map = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
                label_value = label_map[label]
            
            data.append({
                'text': combined_text,
                'label': label_value,
                'original_label': label
            })
        
        df = pd.DataFrame(data)
        
        logger.info(
            f"Classification dataset prepared: {len(df)} examples, "
            f"label distribution: {df['label'].value_counts().to_dict()}"
        )
        
        return df
    
    def create_train_val_test_split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            df: Input dataframe
            val_size: Validation set ratio
            test_size: Test set ratio
            random_state: Random seed
            
        Returns:
            (train_df, val_df, test_df)
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label'] if 'label' in df.columns else None
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val['label'] if 'label' in train_val.columns else None
        )
        
        logger.info(
            f"Split created: train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )
        
        return train, val, test
    
    def export_for_training(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = "csv"
    ) -> None:
        """
        Export prepared dataset for training.
        
        Args:
            df: Prepared dataframe
            output_path: Output file path
            format: Export format ('csv', 'json', 'parquet')
        """
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient='records', lines=True)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Dataset exported to {output_path}")
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_queries': len(self.query_df),
            'num_products': len(self.product_df),
            'num_labels': len(self.label_df),
            'label_distribution': self.label_df['label'].value_counts().to_dict(),
            'avg_labels_per_query': len(self.label_df) / len(self.query_df),
            'queries_with_exact_match': len(
                self.label_df[self.label_df['label'] == 'Exact']['query_id'].unique()
            )
        }
        
        return stats
    
    def print_statistics(self) -> None:
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        print(f"  Queries: {stats['num_queries']}")
        print(f"  Products: {stats['num_products']}")
        print(f"  Labels: {stats['num_labels']}")
        print(f"\n  Label Distribution:")
        for label, count in stats['label_distribution'].items():
            pct = (count / stats['num_labels']) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")
        print(f"\n  Avg labels per query: {stats['avg_labels_per_query']:.1f}")
        print(f"  Queries with exact match: {stats['queries_with_exact_match']}")
        print("="*70 + "\n")


# Helper functions for quick access

def prepare_crossencoder_dataset(
    query_df: pd.DataFrame,
    product_df: pd.DataFrame,
    label_df: pd.DataFrame,
    output_dir: str = "data/prepared"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Quick function to prepare CrossEncoder dataset with train/val/test split.
    
    Returns:
        (train_df, val_df, test_df)
    """
    preparer = DatasetPreparer(query_df, product_df, label_df)
    
    # Prepare full dataset
    full_df = preparer.prepare_crossencoder_data(balance_samples=True)
    
    # Split
    train, val, test = preparer.create_train_val_test_split(full_df)
    
    # Export
    preparer.export_for_training(train, f"{output_dir}/crossencoder_train.csv")
    preparer.export_for_training(val, f"{output_dir}/crossencoder_val.csv")
    preparer.export_for_training(test, f"{output_dir}/crossencoder_test.csv")
    
    logger.info("CrossEncoder dataset prepared and exported")
    
    return train, val, test


def prepare_dpr_dataset(
    query_df: pd.DataFrame,
    product_df: pd.DataFrame,
    label_df: pd.DataFrame,
    output_path: str = "data/prepared/dpr_dataset.json"
) -> List[Dict]:
    """
    Quick function to prepare DPR dataset.
    
    Returns:
        List of DPR training examples
    """
    preparer = DatasetPreparer(query_df, product_df, label_df)
    
    # Prepare DPR data
    dpr_data = preparer.prepare_dpr_data(
        num_hard_negatives=5,
        num_random_negatives=2
    )
    
    # Export
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dpr_data, f, indent=2)
    
    logger.info(f"DPR dataset exported to {output_path}")
    
    return dpr_data