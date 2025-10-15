"""
BERT Fine-tuned Retriever for Relevance Classification.

Fine-tunes BERT to classify query-document pairs as:
- Binary: Relevant (1) vs Irrelevant (0)
- Multiclass: Exact (2), Partial (1), Irrelevant (0)

Then uses classification scores for ranking.
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        BertForSequenceClassification,
        BertTokenizer,
        AdamW,
        get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")

from .base import BaseRetriever
from .bm25 import BM25Retriever
from ..config import config


class RelevanceDataset(Dataset):
    """Dataset for BERT relevance classification."""
    
    def __init__(
        self,
        queries: List[str],
        documents: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]
        label = self.labels[idx]
        
        # Encode [CLS] query [SEP] document [SEP]
        encoded = self.tokenizer(
            query,
            document,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTRetriever(BaseRetriever):
    """
    BERT fine-tuned for query-document relevance classification.
    
    Strategy:
    1. Fine-tune BERT on (query, document, label) triplets
    2. At inference: Get candidates from base retriever
    3. Classify each (query, candidate) pair
    4. Rank by classification confidence scores
    
    Advantages:
    - Highly accurate when fine-tuned on domain data
    - Can learn complex relevance patterns
    - Captures query-document interactions
    - Explainable (classification probabilities)
    
    Trade-offs:
    - Requires labeled training data
    - Training time: hours to days
    - Slower inference than BM25
    - Need to re-train for different domains
    
    Performance:
    - With good training data: Can achieve MAP@10 > 0.45
    - Best when combined with hard negative mining
    
    Example:
        >>> retriever = BERTRetriever(
        ...     model_name='bert-base-uncased',
        ...     num_labels=2  # Binary classification
        ... )
        >>> # Prepare training data first
        >>> retriever.train(train_df, val_df, epochs=3)
        >>> retriever.fit(product_df)
        >>> results = retriever.retrieve("comfortable chair")
    """
    
    def __init__(
        self,
        top_k: int = 10,
        model_name: str = 'bert-base-uncased',
        num_labels: int = 2,
        base_retriever: Optional[BaseRetriever] = None,
        candidate_k: int = 50,
        max_length: int = 512,
        device: str = None,
        text_fields: Optional[List[str]] = None
    ):
        """
        Initialize BERT retriever.
        
        Args:
            top_k: Number of final results
            model_name: BERT model name
            num_labels: Number of classes (2=binary, 3=multiclass)
            base_retriever: Base retriever for candidates
            candidate_k: Number of candidates to classify
            max_length: Maximum sequence length
            device: Device for inference
            text_fields: Product fields to use
        """
        super().__init__(top_k)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers required. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.candidate_k = candidate_k
        self.max_length = max_length
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.text_fields = text_fields or [
            'product_name',
            'product_description'
        ]
        
        # Load BERT model
        logger.info(f"Loading BERT model: {model_name}")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Base retriever
        self.base_retriever = base_retriever or BM25Retriever(top_k=candidate_k)
        
        # Training state
        self.is_trained = False
        
        logger.info(
            f"BERTRetriever initialized: "
            f"model={model_name}, num_labels={num_labels}, device={self.device}"
        )
    
    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ) -> Dict:
        """
        Fine-tune BERT on relevance classification task.
        
        Args:
            train_df: Training data with [query, document, label]
            val_df: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            
        Returns:
            Training history
        """
        logger.info(f"Training BERT for {epochs} epochs...")
        
        # Prepare datasets
        train_dataset = RelevanceDataset(
            train_df['query'].tolist(),
            train_df['document'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        self.model.train()
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                if (batch_idx + 1) % 50 == 0:
                    logger.debug(
                        f"Batch {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )
            
            avg_train_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_df is not None:
                val_loss, val_acc = self._evaluate(val_df, batch_size)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                logger.info(
                    f"Validation loss: {val_loss:.4f}, "
                    f"accuracy: {val_acc:.4f}"
                )
        
        self.is_trained = True
        logger.info("Training complete!")
        
        return history
    
    def _evaluate(
        self,
        eval_df: pd.DataFrame,
        batch_size: int = 16
    ) -> Tuple[float, float]:
        """Evaluate model on validation data."""
        eval_dataset = RelevanceDataset(
            eval_df['query'].tolist(),
            eval_df['document'].tolist(),
            eval_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        self.model.train()
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self, product_df: pd.DataFrame) -> 'BERTRetriever':
        """
        Fit retriever (mainly fits base retriever).
        
        Note: You should call train_model() before fit().
        
        Args:
            product_df: DataFrame with product information
            
        Returns:
            Self for method chaining
        """
        if not self.is_trained:
            logger.warning(
                "BERT model not trained yet. "
                "Call train_model() before fit() for best results."
            )
        
        logger.info("Fitting BERT retriever...")
        
        self._validate_product_df(product_df)
        self.product_df = product_df.copy()
        
        # Fit base retriever
        self.base_retriever.fit(product_df)
        
        # Prepare product texts
        self._prepare_product_texts()
        
        self.is_fitted = True
        logger.info("BERT retriever fitted")
        
        return self
    
    def _prepare_product_texts(self) -> None:
        """Prepare product texts for classification."""
        self.product_texts = {}
        
        available_fields = [
            f for f in self.text_fields if f in self.product_df.columns
        ]
        
        for _, row in self.product_df.iterrows():
            product_id = row['product_id']
            text_parts = [
                str(row[field]) for field in available_fields
                if pd.notna(row[field])
            ]
            self.product_texts[product_id] = " ".join(text_parts)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
        """Retrieve products using BERT classification."""
        results = self.retrieve_with_scores(query, top_k)
        return [pid for pid, _ in results]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve with BERT classification scores.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (product_id, score) tuples
        """
        self._validate_fitted()
        
        k = top_k or self.top_k
        
        try:
            # Get candidates
            candidates = self.base_retriever.retrieve(query, self.candidate_k)
            
            if not candidates:
                return []
            
            # Prepare pairs for classification
            pairs = []
            valid_candidates = []
            
            for pid in candidates:
                if pid in self.product_texts:
                    pairs.append((query, self.product_texts[pid]))
                    valid_candidates.append(pid)
            
            if not pairs:
                return []
            
            # Classify pairs
            self.model.eval()
            scores = []
            
            with torch.no_grad():
                for i in range(0, len(pairs), 16):  # Batch size 16
                    batch_pairs = pairs[i:i + 16]
                    
                    # Tokenize
                    encoded = self.tokenizer(
                        [p[0] for p in batch_pairs],
                        [p[1] for p in batch_pairs],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Predict
                    outputs = self.model(**encoded)
                    logits = outputs.logits
                    
                    # Get probabilities for positive class
                    probs = torch.softmax(logits, dim=1)
                    
                    if self.num_labels == 2:
                        # Binary: probability of class 1
                        batch_scores = probs[:, 1].cpu().numpy()
                    else:
                        # Multiclass: weighted probability
                        # Score = 0*P(0) + 0.5*P(1) + 1.0*P(2)
                        weights = torch.tensor([0.0, 0.5, 1.0]).to(self.device)
                        batch_scores = (probs * weights).sum(dim=1).cpu().numpy()
                    
                    scores.extend(batch_scores)
            
            # Combine and sort
            results = list(zip(valid_candidates, scores))
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(
                f"BERT classification complete: "
                f"top score={results[0][1]:.4f}"
            )
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in BERT retrieval: {e}")
            return []
    
    def save_model(self, path: str) -> None:
        """Save fine-tuned model to disk."""
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        logger.info(f"BERT model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load fine-tuned model from disk."""
        logger.info(f"Loading BERT model from {path}")
        
        self.model = BertForSequenceClassification.from_pretrained(
            path
        ).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        
        self.is_trained = True
        logger.info("BERT model loaded")