import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .benjamin_glba import GLBADistilBertClassifier, GLBATokenizer
from ..evaluation.metrics import GLBAMetrics
from ..preprocessing.text_cleaner import GLBATextCleaner

logger = logging.getLogger(__name__)


class GLBADataset(Dataset):
    """
    Dataset class for GLBA violation detection training.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: GLBATokenizer,
        max_length: int = 512,
        text_cleaner: Optional[GLBATextCleaner] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_cleaner = text_cleaner
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Clean text if cleaner provided
        if self.text_cleaner:
            text = self.text_cleaner.clean(text)
        
        # Preprocess with domain-specific tokens
        text = self.tokenizer.preprocess_glba_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class GLBATrainer:
    """
    Trainer class for GLBA DistilBERT model.
    """
    
    def __init__(
        self,
        model: GLBADistilBertClassifier,
        tokenizer: GLBATokenizer,
        config: Dict[str, Any],
        device: str = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize components
        self.metrics = GLBAMetrics()
        self.text_cleaner = GLBATextCleaner() if config.get('use_text_cleaner', True) else None
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.train_losses = []
        self.val_losses = []
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup logging
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'glba-distilbert'),
                config=config
            )
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        
        # Different learning rates for BERT and classifier layers
        bert_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'distilbert' in name:
                bert_params.append(param)
            else:
                classifier_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                'params': bert_params,
                'lr': self.config.get('bert_learning_rate', 2e-5),
                'weight_decay': self.config.get('weight_decay', 0.01)
            },
            {
                'params': classifier_params,
                'lr': self.config.get('classifier_learning_rate', 1e-4),
                'weight_decay': self.config.get('weight_decay', 0.01)
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.config.get('adam_epsilon', 1e-8)
        )
        
    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        
        scheduler_type = self.config.get('scheduler_type', 'linear_warmup')
        
        if scheduler_type == 'linear_warmup':
            num_warmup_steps = int(0.1 * num_training_steps)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=2,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps
            )
        else:
            self.scheduler = None
    
    def prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation dataloaders.
        """
        
        # Split data if validation set not provided
        if val_df is None:
            train_df, val_df = train_test_split(
                train_df,
                test_size=test_size,
                stratify=train_df['label'],
                random_state=42
            )
        
        # Create datasets
        train_dataset = GLBADataset(
            texts=train_df['text'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config.get('max_length', 512),
            text_cleaner=self.text_cleaner
        )
        
        val_dataset = GLBADataset(
            texts=val_df['text'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config.get('max_length', 512),
            text_cleaner=self.text_cleaner
        )
        
        # Calculate class weights for imbalanced data
        class_weights = None
        if self.config.get('use_class_weights', True):
            labels = train_df['label'].values
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(labels),
                y=labels
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        self.class_weights = class_weights
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            
            # Apply class weights if available
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(
                    outputs['logits'].view(-1, self.model.num_labels),
                    batch['labels'].view(-1)
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            
            if self.scheduler and self.config.get('scheduler_type') != 'reduce_on_plateau':
                self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                if self.class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fct(
                        outputs['logits'].view(-1, self.model.num_labels),
                        batch['labels'].view(-1)
                    )
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # Calculate metrics
        metrics = self.metrics.compute_metrics(all_labels, all_predictions)
        metrics['val_loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        """
        
        num_epochs = num_epochs or self.config.get('num_epochs', 5)
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        self._setup_scheduler(total_steps)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate scheduler
            if self.scheduler and self.config.get('scheduler_type') == 'reduce_on_plateau':
                self.scheduler.step(val_metrics['f1_weighted'])
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val F1: {val_metrics['f1_weighted']:.4f}"
            )
            
            # Update training history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_metrics['val_loss'])
            training_history['val_f1'].append(val_metrics['f1_weighted'])
            training_history['val_precision'].append(val_metrics['precision_weighted'])
            training_history['val_recall'].append(val_metrics['recall_weighted'])
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    **val_metrics
                })
            
            # Save best model
            current_score = val_metrics['f1_weighted']
            if current_score > self.best_score:
                self.best_score = current_score
                self.save_model(
                    Path(self.config.get('output_dir', 'models')) / 'best_model.pt',
                    metrics=val_metrics
                )
            
            # Early stopping
            if self._should_early_stop(val_metrics['f1_weighted']):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        return training_history
    
    def _should_early_stop(self, current_score: float) -> bool:
        """Check if training should stop early."""
        
        patience = self.config.get('early_stopping_patience', 3)
        if len(self.val_losses) < patience:
            return False
        
        # Check if validation loss hasn't improved
        recent_losses = self.val_losses[-patience:]
        return all(loss >= min(self.val_losses) for loss in recent_losses)
    
    def save_model(
        self,
        path: Path,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Save model checkpoint."""
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.current_epoch,
            'best_score': self.best_score,
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
        
        # Save config separately
        config_path = path.parent / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_model(self, path: Path):
        """Load model checkpoint."""
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_score = checkpoint.get('best_score', 0.0)
        
        logger.info(f"Model loaded from {path}")
        return checkpoint.get('metrics', {})


def create_trainer_config(
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 5,
    max_length: int = 512,
    use_wandb: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Create training configuration.
    """
    
    config = {
        'batch_size': batch_size,
        'bert_learning_rate': learning_rate,
        'classifier_learning_rate': learning_rate * 5,  # Higher LR for classifier
        'num_epochs': num_epochs,
        'max_length': max_length,
        'weight_decay': 0.01,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_ratio': 0.1,
        'scheduler_type': 'linear_warmup',
        'early_stopping_patience': 3,
        'use_class_weights': True,
        'use_text_cleaner': True,
        'use_wandb': use_wandb,
        'wandb_project': 'glba-distilbert',
        'num_workers': 4,
        'output_dir': 'models',
        **kwargs
    }
    
    return config