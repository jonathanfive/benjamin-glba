import torch
import torch.nn as nn
from transformers import (
    DistilBertModel, 
    DistilBertTokenizer, 
    DistilBertConfig,
    AutoModel,
    AutoTokenizer
)
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class GLBADistilBertClassifier(nn.Module):
    """
    Domain-specific DistilBERT model for GLBA privacy violations detection.
    
    Designed to classify text for:
    - Gramm-Leach-Bliley Act privacy violations
    - Safeguards Rule violations  
    - Pretexting Rule violations
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 4,  # No violation, GLBA, Safeguards, Pretexting
        dropout_rate: float = 0.3,
        hidden_dim: int = 768,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
        # Domain-specific feature extraction layers
        self.privacy_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract sequence output and pooled output
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # Apply privacy-specific attention
        privacy_attended, _ = self.privacy_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Pool the attended features (mean pooling)
        pooled_output = privacy_attended.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.domain_classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled_output
        }
    
    def predict(
        self,
        texts: Union[str, List[str]],
        tokenizer: DistilBertTokenizer,
        device: str = 'cpu',
        max_length: int = 512
    ) -> Dict[str, Union[List[int], List[float]]]:
        """
        Predict GLBA violations for given texts.
        """
        self.eval()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return {
            'predictions': predictions.cpu().tolist(),
            'probabilities': probabilities.cpu().tolist(),
            'labels': ['No Violation', 'GLBA Violation', 'Safeguards Violation', 'Pretexting Violation']
        }
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention weights for interpretability.
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Get attention weights from last layer
            attention_weights = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
            
        return attention_weights


class GLBATokenizer:
    """
    Custom tokenizer wrapper for GLBA domain-specific preprocessing.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Add domain-specific tokens
        domain_tokens = [
            '[GLBA]', '[SAFEGUARDS]', '[PRETEXTING]', '[PII]', '[PHI]',
            '[FINANCIAL_INFO]', '[CUSTOMER_DATA]', '[PRIVACY_NOTICE]'
        ]
        
        self.tokenizer.add_tokens(domain_tokens)
        self.domain_tokens = domain_tokens
        
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def get_vocab_size(self) -> int:
        return len(self.tokenizer)
    
    def preprocess_glba_text(self, text: str) -> str:
        """
        Preprocess text with domain-specific token insertion.
        """
        # Common GLBA-related terms to highlight
        glba_terms = {
            'nonpublic personal information': '[PII] nonpublic personal information [/PII]',
            'customer information': '[CUSTOMER_DATA] customer information [/CUSTOMER_DATA]',
            'privacy notice': '[PRIVACY_NOTICE] privacy notice [/PRIVACY_NOTICE]',
            'safeguards rule': '[SAFEGUARDS] safeguards rule [/SAFEGUARDS]',
            'pretexting': '[PRETEXTING] pretexting [/PRETEXTING]',
            'financial information': '[FINANCIAL_INFO] financial information [/FINANCIAL_INFO]'
        }
        
        processed_text = text.lower()
        for term, replacement in glba_terms.items():
            processed_text = processed_text.replace(term, replacement)
            
        return processed_text


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 4,
    device: str = 'cpu'
) -> Tuple[GLBADistilBertClassifier, GLBATokenizer]:
    """
    Load pre-trained GLBA model and tokenizer.
    """
    tokenizer = GLBATokenizer(model_name)
    
    if model_path:
        # Load fine-tuned model
        model = GLBADistilBertClassifier(
            model_name=model_name,
            num_labels=num_labels
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        # Load base model
        model = GLBADistilBertClassifier(
            model_name=model_name,
            num_labels=num_labels
        )
        logger.info("Loaded base DistilBERT model")
    
    # Resize model embeddings if tokenizer was extended
    model.distilbert.resize_token_embeddings(len(tokenizer.tokenizer))
    
    model.to(device)
    return model, tokenizer