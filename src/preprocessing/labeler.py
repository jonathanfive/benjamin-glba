import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import re
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from enum import Enum
import spacy
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """GLBA violation types."""
    NO_VIOLATION = 0
    GLBA_VIOLATION = 1
    SAFEGUARDS_VIOLATION = 2
    PRETEXTING_VIOLATION = 3


@dataclass
class GLBAKeywords:
    """GLBA-related keywords and patterns."""
    
    # Core GLBA terms
    glba_core = [
        'nonpublic personal information', 'npi', 'customer information',
        'privacy notice', 'privacy policy', 'opt out', 'opt-out',
        'gramm leach bliley', 'gramm-leach-bliley', 'glba',
        'financial privacy rule', 'privacy rule'
    ]
    
    # Safeguards Rule terms
    safeguards = [
        'safeguards rule', 'safeguard', 'information security',
        'access controls', 'encryption', 'data security',
        'administrative safeguards', 'physical safeguards',
        'technical safeguards', 'information systems',
        'security incident', 'data breach', 'unauthorized access'
    ]
    
    # Pretexting terms
    pretexting = [
        'pretexting', 'pretext', 'false pretense', 'false pretences',
        'impersonation', 'social engineering', 'fraudulent solicitation',
        'deceptive practice', 'misrepresentation', 'unauthorized disclosure'
    ]
    
    # PII/PHI indicators
    personal_info = [
        'social security number', 'ssn', 'account number', 'credit report',
        'financial record', 'bank account', 'credit card', 'debit card',
        'personal identification', 'customer record', 'transaction history',
        'credit score', 'loan information', 'investment account'
    ]
    
    # Violation indicators
    violation_indicators = [
        'disclosed', 'shared', 'revealed', 'unauthorized', 'breach',
        'violation', 'failed to', 'did not provide', 'without consent',
        'improper', 'inadequate', 'insufficient', 'negligent'
    ]
    
    # Privacy notice requirements
    privacy_notice_terms = [
        'initial privacy notice', 'annual privacy notice',
        'privacy notice delivery', 'notice requirements',
        'customer notification', 'disclosure statement'
    ]


class GLBAPatternMatcher:
    """Pattern matching for GLBA violations."""
    
    def __init__(self):
        self.keywords = GLBAKeywords()
        self.patterns = self._compile_patterns()
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Using basic matching only.")
            self.nlp = None
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for each violation type."""
        
        patterns = {
            'glba': [
                re.compile(r'fail(?:ed|ure)?\s+to\s+(?:provide|deliver|send)\s+.*privacy\s+notice', re.I),
                re.compile(r'(?:disclosed|shared|revealed)\s+.*(?:nonpublic|customer|personal)\s+information', re.I),
                re.compile(r'privacy\s+notice\s+(?:not\s+)?(?:provided|delivered|sent)', re.I),
                re.compile(r'unauthorized\s+(?:disclosure|sharing)\s+of\s+(?:customer|personal)\s+information', re.I)
            ],
            'safeguards': [
                re.compile(r'(?:inadequate|insufficient|lack\s+of)\s+(?:safeguards|security|protection)', re.I),
                re.compile(r'(?:failed|failure)\s+to\s+(?:implement|maintain|establish)\s+safeguards', re.I),
                re.compile(r'(?:data|security)\s+breach\s+(?:due\s+to|caused\s+by)\s+inadequate', re.I),
                re.compile(r'(?:unauthorized\s+access|security\s+incident)\s+due\s+to', re.I),
                re.compile(r'(?:weak|insufficient)\s+(?:encryption|access\s+controls)', re.I)
            ],
            'pretexting': [
                re.compile(r'pretext(?:ing)?\s+(?:to\s+obtain|for\s+obtaining|attempt)', re.I),
                re.compile(r'(?:false\s+pretense|impersonat|fraudulent(?:ly)?)\s+(?:obtain|solicit|request)', re.I),
                re.compile(r'social\s+engineering\s+(?:attack|attempt|to\s+obtain)', re.I),
                re.compile(r'(?:deceptive|fraudulent)\s+(?:practice|solicitation|request)', re.I),
                re.compile(r'misrepresent(?:ed|ing)?\s+(?:identity|purpose)\s+to\s+obtain', re.I)
            ]
        }
        
        return patterns
    
    def match_patterns(self, text: str) -> Dict[str, int]:
        """Match patterns in text and return scores."""
        
        scores = defaultdict(int)
        
        for violation_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                scores[violation_type] += len(matches)
        
        return dict(scores)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract relevant entities using spaCy NER."""
        
        entities = defaultdict(list)
        
        if self.nlp is None:
            return dict(entities)
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:  # Person, Organization, Location
                entities[ent.label_].append(ent.text)
            elif ent.label_ in ['CARDINAL', 'MONEY']:  # Numbers, Money
                entities['FINANCIAL'].append(ent.text)
        
        return dict(entities)


class GLBALabeler:
    """
    Automatic labeler for GLBA violation detection.
    
    Uses rule-based and heuristic approaches to label text data
    for training the GLBA DistilBERT model.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.keywords = GLBAKeywords()
        self.pattern_matcher = GLBAPatternMatcher()
        self.label_stats = defaultdict(int)
        
    def _calculate_keyword_scores(self, text: str) -> Dict[str, float]:
        """Calculate keyword-based scores for each violation type."""
        
        text_lower = text.lower()
        scores = {}
        
        # GLBA violations
        glba_score = sum(1 for keyword in self.keywords.glba_core 
                        if keyword in text_lower)
        privacy_score = sum(1 for keyword in self.keywords.privacy_notice_terms
                           if keyword in text_lower)
        violation_score = sum(1 for keyword in self.keywords.violation_indicators
                             if keyword in text_lower)
        
        scores['glba'] = (glba_score + privacy_score) * (1 + violation_score * 0.5)
        
        # Safeguards violations
        safeguards_score = sum(1 for keyword in self.keywords.safeguards
                              if keyword in text_lower)
        scores['safeguards'] = safeguards_score * (1 + violation_score * 0.5)
        
        # Pretexting violations
        pretexting_score = sum(1 for keyword in self.keywords.pretexting
                              if keyword in text_lower)
        scores['pretexting'] = pretexting_score * (1 + violation_score * 0.5)
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _calculate_pattern_scores(self, text: str) -> Dict[str, float]:
        """Calculate pattern-based scores."""
        
        pattern_scores = self.pattern_matcher.match_patterns(text)
        
        # Normalize scores
        max_score = max(pattern_scores.values()) if pattern_scores.values() else 1
        if max_score > 0:
            pattern_scores = {k: v / max_score for k, v in pattern_scores.items()}
        
        return pattern_scores
    
    def _calculate_context_scores(self, text: str) -> Dict[str, float]:
        """Calculate contextual scores based on text structure and content."""
        
        scores = defaultdict(float)
        
        # Check for legal/regulatory context
        legal_indicators = ['section', 'regulation', 'rule', 'violation', 'compliance',
                           'federal', 'commission', 'enforcement', 'penalty', 'fine']
        legal_score = sum(1 for indicator in legal_indicators if indicator in text.lower())
        
        # Check for financial institution context
        financial_indicators = ['bank', 'credit union', 'financial institution',
                               'broker-dealer', 'investment advisor', 'mortgage']
        financial_score = sum(1 for indicator in financial_indicators if indicator in text.lower())
        
        # Apply context boost
        context_boost = min((legal_score + financial_score) * 0.1, 0.5)
        
        scores['glba'] = context_boost
        scores['safeguards'] = context_boost
        scores['pretexting'] = context_boost
        
        return dict(scores)
    
    def label_text(
        self, 
        text: str,
        return_confidence: bool = False
    ) -> Union[int, Tuple[int, float, Dict[str, float]]]:
        """
        Label a single text sample.
        
        Args:
            text: Input text to label
            return_confidence: If True, return confidence score and breakdown
            
        Returns:
            Label (int) or tuple of (label, confidence, score_breakdown)
        """
        
        if not text or len(text.strip()) < 10:
            return ViolationType.NO_VIOLATION.value
        
        # Calculate different types of scores
        keyword_scores = self._calculate_keyword_scores(text)
        pattern_scores = self._calculate_pattern_scores(text)
        context_scores = self._calculate_context_scores(text)
        
        # Combine scores with weights
        combined_scores = defaultdict(float)
        
        for violation_type in ['glba', 'safeguards', 'pretexting']:
            combined_scores[violation_type] = (
                keyword_scores.get(violation_type, 0) * 0.4 +
                pattern_scores.get(violation_type, 0) * 0.5 +
                context_scores.get(violation_type, 0) * 0.1
            )
        
        # Determine final label
        max_score = max(combined_scores.values()) if combined_scores.values() else 0
        
        if max_score < self.confidence_threshold:
            label = ViolationType.NO_VIOLATION.value
            confidence = 1 - max_score
        else:
            # Find the violation type with highest score
            best_violation = max(combined_scores.items(), key=lambda x: x[1])
            
            if best_violation[0] == 'glba':
                label = ViolationType.GLBA_VIOLATION.value
            elif best_violation[0] == 'safeguards':
                label = ViolationType.SAFEGUARDS_VIOLATION.value
            elif best_violation[0] == 'pretexting':
                label = ViolationType.PRETEXTING_VIOLATION.value
            else:
                label = ViolationType.NO_VIOLATION.value
            
            confidence = best_violation[1]
        
        self.label_stats[label] += 1
        
        if return_confidence:
            return label, confidence, dict(combined_scores)
        else:
            return label
    
    def label_dataset(
        self,
        texts: List[str],
        return_confidence: bool = False
    ) -> Union[List[int], pd.DataFrame]:
        """
        Label a dataset of texts.
        
        Args:
            texts: List of texts to label
            return_confidence: If True, return DataFrame with additional info
            
        Returns:
            List of labels or DataFrame with labels, confidence, and scores
        """
        
        results = []
        
        logger.info(f"Labeling {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
            
            result = self.label_text(text, return_confidence=True)
            label, confidence, scores = result
            
            if return_confidence:
                results.append({
                    'text': text,
                    'label': label,
                    'confidence': confidence,
                    'glba_score': scores.get('glba', 0),
                    'safeguards_score': scores.get('safeguards', 0),
                    'pretexting_score': scores.get('pretexting', 0)
                })
            else:
                results.append(label)
        
        if return_confidence:
            df = pd.DataFrame(results)
            logger.info(f"Labeling complete. Label distribution:\n{df['label'].value_counts()}")
            return df
        else:
            logger.info(f"Labeling complete. Label distribution:\n{Counter(results)}")
            return results
    
    def get_label_statistics(self) -> Dict[str, Any]:
        """Get statistics about labeling results."""
        
        total_labels = sum(self.label_stats.values())
        
        stats = {
            'total_samples': total_labels,
            'distribution': dict(self.label_stats),
            'proportions': {
                label: count / total_labels if total_labels > 0 else 0
                for label, count in self.label_stats.items()
            }
        }
        
        return stats
    
    def filter_by_confidence(
        self,
        df: pd.DataFrame,
        min_confidence: float = 0.7
    ) -> pd.DataFrame:
        """Filter dataset by confidence threshold."""
        
        if 'confidence' not in df.columns:
            raise ValueError("DataFrame must contain 'confidence' column")
        
        filtered_df = df[df['confidence'] >= min_confidence].copy()
        
        logger.info(
            f"Filtered dataset: {len(filtered_df)} / {len(df)} samples "
            f"with confidence >= {min_confidence}"
        )
        
        return filtered_df
    
    def balance_dataset(
        self,
        df: pd.DataFrame,
        max_samples_per_class: Optional[int] = None
    ) -> pd.DataFrame:
        """Balance dataset by undersampling majority classes."""
        
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        
        label_counts = df['label'].value_counts()
        
        if max_samples_per_class is None:
            max_samples_per_class = label_counts.min()
        
        balanced_dfs = []
        
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            
            if len(label_df) > max_samples_per_class:
                # Sample with stratification by confidence if available
                if 'confidence' in df.columns:
                    # Sort by confidence and take top samples
                    label_df = label_df.sort_values('confidence', ascending=False)
                    label_df = label_df.head(max_samples_per_class)
                else:
                    label_df = label_df.sample(max_samples_per_class, random_state=42)
            
            balanced_dfs.append(label_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        logger.info(
            f"Balanced dataset: {len(balanced_df)} samples, "
            f"distribution: {balanced_df['label'].value_counts().to_dict()}"
        )
        
        return balanced_df
    
    def export_labeled_data(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format: str = 'csv'
    ):
        """Export labeled dataset."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Labeled data exported to {output_path}")


def create_labeler(
    confidence_threshold: float = 0.6,
    custom_keywords: Optional[Dict[str, List[str]]] = None
) -> GLBALabeler:
    """
    Create a GLBA labeler with custom configuration.
    
    Args:
        confidence_threshold: Minimum confidence for positive labels
        custom_keywords: Additional custom keywords by violation type
        
    Returns:
        Configured GLBALabeler instance
    """
    
    labeler = GLBALabeler(confidence_threshold)
    
    if custom_keywords:
        for violation_type, keywords in custom_keywords.items():
            if hasattr(labeler.keywords, violation_type):
                existing_keywords = getattr(labeler.keywords, violation_type)
                setattr(labeler.keywords, violation_type, existing_keywords + keywords)
    
    return labeler