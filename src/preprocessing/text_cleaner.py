import re
import string
from typing import List, Optional, Dict, Tuple, Set
import unicodedata
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class GLBATextCleaner:
    """
    Text preprocessing and cleaning specifically designed for GLBA documents.
    
    Handles legal document formatting, regulatory text normalization,
    and domain-specific preprocessing for financial services compliance text.
    """
    
    def __init__(
        self,
        preserve_legal_structure: bool = True,
        normalize_financial_terms: bool = True,
        preserve_citations: bool = True,
        max_length: Optional[int] = None
    ):
        self.preserve_legal_structure = preserve_legal_structure
        self.normalize_financial_terms = normalize_financial_terms
        self.preserve_citations = preserve_citations
        self.max_length = max_length
        
        # Initialize patterns and mappings
        self._setup_patterns()
        self._setup_financial_term_mappings()
        self._setup_legal_abbreviations()
    
    def _setup_patterns(self):
        """Setup regex patterns for text cleaning."""
        
        # Citation patterns (preserve these if preserve_citations=True)
        self.citation_patterns = [
            r'\d+\s+U\.S\.C\.\s*§?\s*\d+(?:\([a-z]\))?',  # USC citations
            r'\d+\s+C\.F\.R\.\s*§?\s*\d+(?:\.\d+)*',      # CFR citations
            r'§\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))?',       # Section references
            r'Pub\.\s*L\.\s*No\.\s*\d+-\d+',              # Public Law
            r'\d+\s+Fed\.\s*Reg\.\s*\d+',                 # Federal Register
        ]
        
        # Legal document structure patterns
        self.structure_patterns = [
            r'^(?:SECTION|SEC\.|§)\s+\d+',                 # Section headers
            r'^(?:[A-Z]+\.?\s*){1,3}(?:DEFINITIONS?|REQUIREMENTS?|PROCEDURES?)',  # Headings
            r'^\([a-z0-9]+\)\s+',                          # Subsection markers
            r'^\d+\.\s+',                                  # Numbered lists
        ]
        
        # Patterns to clean/normalize
        self.cleanup_patterns = [
            (r'\s*\n\s*\n\s*', '\n\n'),                   # Multiple newlines
            (r'\s*\n\s*', ' '),                           # Single newlines to spaces
            (r'\s+', ' '),                                # Multiple spaces
            (r'([.!?])\s*([A-Z])', r'\1 \2'),            # Proper sentence spacing
            (r'\s*([,;:.])\s*', r'\1 '),                  # Punctuation spacing
        ]
        
        # Common OCR/digitization errors in legal documents
        self.ocr_corrections = [
            (r'\bI\b(?=\s+[a-z])', 'l'),                  # Capital I -> lowercase l
            (r'\b0(?=\s*[a-z])', 'o'),                    # Zero -> letter o
            (r'(?<=[a-z])\s+1\s+(?=[a-z])', ' l '),       # 1 -> l in middle of words
            (r'\bm(?=\s+the\b)', 'in'),                   # m -> in before 'the'
        ]
        
        # Financial document specific patterns
        self.financial_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',                      # Dollar amounts
            r'\d{1,3}(?:,\d{3})*(?:\.\d+)?%?',            # Numbers with commas
            r'\d{2}/\d{2}/\d{4}',                         # Dates MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',                     # Dates MM-DD-YYYY
        ]
    
    def _setup_financial_term_mappings(self):
        """Setup mappings for financial term normalization."""
        
        self.financial_mappings = {
            # Common abbreviations and variations
            'npi': 'nonpublic personal information',
            'n.p.i.': 'nonpublic personal information',
            'cust info': 'customer information',
            'customer info': 'customer information',
            'acct': 'account',
            'w/': 'with',
            'w/o': 'without',
            
            # GLBA specific terms
            'gramm leach bliley act': 'gramm-leach-bliley act',
            'gramm-leach bliley act': 'gramm-leach-bliley act',
            'privacy regs': 'privacy regulations',
            'safeguards regs': 'safeguards regulations',
            
            # Financial institution types
            'fin inst': 'financial institution',
            'financial inst': 'financial institution',
            'credit union': 'credit union',
            'broker dealer': 'broker-dealer',
            'investment adviser': 'investment advisor',
            
            # Common legal abbreviations
            'et al': 'et al.',
            'i.e.': 'that is',
            'e.g.': 'for example',
            'cf.': 'compare',
            'v.': 'versus',
            'vs.': 'versus',
        }
    
    def _setup_legal_abbreviations(self):
        """Setup common legal abbreviations for preservation."""
        
        self.legal_abbreviations = {
            'U.S.C.': 'United States Code',
            'C.F.R.': 'Code of Federal Regulations',
            'Fed. Reg.': 'Federal Register',
            'Pub. L.': 'Public Law',
            'et seq.': 'et seq.',
            'supra': 'supra',
            'infra': 'infra',
            'id.': 'id.',
            'ibid.': 'ibid.',
        }
    
    def _preserve_special_elements(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract and preserve special elements (citations, financial data) 
        before cleaning, then restore them after.
        """
        
        preserved = {}
        placeholder_counter = 0
        
        # Preserve citations if required
        if self.preserve_citations:
            for pattern in self.citation_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    placeholder = f"__CITATION_{placeholder_counter}__"
                    preserved[placeholder] = match.group()
                    text = text.replace(match.group(), placeholder, 1)
                    placeholder_counter += 1
        
        # Preserve financial data patterns
        for pattern in self.financial_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                placeholder = f"__FINANCIAL_{placeholder_counter}__"
                preserved[placeholder] = match.group()
                text = text.replace(match.group(), placeholder, 1)
                placeholder_counter += 1
        
        # Preserve legal structure if required
        if self.preserve_legal_structure:
            for pattern in self.structure_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE)
                for match in matches:
                    placeholder = f"__STRUCTURE_{placeholder_counter}__"
                    preserved[placeholder] = match.group()
                    text = text.replace(match.group(), placeholder, 1)
                    placeholder_counter += 1
        
        return text, preserved
    
    def _restore_special_elements(self, text: str, preserved: Dict[str, str]) -> str:
        """Restore preserved special elements."""
        
        for placeholder, original in preserved.items():
            text = text.replace(placeholder, original)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters and remove non-printable characters."""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-printable characters but preserve common legal symbols
        legal_symbols = {'§', '¶', '©', '®', '"', '°', '±', '×', '÷'}
        
        cleaned_chars = []
        for char in text:
            if (char.isprintable() or 
                char in legal_symbols or 
                char in string.whitespace):
                cleaned_chars.append(char)
        
        return ''.join(cleaned_chars)
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in digitized legal documents."""
        
        for pattern, replacement in self.ocr_corrections:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _normalize_financial_terms(self, text: str) -> str:
        """Normalize financial and legal terminology."""
        
        if not self.normalize_financial_terms:
            return text
        
        # Apply term mappings
        for abbrev, full_term in self.financial_mappings.items():
            # Word boundary matching to avoid partial replacements
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_term, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_cleanup_patterns(self, text: str) -> str:
        """Apply general cleanup patterns."""
        
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def _handle_special_characters(self, text: str) -> str:
        """Handle special characters common in legal documents."""
        
        # Common replacements for legal documents
        replacements = {
            '"': '"',    # Smart quotes
            '"': '"',
            ''': "'",
            ''': "'",
            '&': '...',  # Ellipsis
            '': '-',    # En dash
            '': '--',   # Em dash
            '°': ' degrees ',
            '±': ' plus or minus ',
        }
        
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)
        
        return text
    
    def _truncate_if_needed(self, text: str) -> str:
        """Truncate text if it exceeds max_length."""
        
        if self.max_length and len(text) > self.max_length:
            # Try to truncate at sentence boundary
            truncated = text[:self.max_length]
            
            # Find last sentence ending
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            
            last_sentence_end = max(last_period, last_exclamation, last_question)
            
            if last_sentence_end > self.max_length * 0.8:  # If within 80% of limit
                text = truncated[:last_sentence_end + 1]
            else:
                text = truncated
                
            logger.debug(f"Text truncated from {len(text)} to {len(truncated)} characters")
        
        return text
    
    def clean(self, text: str) -> str:
        """
        Main cleaning function that applies all preprocessing steps.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and preprocessed text
        """
        
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Preserve special elements
        text, preserved = self._preserve_special_elements(text)
        
        # Step 2: Normalize unicode
        text = self._normalize_unicode(text)
        
        # Step 3: Fix OCR errors
        text = self._fix_ocr_errors(text)
        
        # Step 4: Handle special characters
        text = self._handle_special_characters(text)
        
        # Step 5: Normalize financial terms
        text = self._normalize_financial_terms(text)
        
        # Step 6: Apply cleanup patterns
        text = self._apply_cleanup_patterns(text)
        
        # Step 7: Restore special elements
        text = self._restore_special_elements(text, preserved)
        
        # Step 8: Truncate if needed
        text = self._truncate_if_needed(text)
        
        return text
    
    def clean_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of texts to clean
            show_progress: Whether to show progress
            
        Returns:
            List of cleaned texts
        """
        
        if show_progress:
            logger.info(f"Cleaning {len(texts)} texts...")
        
        cleaned_texts = []
        
        for i, text in enumerate(texts):
            if show_progress and i > 0 and i % 1000 == 0:
                logger.info(f"Cleaned {i}/{len(texts)} texts")
            
            cleaned_text = self.clean(text)
            cleaned_texts.append(cleaned_text)
        
        if show_progress:
            logger.info(f"Batch cleaning complete: {len(cleaned_texts)} texts processed")
        
        return cleaned_texts
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        output_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Clean text column in a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to clean
            output_column: Name of output column (default: overwrite input column)
            
        Returns:
            DataFrame with cleaned text
        """
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        output_col = output_column or text_column
        
        logger.info(f"Cleaning text in column '{text_column}'...")
        
        df[output_col] = df[text_column].apply(self.clean)
        
        # Log cleaning statistics
        original_lengths = df[text_column].str.len()
        cleaned_lengths = df[output_col].str.len()
        
        logger.info(
            f"Cleaning complete. Average length: "
            f"{original_lengths.mean():.1f} -> {cleaned_lengths.mean():.1f} characters"
        )
        
        return df
    
    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, any]:
        """Get statistics about the cleaning process."""
        
        stats = {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'length_reduction': len(original_text) - len(cleaned_text),
            'reduction_percentage': (
                (len(original_text) - len(cleaned_text)) / len(original_text) * 100
                if len(original_text) > 0 else 0
            ),
            'original_word_count': len(original_text.split()),
            'cleaned_word_count': len(cleaned_text.split()),
        }
        
        return stats


def create_legal_text_cleaner(
    preserve_citations: bool = True,
    preserve_structure: bool = True,
    normalize_terms: bool = True,
    max_length: Optional[int] = None
) -> GLBATextCleaner:
    """
    Create a text cleaner optimized for legal documents.
    
    Args:
        preserve_citations: Whether to preserve legal citations
        preserve_structure: Whether to preserve document structure
        normalize_terms: Whether to normalize financial terms
        max_length: Maximum text length (None for no limit)
        
    Returns:
        Configured GLBATextCleaner instance
    """
    
    return GLBATextCleaner(
        preserve_legal_structure=preserve_structure,
        normalize_financial_terms=normalize_terms,
        preserve_citations=preserve_citations,
        max_length=max_length
    )


def create_training_text_cleaner(max_length: int = 512) -> GLBATextCleaner:
    """
    Create a text cleaner optimized for model training.
    
    Args:
        max_length: Maximum sequence length for model
        
    Returns:
        Configured GLBATextCleaner instance for training
    """
    
    return GLBATextCleaner(
        preserve_legal_structure=False,  # Remove structure for training
        normalize_financial_terms=True,
        preserve_citations=False,        # Remove citations for training
        max_length=max_length
    )