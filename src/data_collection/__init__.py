"""
Data Collection Module for GLBA Violation Detection

This module contains tools and utilities for collecting, processing, and managing
training data for the GLBA DistilBERT model. It includes scrapers, data processors,
and utilities for handling legal and financial documents.

Key Components:
- Web scrapers for legal databases (Westlaw, Nexis Uni, etc.)
- PDF processors for regulatory documents
- Data validation and quality checking
- Dataset management utilities
"""

from .westlaw_scraper import WestlawScraper
from .nexisuni_scraper import NexisUniScraper

__all__ = [
    'WestlawScraper',
    'NexisUniScraper'
]

__version__ = '1.0.0'