"""
Evaluation Module for GLBA Violation Detection

This module provides comprehensive evaluation tools and metrics for the
GLBA DistilBERT model, including standard ML metrics and domain-specific
evaluation measures for financial compliance applications.

Key Components:
- GLBAMetrics: Comprehensive metrics computation
- Visualization tools for model performance
- Domain-specific evaluation criteria
- Report generation utilities
"""

from .metrics import GLBAMetrics, create_glba_metrics

__all__ = [
    'GLBAMetrics',
    'create_glba_metrics'
]

__version__ = '1.0.0'