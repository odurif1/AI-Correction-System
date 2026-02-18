"""
Analysis module for cross-copy analysis and clustering.

Provides embedding-based clustering and pattern detection.
"""

from analysis.clustering import EmbeddingClustering
from analysis.cross_copy import CrossCopyAnalyzer

__all__ = [
    'EmbeddingClustering',
    'CrossCopyAnalyzer',
]
