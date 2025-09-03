"""
Database module for the AI Watermark Detection Tool.

This module provides database connectivity, models, and operations
for storing analysis results, test datasets, and application data.
"""

from .connection import DatabaseManager, get_database, initialize_database, cleanup_database
from .models import AnalysisResultModel, TestDatasetModel
from .operations import AnalysisResultOperations, TestDatasetOperations

__all__ = [
    "DatabaseManager",
    "get_database",
    "initialize_database",
    "cleanup_database",
    "AnalysisResultModel",
    "TestDatasetModel", 
    "AnalysisResultOperations",
    "TestDatasetOperations"
]