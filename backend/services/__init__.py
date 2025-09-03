"""
Services module for the AI Watermark Detection Tool.

This module provides business logic services including watermarking operations,
text processing, and other core functionality.
"""

from .watermark_service import WatermarkService, get_watermark_service

__all__ = [
    "WatermarkService",
    "get_watermark_service"
]