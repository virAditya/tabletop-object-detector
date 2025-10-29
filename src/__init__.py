"""
Tabletop Object Detection System
Classical computer vision pipeline for object analysis
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .capture import ImageCapture
from .preprocess import ImagePreprocessor
from .segmentation import ObjectSegmenter
from .features import FeatureExtractor
from .visualization import ResultVisualizer
from .logger import DataLogger

__all__ = [
    'ImageCapture',
    'ImagePreprocessor',
    'ObjectSegmenter',
    'FeatureExtractor',
    'ResultVisualizer',
    'DataLogger'
]
