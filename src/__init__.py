"""
ML-NIDS: Machine Learning-based Network Intrusion Detection System

A comprehensive package for detecting network intrusions using machine learning.
"""

__version__ = "0.1.1"
__author__ = "Adam Higazi"
__email__ = "higaziadam03@gmail.com"

# Import main modules for easier access
from src.config import CONFIG
from src.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

__all__ = [
    "CONFIG",
    "logger",
]
