"""ComfyUI nodes for SoulX-Singer - Zero-Shot Singing Voice Synthesis.

This package provides ComfyUI integration for SoulX-Singer model.
"""

# Set NLTK data directory BEFORE any imports to avoid AppData/Roaming
import os
import sys
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.environ['NLTK_DATA'] = os.path.abspath(nltk_data_dir)

# Suppress verbose logging from dependencies
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__version__ = "0.8.0"
__author__ = "Saganaki22"

import logging
import sys
from typing import Dict, Any

# Setup logging
logger = logging.getLogger("SoulX-Singer")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[SoulX-Singer] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def ensure_dependencies():
    """Check and report required dependencies."""
    missing_packages = []
    optional_missing = []
    
    # Required dependencies
    try:
        import torch
    except ImportError:
        missing_packages.append("torch>=2.0.0")
    
    try:
        import soundfile
    except ImportError:
        missing_packages.append("soundfile")
    
    try:
        import librosa
    except ImportError:
        missing_packages.append("librosa")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import omegaconf
    except ImportError:
        missing_packages.append("omegaconf")
    
    # Optional dependencies (for preprocessing)
    try:
        import gradio
    except ImportError:
        optional_missing.append("gradio (for preprocessing)")
    
    try:
        import sageattention
    except ImportError:
        optional_missing.append("sageattention (for optimized attention)")
    
    if missing_packages:
        logger.error("=" * 60)
        logger.error("MISSING REQUIRED DEPENDENCIES")
        logger.error("=" * 60)
        for pkg in missing_packages:
            logger.error(f"  - {pkg}")
        logger.error("")
        logger.error("Install with: pip install torch soundfile librosa numpy omegaconf")
        logger.error("=" * 60)
        return False
    
    if optional_missing:
        logger.info("Optional features unavailable:")
        for pkg in optional_missing:
            logger.info(f"  - {pkg}")
    
    return True


# Node mappings - REQUIRED by ComfyUI
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Register nodes if dependencies are available
if ensure_dependencies():
    try:
        from .nodes.model_loader import SoulXSingerModelLoader
        from .nodes.simple_synthesizer import SoulXSingerSimple
        from .nodes.advanced_synthesizer import SoulXSingerAdvanced
        
        NODE_CLASS_MAPPINGS["SoulXSingerModelLoader"] = SoulXSingerModelLoader
        NODE_CLASS_MAPPINGS["SoulXSingerSimple"] = SoulXSingerSimple
        NODE_CLASS_MAPPINGS["SoulXSingerAdvanced"] = SoulXSingerAdvanced
        
        NODE_DISPLAY_NAME_MAPPINGS["SoulXSingerModelLoader"] = "🎤 SoulX-Singer Model Loader"
        NODE_DISPLAY_NAME_MAPPINGS["SoulXSingerSimple"] = "🎙️ SoulX-Singer Simple"
        NODE_DISPLAY_NAME_MAPPINGS["SoulXSingerAdvanced"] = "🎙️ SoulX-Singer Advanced"
        
        logger.info(f"Nodes registered successfully (v{__version__})")
        logger.info(f"Registered nodes: {len(NODE_CLASS_MAPPINGS)}")
        
    except Exception as e:
        logger.error(f"Failed to register nodes: {e}")
        import traceback
        logger.error(traceback.format_exc())
else:
    logger.warning("Nodes unavailable - missing dependencies")

# REQUIRED exports
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']
