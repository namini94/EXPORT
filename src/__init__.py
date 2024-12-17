"""
SVEGA (Supervised Variational Encoder for Gene Activity)

SVEGA (EXPORT) is a deep learning framework for analyzing gene expression data with pathway annotations.
It uses a variational autoencoder architecture with a customized decoder that incorporates
prior biological knowledge in the form of pathway annotations.

The framework supports:
- Gene expression analysis
- Pathway activity inference
- Differential expression analysis
- Ordinal regression for dose-response studies
"""

# Version of the vega package
__version__ = "1.0.0"

# List of maintainers
__maintainer__ = "Nami Niyakan"
__email__ = "naminiyakan@tamu.edu"

# Import main classes and functions for easy access
from .models import SVEGA, DecoderVEGA, DecoderVEGACount
from .datasets import RadbioDataset
from .utils.preprocessing import create_mask
from .utils.plotting import volcano

# Define public interface
__all__ = [
    # Main model classes
    'SVEGA',
    'DecoderVEGA',
    'DecoderVEGACount',
    
    # Dataset classes
    'RadbioDataset',
    
    # Utility functions
    'create_mask',
    'volcano',
]

def get_version():
    """Return the version of VEGA package."""
    return __version__
