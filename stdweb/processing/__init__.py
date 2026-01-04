"""
Image processing pipeline for STDWeb.

This package provides the core image processing functionality including:
- Image inspection and parameter extraction
- Photometric calibration
- Simple transient detection
- Template subtraction
- Image stacking

For backward compatibility, all public functions and constants are re-exported
from this module.
"""

# Constants
from .constants import *
# Utility functions
from .utils import *
# Catalog functions
from .catalogs import *

# Main processing functions
from .inspect import *
from .photometry import *
from .transients import *
from .subtraction import *
from .stacking import *

# Disable some annoying warnings from astropy
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)
