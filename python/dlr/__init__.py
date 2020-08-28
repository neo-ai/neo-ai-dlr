# coding: utf-8
"""
DLR: Compact Runtime for Machine Learning Models
"""

from __future__ import absolute_import as _abs

import sys
sys.path.append("..")

from .api import DLRModel
from meta import NAME, VERSION

__all__ = NAME

__version__ = VERSION
