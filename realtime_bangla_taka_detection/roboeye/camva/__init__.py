"""CAMVA: Confidence-Aware Adaptive Multi-View Authentication.

New research module. Does not replace roboeye.authenticity.
"""

from .model import CAMVANet
from .notes import assert_disjoint_splits, build_and_save_splits, load_splits

__all__ = ["CAMVANet", "assert_disjoint_splits", "build_and_save_splits", "load_splits"]
