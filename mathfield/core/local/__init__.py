"""
mathfield/core/local/

Stage 3: Local Decomposition & Subdivision.
"""

from mathfield.core.local.local_patch import LocalPatch
from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
from mathfield.core.local.quadtree import QuadtreeDecomposition, QuadtreeNode

__all__ = [
    "QuadtreeDecomposition",
    "QuadtreeNode",
    "WendlandPartitionOfUnity",
    "LocalPatch",
]
