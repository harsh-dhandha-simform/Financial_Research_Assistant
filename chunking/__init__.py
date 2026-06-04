"""
Chonkie-based chunking with parent-child hierarchy.

Usage:
    from chunking import chunk_document

    parents, children = chunk_document(raw_document)
"""

from chunking.chunker import chunk_document

__all__ = ["chunk_document"]
