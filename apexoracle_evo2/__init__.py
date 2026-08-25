"""ApexOracle utilities built on the Evo 2 runtime."""

from .windowing import (
    WINDOW_INDEXING_CONTRACT,
    FastaWindow,
    discover_fasta_files,
    iter_fasta_windows,
)

__all__ = [
    "WINDOW_INDEXING_CONTRACT",
    "FastaWindow",
    "discover_fasta_files",
    "iter_fasta_windows",
]
