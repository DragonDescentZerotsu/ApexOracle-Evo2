"""Deterministic FASTA windowing with record-level provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from Bio import SeqIO

FASTA_SUFFIXES = frozenset({".fa", ".fasta", ".fna", ".ffn", ".fas"})
WINDOW_INDEXING_CONTRACT = "per_record_zero_based_v1"


@dataclass(frozen=True)
class FastaWindow:
    """One ordered sequence window from one FASTA record."""

    record_id: str
    record_index: int
    record_window_index: int
    window_index: int
    start: int
    end: int
    record_length: int
    sequence: str
    is_partial: bool

    def provenance(self) -> dict[str, int | str | bool]:
        """Return JSON-serializable provenance without duplicating sequence payloads."""

        values = asdict(self)
        values.pop("sequence")
        return values


def _validate_window_parameters(chunk_length: int, step_length: int) -> None:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be a positive integer")
    if step_length <= 0:
        raise ValueError("step_length must be a positive integer")


def iter_fasta_windows(
    fasta_path: str | Path,
    *,
    chunk_length: int = 11_000,
    step_length: int = 10_000,
    include_partial: bool = True,
) -> Iterator[FastaWindow]:
    """Yield windows in FASTA record order and ascending start order.

    Window coordinates are zero-based half-open intervals within each FASTA record.
    Each record starts its own coordinate system at zero. By default, a terminal
    shorter-than-``chunk_length`` window is retained whenever its start is inside the
    record. The per-record start reset is intentional: a historical reviewer producer
    used one global counter across records and could omit later contigs, but that indexing
    is not valid for new embedding artifacts.
    """

    _validate_window_parameters(chunk_length, step_length)
    path = Path(fasta_path)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file does not exist: {path}")

    record_count = 0
    window_index = 0
    for record_index, record in enumerate(SeqIO.parse(path, "fasta")):
        record_count += 1
        sequence = str(record.seq).upper()
        record_length = len(sequence)
        record_window_index = 0

        for start in range(0, record_length, step_length):
            end = min(start + chunk_length, record_length)
            is_partial = end - start < chunk_length
            if is_partial and not include_partial:
                break

            yield FastaWindow(
                record_id=str(record.id),
                record_index=record_index,
                record_window_index=record_window_index,
                window_index=window_index,
                start=start,
                end=end,
                record_length=record_length,
                sequence=sequence[start:end],
                is_partial=is_partial,
            )
            record_window_index += 1
            window_index += 1

    if record_count == 0:
        raise ValueError(f"FASTA file contains no records: {path}")


def discover_fasta_files(input_path: str | Path) -> Sequence[Path]:
    """Resolve one FASTA file or a flat directory of FASTA files deterministically."""

    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in FASTA_SUFFIXES:
            raise ValueError(f"unsupported FASTA suffix for file: {path}")
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")

    files = tuple(
        sorted(
            (
                candidate
                for candidate in path.iterdir()
                if candidate.is_file() and candidate.suffix.lower() in FASTA_SUFFIXES
            ),
            key=lambda candidate: candidate.name,
        )
    )
    if not files:
        raise ValueError(f"input directory contains no supported FASTA files: {path}")

    stems = [candidate.stem for candidate in files]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        raise ValueError(
            "output filename collision for FASTA stems: " + ", ".join(duplicate_stems)
        )
    return files
