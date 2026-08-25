from pathlib import Path

import pytest

from apexoracle_evo2.windowing import discover_fasta_files, iter_fasta_windows


def write_fasta(path: Path) -> None:
    path.write_text(">contig_a\nACGTACGTACGT\n>contig_b\nTTTTAAA\n", encoding="utf-8")


def test_multi_record_window_order_and_provenance(tmp_path: Path) -> None:
    fasta = tmp_path / "genome.fasta"
    write_fasta(fasta)

    windows = list(iter_fasta_windows(fasta, chunk_length=5, step_length=4))

    assert [window.record_id for window in windows] == [
        "contig_a",
        "contig_a",
        "contig_a",
        "contig_b",
        "contig_b",
    ]
    assert [(window.start, window.end) for window in windows] == [
        (0, 5),
        (4, 9),
        (8, 12),
        (0, 5),
        (4, 7),
    ]
    assert [window.record_window_index for window in windows] == [0, 1, 2, 0, 1]
    assert [window.window_index for window in windows] == list(range(5))
    assert [window.is_partial for window in windows] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert [window.sequence for window in windows] == [
        "ACGTA",
        "ACGTA",
        "ACGT",
        "TTTTA",
        "AAA",
    ]


def test_full_windows_only_is_applied_per_record(tmp_path: Path) -> None:
    fasta = tmp_path / "genome.fna"
    write_fasta(fasta)

    windows = list(
        iter_fasta_windows(fasta, chunk_length=5, step_length=4, include_partial=False)
    )

    assert [(window.record_id, window.start, window.end) for window in windows] == [
        ("contig_a", 0, 5),
        ("contig_a", 4, 9),
        ("contig_b", 0, 5),
    ]


def test_each_record_resets_its_start_after_a_long_first_contig(tmp_path: Path) -> None:
    """Guard against the historical producer's cross-record global counter bug."""

    fasta = tmp_path / "multi_contig.fasta"
    fasta.write_text(
        ">long\n" + "A" * 21_500 + "\n>short\n" + "C" * 500 + "\n",
        encoding="utf-8",
    )

    windows = list(iter_fasta_windows(fasta))

    assert [(window.record_id, window.start, window.end) for window in windows] == [
        ("long", 0, 11_000),
        ("long", 10_000, 21_000),
        ("long", 20_000, 21_500),
        ("short", 0, 500),
    ]
    assert [window.record_window_index for window in windows] == [0, 1, 2, 0]
    assert [window.window_index for window in windows] == [0, 1, 2, 3]


def test_discovery_is_sorted_and_rejects_stem_collisions(tmp_path: Path) -> None:
    (tmp_path / "b.fna").write_text(">b\nAC\n", encoding="utf-8")
    (tmp_path / "a.fasta").write_text(">a\nGT\n", encoding="utf-8")
    assert [path.name for path in discover_fasta_files(tmp_path)] == [
        "a.fasta",
        "b.fna",
    ]

    (tmp_path / "a.fa").write_text(">a2\nAA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collision"):
        discover_fasta_files(tmp_path)


def test_empty_fasta_is_rejected(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        list(iter_fasta_windows(fasta))
