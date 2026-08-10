import json
from pathlib import Path

from apexoracle_evo2.cli import main


def test_plan_only_reports_all_records_without_loading_model(
    tmp_path: Path, capsys
) -> None:
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(">one\nACGTAC\n>two\nTTTT\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"

    exit_code = main(
        [
            "--input",
            str(fasta),
            "--output-dir",
            str(output_dir),
            "--chunk-length",
            "4",
            "--step-length",
            "3",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["file_count"] == 1
    assert payload["record_count"] == 2
    assert payload["window_count"] == 4
    assert "files" not in payload
    assert not output_dir.exists()


def test_plan_file_detail_is_opt_in(tmp_path: Path, capsys) -> None:
    fasta = tmp_path / "genome.fna"
    fasta.write_text(">one\nACGT\n", encoding="utf-8")

    main(
        [
            "--input",
            str(fasta),
            "--output-dir",
            str(tmp_path / "unused"),
            "--chunk-length",
            "3",
            "--step-length",
            "2",
            "--plan-only",
            "--plan-detail",
            "files",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["files"] == [
        {"name": "genome.fna", "record_count": 1, "window_count": 2}
    ]
