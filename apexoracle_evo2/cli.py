"""Command-line entry point for ApexOracle genome embedding extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .extraction import extract_window_embeddings
from .windowing import (
    WINDOW_INDEXING_CONTRACT,
    FastaWindow,
    discover_fasta_files,
    iter_fasta_windows,
)

DEFAULT_LAYERS = {
    "evo2_40b": "blocks.46.mlp.l3",
    "evo2_7b": "blocks.28.mlp.l3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract record-aware Evo 2 genome-window embeddings for ApexOracle."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="evo2_40b")
    parser.add_argument("--layer-name")
    parser.add_argument("--chunk-length", type=int, default=11_000)
    parser.add_argument("--step-length", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--input-device", default="cuda:0")
    parser.add_argument(
        "--full-windows-only",
        action="store_true",
        help="omit terminal windows shorter than --chunk-length",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and summarize FASTA windows without loading Evo 2",
    )
    parser.add_argument(
        "--plan-detail",
        choices=("summary", "files"),
        default="summary",
        help="control whether --plan-only includes per-file rows",
    )
    return parser


def _atomic_torch_save(tensor: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _windows_for_file(fasta_path: Path, args: argparse.Namespace) -> list[FastaWindow]:
    windows = list(
        iter_fasta_windows(
            fasta_path,
            chunk_length=args.chunk_length,
            step_length=args.step_length,
            include_partial=not args.full_windows_only,
        )
    )
    if not windows:
        raise ValueError(f"FASTA file produced no windows: {fasta_path}")
    return windows


def _resolve_layer(model_name: str, layer_name: str | None) -> str:
    if layer_name:
        return layer_name
    try:
        return DEFAULT_LAYERS[model_name]
    except KeyError as error:
        raise ValueError(
            f"--layer-name is required for model without a frozen default: {model_name}"
        ) from error


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fasta_files = discover_fasta_files(args.input)
    layer_name = _resolve_layer(args.model_name, args.layer_name)

    if args.plan_only:
        file_summaries = []
        for fasta_path in fasta_files:
            windows = _windows_for_file(fasta_path, args)
            file_summaries.append(
                {
                    "name": fasta_path.name,
                    "record_count": len({window.record_index for window in windows}),
                    "window_count": len(windows),
                }
            )
        summary = {
            "schema_version": 1,
            "model_name": args.model_name,
            "layer_name": layer_name,
            "chunk_length": args.chunk_length,
            "step_length": args.step_length,
            "include_partial": not args.full_windows_only,
            "window_indexing_contract": WINDOW_INDEXING_CONTRACT,
            "file_count": len(file_summaries),
            "record_count": sum(item["record_count"] for item in file_summaries),
            "window_count": sum(item["window_count"] for item in file_summaries),
        }
        if args.plan_detail == "files":
            summary["files"] = file_summaries
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    from evo2 import Evo2

    model = Evo2(args.model_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fasta_path in fasta_files:
        windows = _windows_for_file(fasta_path, args)
        tensor_path = args.output_dir / f"{fasta_path.stem}.pt"
        manifest_path = args.output_dir / f"{fasta_path.stem}.manifest.json"
        if not args.overwrite and (tensor_path.exists() or manifest_path.exists()):
            raise FileExistsError(
                f"output exists; pass --overwrite to replace: {tensor_path}"
            )

        tensor = extract_window_embeddings(
            windows,
            model=model,
            tokenizer=model.tokenizer,
            layer_name=layer_name,
            batch_size=args.batch_size,
            input_device=args.input_device,
        )
        _atomic_torch_save(tensor, tensor_path)
        manifest = {
            "schema_version": 1,
            "source_fasta": fasta_path.name,
            "source_sha256": sha256_file(fasta_path),
            "model_name": args.model_name,
            "layer_name": layer_name,
            "chunk_length": args.chunk_length,
            "step_length": args.step_length,
            "include_partial": not args.full_windows_only,
            "window_indexing_contract": WINDOW_INDEXING_CONTRACT,
            "pooling": "valid_token_mean",
            "record_count": len({window.record_index for window in windows}),
            "window_count": len(windows),
            "tensor_file": tensor_path.name,
            "tensor_sha256": sha256_file(tensor_path),
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "windows": [window.provenance() for window in windows],
        }
        _atomic_json_dump(manifest, manifest_path)
        print(json.dumps({"tensor": str(tensor_path), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
