"""Evo 2 activation extraction and length-aware pooling."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .windowing import FastaWindow


def prepare_token_batch(
    sequences: Sequence[str],
    tokenizer: object,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Tokenize and right-pad non-empty DNA sequences."""

    if not sequences:
        raise ValueError("cannot prepare an empty sequence batch")
    lengths = [len(sequence) for sequence in sequences]
    if any(length <= 0 for length in lengths):
        raise ValueError("all sequences must be non-empty")

    max_length = max(lengths)
    rows = []
    for sequence in sequences:
        token_ids = list(tokenizer.tokenize(sequence))
        if len(token_ids) != len(sequence):
            raise ValueError("tokenizer must emit exactly one token per nucleotide")
        token_ids.extend([tokenizer.pad_id] * (max_length - len(token_ids)))
        rows.append(token_ids)
    return torch.tensor(rows, dtype=torch.long, device=device), lengths


def mean_pool_valid_tokens(
    activations: torch.Tensor,
    lengths: Sequence[int],
) -> torch.Tensor:
    """Mean-pool ``[batch, length, hidden]`` activations over valid tokens."""

    if activations.ndim != 3:
        raise ValueError("activations must have shape [batch, length, hidden]")
    if activations.shape[0] != len(lengths):
        raise ValueError("activation batch dimension does not match lengths")
    if any(length <= 0 or length > activations.shape[1] for length in lengths):
        raise ValueError("lengths must be within the activation sequence dimension")

    length_tensor = torch.as_tensor(lengths, device=activations.device)
    mask = torch.arange(activations.shape[1], device=activations.device).unsqueeze(
        0
    ) < length_tensor.unsqueeze(1)
    masked = activations * mask.unsqueeze(-1)
    return masked.sum(dim=1) / length_tensor.to(activations.dtype).unsqueeze(1)


def extract_window_embeddings(
    windows: Sequence[FastaWindow],
    *,
    model: object,
    tokenizer: object,
    layer_name: str,
    batch_size: int,
    input_device: str | torch.device,
) -> torch.Tensor:
    """Extract one pooled activation vector per ordered FASTA window."""

    if not windows:
        raise ValueError("cannot extract embeddings for an empty window collection")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    pooled_batches = []
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            input_ids, lengths = prepare_token_batch(
                [window.sequence for window in batch],
                tokenizer,
                device=input_device,
            )
            _, embedding_map = model(
                input_ids,
                return_embeddings=True,
                layer_names=[layer_name],
            )
            if layer_name not in embedding_map:
                raise KeyError(f"model did not return requested layer: {layer_name}")
            pooled = mean_pool_valid_tokens(embedding_map[layer_name], lengths)
            pooled_batches.append(pooled.detach().cpu())

    return torch.cat(pooled_batches, dim=0)
