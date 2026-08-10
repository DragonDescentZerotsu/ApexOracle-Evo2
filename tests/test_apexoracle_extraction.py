import pytest
import torch

from apexoracle_evo2.extraction import (
    extract_window_embeddings,
    mean_pool_valid_tokens,
    prepare_token_batch,
)
from apexoracle_evo2.windowing import FastaWindow


class FakeTokenizer:
    pad_id = 0

    def tokenize(self, sequence: str) -> list[int]:
        return [{"A": 1, "C": 2, "G": 3, "T": 4}[base] for base in sequence]


class FakeModel:
    def __call__(self, input_ids, *, return_embeddings, layer_names):
        assert return_embeddings is True
        layer = layer_names[0]
        values = input_ids.to(torch.float32)
        activations = torch.stack((values, values * 10), dim=-1)
        return None, {layer: activations}


def make_window(sequence: str, window_index: int) -> FastaWindow:
    return FastaWindow(
        record_id="record",
        record_index=0,
        record_window_index=window_index,
        window_index=window_index,
        start=window_index,
        end=window_index + len(sequence),
        record_length=20,
        sequence=sequence,
        is_partial=False,
    )


def test_prepare_batch_and_length_aware_mean_pooling() -> None:
    input_ids, lengths = prepare_token_batch(
        ["ACG", "TT"], FakeTokenizer(), device="cpu"
    )
    assert input_ids.tolist() == [[1, 2, 3], [4, 4, 0]]
    assert lengths == [3, 2]

    activations = torch.stack(
        (input_ids.to(torch.float32), input_ids.to(torch.float32) * 10), dim=-1
    )
    pooled = mean_pool_valid_tokens(activations, lengths)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 20.0], [4.0, 40.0]]))


def test_extract_preserves_window_order_across_batches() -> None:
    windows = [make_window("ACG", 0), make_window("TT", 1), make_window("A", 2)]
    tensor = extract_window_embeddings(
        windows,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        layer_name="layer",
        batch_size=2,
        input_device="cpu",
    )
    torch.testing.assert_close(
        tensor, torch.tensor([[2.0, 20.0], [4.0, 40.0], [1.0, 10.0]])
    )


def test_invalid_pooling_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="within"):
        mean_pool_valid_tokens(torch.ones(1, 2, 3), [3])
