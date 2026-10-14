"""Tests for SFT loss masking and batch padding."""

from common.lm_data import IGNORE_INDEX, masked_labels, pad_batch


def test_masked_labels_masks_prompt_only():
    assert masked_labels([5, 6, 7], [5, 6, 7, 8, 9]) == [IGNORE_INDEX] * 3 + [8, 9]


def test_masked_labels_same_length_as_inputs_when_truncated():
    # prompt longer than the truncated full sequence -> everything masked, length preserved
    labels = masked_labels([1, 2, 3, 4], [1, 2])
    assert labels == [IGNORE_INDEX, IGNORE_INDEX]


def test_pad_batch_pads_to_longest():
    batch = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4], "labels": [4]}]
    out = pad_batch(batch, pad_token_id=0)
    assert out["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert out["labels"] == [[-100, 2, 3], [4, -100, -100]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
